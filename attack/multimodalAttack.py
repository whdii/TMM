import torch
from torchvision import transforms
import numpy as np

images_normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

def equal_normalize(x):
    return x

gx = 1
cx = []
class MultiModalAttacker():
    def __init__(self, net, text_attacker, tokenizer, cls=True, *args, **kwargs):
        self.net = net
        self.text_attacker = text_attacker
        self.tokenizer = tokenizer
        self.cls = cls
        if hasattr(text_attacker, 'sample_batch_size'):
            self.sample_batch_size = text_attacker.sample_batch_size
        
        self.repeat = 1

    def get_origin_and_adv_embeds(self, images, text, device, max_length, k):
        with torch.no_grad():
            text_input = self.tokenizer(text * self.repeat, padding='max_length', truncation=True, max_length=max_length,
                                        return_tensors="pt").to(device)
            origin_output = self.net.inference(self.image_normalize(images).repeat(self.repeat, 1, 1, 1),
                                               text_input, use_embeds=False)
            if self.cls:
                origin_embeds = origin_output['fusion_output'][:, 0, :].detach()
            else:
                origin_embeds = origin_output['fusion_output'].flatten(1).detach()

        with torch.no_grad():
            text_adv = self.text_attacker.attack(self.net, images, text, k)
            text_adv_input = self.tokenizer(text_adv, padding='max_length', truncation=True, max_length=max_length,
                                        return_tensors="pt").to(device)
            text_adv_output = self.net.inference(self.image_normalize(images).repeat(self.repeat, 1, 1, 1),
                                                    text_adv_input, use_embeds=False)
            if self.cls:
                text_adv_embed = text_adv_output['fusion_output'][:, 0, :].detach()
            else:
                text_adv_embed = text_adv_output['fusion_output'].flatten(1).detach()

        return origin_embeds, text_adv_embed, text_input, origin_output, text_adv_input, text_adv
    
    def getAtt(self, attImage, text_input, device, indices, args):
        
        image_embeds = self.net.visual_encoder(images_normalize(attImage))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
        output = self.net.text_encoder(text_input.input_ids,
                            attention_mask=text_input.attention_mask,
                            encoder_hidden_states=image_embeds,
                            encoder_attention_mask=image_atts,
                            return_dict=True,
                            )
        vl_embeddings = output.last_hidden_state[:, 0, :]
        vl_output = self.net.itm_head(vl_embeddings)
        loss_pre = vl_output[:, 1].sum()
        self.net.zero_grad()
        loss_pre.backward()
        
        with torch.no_grad():
            cam = []
            mask_att = text_input.attention_mask.view(text_input.attention_mask.size(0), 1, -1, 1, 1)
            for blk in range(6, len(self.net.text_encoder.base_model.base_model.encoder.layer)):
                grads = self.net.text_encoder.base_model.base_model.encoder.layer[
                    blk].crossattention.self.get_attn_gradients().detach()
                cams = self.net.text_encoder.base_model.base_model.encoder.layer[
                    blk].crossattention.self.get_attention_map().detach()
                cams = cams[:, :, :, 1:].reshape(attImage.size(0), 12, -1, 24, 24) * mask_att
                grads = grads[:, :, :, 1:].clamp(min=0).reshape(attImage.size(0), 12, -1, 24, 24) * mask_att
                gradcam = cams * grads
                gradcam = gradcam.mean(1).mean(1)
                cam.append(gradcam)
            gradcam = 0
            for i in range(len(cam)):
                gradcam += cam[i]
            gradcam = gradcam/attImage.size(0)
            gradcam = gradcam.permute(1, 2, 0)
            gradcam = self.getGradCam(gradcam, attImage[0].shape[1:3])
            gradcam = gradcam.transpose(2, 0, 1)
        
        attMap = np.add.reduceat(gradcam, [0, indices[0]])[::2]
        attMap = (attMap / 5).squeeze()
        attMap = torch.repeat_interleave(torch.tensor(attMap).unsqueeze(0).unsqueeze(0), repeats=3, dim=1)
        
        return attMap

    
    def run_transfer_attack(self, images, text, args, num_iters, k=10, max_length=30):
        device = images.device
        origin_embeds, text_adv_embed, text_input, origin_output, text_adv_input, text_adv = self.get_origin_and_adv_embeds(images, text, device, max_length)
        loss_fn = torch.nn.CosineEmbeddingLoss()

        indices = torch.linspace(4, images.shape[0]-1, int(images.shape[0]/5), dtype=int).cpu()
        attImage = images.index_select(0, indices.to(device))
        attMap = self.getAtt(attImage, text_input, device, indices, args)
        image_attack = self.image_attacker.attack(attImage, attMap, num_iters)

        for i in range(num_iters):
            image_diversity = next(image_attack)
            adv = [image_diversity[i].repeat(5, 1, 1, 1) for i in range(image_diversity.shape[0])]
            adv = torch.cat(adv, dim=0)
            _, _, _, _, text_adv_input, text_adv = self.get_origin_and_adv_embeds(adv, text, device, max_length)
            adv_output = self.net.inference(adv, text_adv_input, use_embeds=False)
            vdt = self.net.inference(adv, text_input, use_embeds=False)
            text_adv_output = self.net.inference(self.image_normalize(images).repeat(self.repeat, 1, 1, 1),
                                                    text_adv_input, use_embeds=False)

            if args.cls:
                adv_embed = adv_output['fusion_output'][:, 0, :]
                vdt_embed = vdt['fusion_output'][:, 0, :]
                text_adv_embed = text_adv_output['fusion_output'][:, 0, :].detach()
            else:
                adv_embed = adv_output['fusion_output'].flatten(1)
                vdt_embed = vdt['fusion_output'].flatten(1)
                text_adv_embed = text_adv_output['fusion_output'].flatten(1).detach()
                
            y = torch.ones(adv_embed.shape[0]).to(device)
            cos_ao_loss_nosoft = loss_fn(adv_embed, origin_embeds, y)
            cos_ot_nosoft = loss_fn(origin_embeds, text_adv_embed, y)
            cos_ov_nosoft = loss_fn(origin_embeds, vdt_embed, y)
            adv_origin_img_ssim_loss = (1 - self.ssim(adv[0].unsqueeze(0), images[0].unsqueeze(0), data_range=1.0))
            loss = cos_ao_loss_nosoft + cos_ot_nosoft + cos_ov_nosoft + 10 * adv_origin_img_ssim_loss
            loss.backward()

        images_adv = next(image_attack)
        images_adv = [images_adv[i].repeat(5, 1, 1, 1) for i in range(images_adv.shape[0])]
        images_adv = torch.cat(images_adv, dim=0)
        _, _, _, _, _, text_adv = self.get_origin_and_adv_embeds(adv, text, device, max_length)

        return images_adv, text_adv
