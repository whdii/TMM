import torch
import torch.nn as nn
import copy
from transformers import BatchEncoding
import torch.nn.functional as F

filter_words = set(['a', '.', '-', 'a the', '/', '?', '"', ',', 'b', '&', '!',
                '@', '%', '^', '*', '(', ')', "-", '-', '+', '=', '<', '>', '|', ':', ";", '～', '·'])

class TextAttacker():
    def __init__(self, ref_net, tokenizer, device, cls=True):
        self.ref_net = ref_net
        self.tokenizer = tokenizer
        self.cls = cls
        self.device = device
        self.cosSimilatity = torch.nn.CosineEmbeddingLoss()

    def attack(self, net, images, texts, k=10, num_perturbation=1, threshold_pred_score=0.3, max_length=30, batch_size=32):
        text_inputs = self._prepare_text_inputs(texts, max_length)
        origin_output = net.inference(images, text_inputs)
        origin_embeds = self._get_origin_embeds(origin_output)
        adv_texts = self._generate_adv_texts(origin_output, texts, origin_embeds, net, images, num_perturbation, max_length)
        return adv_texts

    def _prepare_text_inputs(self, texts, max_length):
        text_inputs = self.tokenizer(texts, padding='max_length', truncation=True, max_length=max_length,
                                     return_tensors='pt').to(self.device)
        return text_inputs

    def _get_origin_embeds(self, origin_output):
        if self.cls:
            origin_embeds = origin_output['fusion_output'][:, 0, :].detach()
        else:
            origin_embeds = origin_output['fusion_output'].flatten(1).detach()
        return origin_embeds

    def _generate_adv_texts(self, origin_output, texts, origin_embeds, net, images, num_perturbation, max_length):
        adv_texts = []
        for i in range(len(texts)):
            text = texts[i]
            weights = origin_output['weights'][i]
            sorted_weights, sorted_indices = torch.sort(weights, descending=True)

            words, sub_words, keys = self._tokenize(text)
            final_words = copy.deepcopy(words)
            change = 0

            for top_index in sorted_indices:
                if change >= num_perturbation:
                    break
                if top_index > len(words) - 1:
                    continue
                tgt_word = words[top_index]
                if tgt_word in filter_words:
                    continue
                if keys[top_index][0] > max_length - 2:
                    continue
                
                mask_text = [self.replace_word_with_mask(text, top_index)]
                tokenized_text = self.tokenizer(mask_text)
                input_ids = torch.LongTensor(tokenized_text['input_ids']).to(self.device)
                outputs = self.ref_net(input_ids).logits
                mask_position = tokenized_text['input_ids'][0].index(self.tokenizer.mask_token_id)
                pred_value, predicted_index = torch.topk(outputs[0, mask_position], 10)
                substitutes = self._get_substitutes(predicted_index)

                replace_texts, available_substitutes = self._get_replace_texts_and_available_substitutes(substitutes, final_words, top_index, tgt_word)
                
                replace_text_input = self._prepare_text_inputs(replace_texts, max_length)

                adv_output = net.inference(images[0].unsqueeze(0).repeat(len(replace_texts), 1, 1, 1), replace_text_input)
        
                adv_embeds = self._get_adv_embeds(adv_output)

                loss = self._calculate_loss(origin_embeds, adv_embeds, i, len(replace_texts))
                
                candidate_idx = loss.argmax()

                final_words[top_index] = available_substitutes[candidate_idx]

                if available_substitutes[candidate_idx] != tgt_word:
                    change += 1
        
            adv_texts.append(' '.join(final_words))

        return adv_texts

    def _get_substitutes(self, predicted_index):
        substitutes = []
        for j in range(predicted_index.shape[0]):
            substitutes.append(self.tokenizer.decode(predicted_index[j]).replace(' ', ''))
        return substitutes

    def _get_replace_texts_and_available_substitutes(self, substitutes, final_words, top_index, tgt_word):
        replace_texts = []
        available_substitutes = [tgt_word]
        for substitute_ in substitutes:
            substitute = substitute_

            if substitute == tgt_word:
                continue

            if '##' in substitute:
                substitute = substitute.replace('##', '')

            if substitute in filter_words:
                continue

            temp_replace = copy.deepcopy(final_words)
            temp_replace[top_index] = substitute
            available_substitutes.append(substitute)
            replace_texts.append(' '.join(temp_replace))
        return replace_texts, available_substitutes

    def _get_adv_embeds(self, adv_output):
        if self.cls:
            adv_embeds = adv_output['fusion_output'][:, 0, :].detach()
        else:
            adv_embeds = adv_output['fusion_output'].flatten(1).detach()
        return adv_embeds

    def _calculate_loss(self, origin_embeds, adv_embeds, i, len_replace_texts):
        loss = []
        for j in range(len_replace_texts):
            y = torch.ones(1).to(self.device)
            lossss = self.cosSimilatity(origin_embeds[i].unsqueeze(0), adv_embeds[j].unsqueeze(0), y)
            loss.append(lossss)
        loss = torch.stack((loss))
        return loss

    def replace_word_with_mask(self, text, position):
        words = text.split()
        if 0 <= position < len(words):
            words[position] = '[MASK]'
        return ' '.join(words)

    def _tokenize(self, text):
        words = text.split(' ')

        sub_words = []
        keys = []
        index = 0
        for word in words:
            sub = self.tokenizer.tokenize(word)
            sub_words += sub
            keys.append([index, index + len(sub)])
            index += len(sub)

        return words, sub_words, keys