import json
import os
import random

from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from dataset.utils import pre_caption

class pair_dataset_attack(Dataset):
    def __init__(self, ann_file, transform, image_root, args, max_words=30):
        self.ann = json.load(open(ann_file, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words

        self.text = []
        self.image = []
        

        self.txt2img = {}
        self.img2txt = {}
        self.image_ids = {}

        txt_id = 0
        for i, ann in enumerate(self.ann):
            self.img2txt[i] = []
            for j, caption in enumerate(ann['caption']):
                self.image.append(ann['image'])
                self.text.append(pre_caption(caption, self.max_words))
                self.txt2img[txt_id] = i
                if args.dataset == 'flickr':
                    self.image_ids[txt_id] = ann['image'].split('/')[1].split('.')[0]
                elif args.dataset == 'mscoco':
                    self.image_ids[txt_id] = ann['image'].split('/')[1].split('.')[0].split('_')[2]
                if j == 5:
                    print(ann['image'].split('/')[1].split('.')[0])
                self.img2txt[i].append(txt_id)
                txt_id += 1

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_root, self.image[index])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        text = self.text[index]
        image_id = self.image_ids[index]

        return image, text, index, image_id