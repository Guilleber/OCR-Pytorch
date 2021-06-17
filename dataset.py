import torch
from torch.utils.data import DataLoader, Dataset

import jsonlines
from PIL import Image, ImageFile
import random
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True


class OCRDataset(Dataset):
    def __init__(self, path, hparams, is_train=True):
        super().__init__()
        self.hparams = hparams
        self.data = list(jsonlines.open(path, 'r'))
        self.folder_path = '/'.join(path.split('/')[:-1]) + '/'
        self.is_train = is_train

    def load_and_transform(self, img_path, crop=None):
        img = Image.open(img_path)

        if self.hparams.grayscale:
            img = img.convert('L')
            channel = 1
        else:
            img = img.convert('RGB')
            channel = 3

        if crop is not None:
            img = img.crop((crop['x'], crop['y'], crop['x'] + crop['width'], crop['y'] + crop['height']))

        if self.hparams.resize:
            img = img.resize((self.hparams.width, self.hparams.height), Image.BICUBIC)

        if self.is_train:
            angle = random.randint(-34, 34)
            img = img.rotate(angle)

        w, h = img.size

        img = np.array(img, dtype=np.uint8)
        img = np.reshape(img, (h, w, channel))
        return img

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            raw_img = self.load_and_transform(self.folder_path + self.data[index]['img'], crop=self.data[index]['box'])
            raw_label = self.data[index]['tag']
            return {'raw_img': raw_img, 'raw_label': raw_label}
        except:
            return None
