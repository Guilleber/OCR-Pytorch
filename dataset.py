import torch
from torch.utils.data import DataLoader, Dataset
import sys

import jsonlines
from PIL import Image, ImageFile, ImageDraw, ImageFilter
import random
import numpy as np
import math
from typing import Dict, Optional, Tuple

ImageFile.LOAD_TRUNCATED_IMAGES = False
LIMITS = [2, 4, 6, 8, 10]


class OCRDataset(Dataset):
    def __init__(self, path: str, hparams: Dict, is_train: Optional[bool] = True):
        super().__init__()

        self.length_limit = hparams.length_limit

        path = path.split(':')
        self.length = None
        self.nb_repeat = 1
        for opt in path[:-1]:
            if opt[0] == '*':
                self.nb_repeat = int(opt[1:])
            elif opt[0] == '!':
                self.length = int(opt[1:])
        path = path[-1]

        self.hparams = hparams
        self.resize = [self.hparams.width, self.hparams.height] if self.hparams.resize else None
        self.data = list(jsonlines.open(path, 'r'))
        self.folder_path = '/'.join(path.split('/')[:-1]) + '/'
        self.is_train = is_train

        self.length = len(self.data) if self.length is None else self.length
        self.input_map = []
        for i, item in enumerate(self.data[:self.length]):
            if len(item['tag']) <= self.length_limit:
                self.input_map.append(i)
        self.length = len(self.input_map)


    @staticmethod
    def load_and_transform(img_path: str, crop: Optional[Dict] = None,
                           resize: Optional[Tuple[int, int]] = None, is_train: Optional[bool] = False,
                           grayscale: Optional[bool] = True, augmentation: Optional[str] = 'none') -> np.ndarray:
        img = Image.open(img_path)

        """draw = ImageDraw.Draw(img)
        print(crop)
        print(img_path)
        draw.rectangle((crop['x'], crop['y'], crop['x'] + crop['width'], crop['y'] + crop['height']), outline='red')
        img.save("../../test.png")"""

        if grayscale:
            img = img.convert('L')
            channel = 1
        else:
            img = img.convert('RGB')
            channel = 3

        if crop is not None:
            img = img.crop((crop['x'], crop['y'], crop['x'] + crop['width'], crop['y'] + crop['height']))

        if is_train and augmentation != 'none':
            if augmentation == 'simple':
                # Data augmentation -> applies random rotation to the image
                angle = random.randint(-10, 10)
                img = img.rotate(angle, resample=Image.BILINEAR, expand=True, fillcolor='white')
            elif augmentation == 'funsd':
                #rotations
                angle = random.randint(-10, 10)
                img = img.rotate(angle, resample=Image.BILINEAR, expand=True, fillcolor='white')

                #distortions + white borders
                w, h = img.size
                new_w, new_h = random.randint(w, w+30), random.randint(h, int(1.2*h))
                new_img = Image.new('L' if grayscale else 'RGB', (new_w, new_h), color='white')
                new_img.paste(img, (random.randint(0, new_w-w), random.randint(0, new_h-h)))
                img = new_img

        if resize is not None:
            if resize[0] == -1:
                aspect_ratio = img.size[0]/img.size[1]
                resize[0] = int(aspect_ratio * resize[1])
                resize[0] = (min(resize[0], 200)//4 + 1)*4
            img = img.resize(resize, Image.BICUBIC)

        w, h = img.size

        img = np.array(img, dtype=np.uint8)
        img = np.reshape(img, (h, w, channel))
        return img

    def __len__(self):
        return self.length*self.nb_repeat

    def __getitem__(self, index: int) -> Dict:
        try:
            try:
                index = index%self.length
                index = self.input_map[index]
                raw_img = OCRDataset.load_and_transform(self.folder_path + self.data[index]['img'],
                                                    crop=self.data[index]['box'],
                                                    resize=self.resize,
                                                    is_train=self.is_train,
                                                    grayscale=self.hparams.grayscale,
                                                    augmentation = self.hparams.augmentation)
                raw_label = self.data[index]['tag']
                if len(raw_label) == 0:
                    raw_label = '*'
                return {'raw_img': raw_img, 'raw_label': raw_label}
            except:
                index = random.randint(0, self.length-1)
                index = self.input_map[index]
                raw_img = OCRDataset.load_and_transform(self.folder_path + self.data[index]['img'],
                                                    crop=self.data[index]['box'],
                                                    resize=self.resize,
                                                    is_train=self.is_train,
                                                    grayscale=self.hparams.grayscale,
                                                    augmentation = self.hparams.augmentation)
                raw_label = self.data[index]['tag']
                if len(raw_label) == 0:
                    raw_label = '*'
                return {'raw_img': raw_img, 'raw_label': raw_label}
        except Exception as e:
            # If anything happens during loading the example is set to 'None' and will be eliminated by the 'collate_fn'
            # function in datamodule.py
            print(e, file=sys.stderr, flush=True)
            return None
