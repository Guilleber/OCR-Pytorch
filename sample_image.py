import sys
from argparse import Namespace
import random
from PIL import Image
import numpy as np

from dataset import OCRDataset

from parameters import datasets

hparams = {'grayscale': True,
           'resize': False,
           'width': 100,
           'height': 32}

dataset = OCRDataset(datasets[sys.argv[1]][0] + 'train.jsonl', Namespace(**hparams), is_train=False)

sample = dataset[random.randint(0, len(dataset)-1)]
print(sample['raw_label'])

img = sample['raw_img']
h, w, c = np.shape(img)

if hparams['grayscale']:
    img = np.reshape(img, (h, w))
    img = Image.fromarray(img, 'L')
else:
    img = Image.fromarray(img)
img.save("/home/gleberre/sample.jpg")
