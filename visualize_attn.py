import argparse
import json
import torch
from PIL import Image
import numpy as np
from matplotlib import cm

from datamodule import CharTokenizer
from dataset import OCRDataset
from model import SATRNModel


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('load_weights_from', type=str)
    parser.add_argument('img_path', type=str)
    
    parser.add_argument('--height', type=int, default=32)
    parser.add_argument('--width', type=int, default=100)

    parser.add_argument('--grayscale', help="transform images to grayscale", action='store_true')
    parser.add_argument('--resize', help="resize images to [--width] x [--height]", action='store_true')
    parser.add_argument('--case_sensitive', action='store_true')

    parser.add_argument('--crop', type=json.loads, default=None)

    args = parser.parse_args()

    tokenizer = CharTokenizer(case_sensitive=args.case_sensitive)
    model = SATRNModel.load_from_checkpoint(args.load_weights_from, tokenizer=tokenizer)

    resize = (args.width, args.height) if args.resize else None
    image = OCRDataset.load_and_transform(args.img_path, crop=args.crop, resize=resize, is_train=False, grayscale=args.grayscale)

    img = torch.from_numpy(image).long()
    img = img.permute(2, 0, 1)
    img = img.unsqueeze(0)
    img = (img/255.) - 0.5

    img_padding_mask = torch.zeros((1, img.size(2), img.size(3)), dtype=bool)

    with torch.no_grad():
        pred, attn_weights = model.generate(img, img_padding_mask)
        pred = pred.numpy()
        attn_weights = attn_weights.numpy()
    print(tokenizer.decode(pred))

    image = Image.fromarray(image.squeeze(), mode='L') if args.grayscale else Image.fromarray(image, mode='RGB')
    image = image.convert('RGBA')

    for layer in range(len(attn_weights)):
        for token in range(len(attn_weights[layer][0])):
            heat_map = Image.fromarray(np.uint8(cm.rainbow(attn_weights[layer][0][token]*50)*255))
            heat_map = heat_map.convert('RGBA')
            heat_map = heat_map.resize(image.size, resample=Image.BICUBIC)
            heat_map = Image.blend(image, heat_map, alpha=0.5)
            heat_map.save('./imgs/attn_heatmaps/layer={}_token={}.png'.format(layer, token))
