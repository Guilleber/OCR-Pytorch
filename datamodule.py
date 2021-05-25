import pytorch_lightning as pl

import torch
from torch.utils.data import DataLoader, Dataset

import jsonlines
from PIL import Image
import random
import numpy as np

from dataset import OCRDataset


class Tokenizer:
    def ids2tokens(self, ids):
        return [[self.vocab[idx] for idx in seq] for seq in ids]

    def decode(self, ids):
        tokens = self.ids2tokens(ids)
        return self.detokenize(tokens)

    def tokens2ids(self, tokens):
        return [[self.token_to_idx[token] for token in seq] for seq in tokens]

    def encode(self, sentences):
        tokens = self.tokenize(sentences)
        return self.tokens2ids(tokens)


class CharTokenizer(Tokenizer):
    def __init__(self, hparams):
       self.vocab = ["GO", "END"] + list(open("./ressources/charset.txt", 'r').read())
       self.token_to_id = {token: i for i, token in enumerate(self.vocab)}

    def tokenize(self, sentences):
        return [list(seq) for seq in sentences]

    def detokenize(self, tokens):
        return [''.join(seq) for seq in tokens]
