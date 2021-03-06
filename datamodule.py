import pytorch_lightning as pl

import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset

import jsonlines
from PIL import Image
import random
import numpy as np
import os

from typing import Iterable, List, Tuple, Optional

from dataset import OCRDataset


class Tokenizer:
    def tokenize(self, sentences: Iterable[str]) -> List[List[str]]:
        raise NotImplementedError()

    def detokenize(self, tokens: Iterable[Iterable[str]]) -> List[str]:
        raise NotImplementedError()

    def ids2tokens(self, ids: Iterable[Iterable[int]]) -> List[List[str]]:
        return [[self.vocab[idx] for idx in seq if idx not in [self.go_token_idx, self.end_token_idx]] for seq in ids]

    def decode(self, ids: Iterable[Iterable[int]]) -> List[str]:
        tokens = self.ids2tokens(ids)
        return self.detokenize(tokens)

    def tokens2ids(self, tokens: Iterable[Iterable[str]]) -> List[List[int]]:
        return [[self.token_to_idx[token] if token in self.token_to_idx else self.unk_token_idx for token in seq] for seq in tokens]

    def pad(self, ids: Iterable[Iterable[int]]) -> Tuple[np.ndarray, np.ndarray]:
        max_len = max([len(seq) for seq in ids])
        max_len = min(max_len, 50)
        bs = len(ids)
        padded_ids = self.end_token_idx * np.ones((bs, max_len), dtype=np.int8)
        mask = np.ones((bs, max_len), dtype=bool)
        for i in range(bs):
            for j in range(len(ids[i]))[:max_len]:
                padded_ids[i, j] = ids[i][j]
                mask[i][j] = False
        return padded_ids, mask

    def encode(self, sentences: Iterable[str]) -> Tuple[np.ndarray, np.ndarray]:
        tokens = self.tokenize(sentences)
        ids = self.tokens2ids(tokens)
        return self.pad(ids)


class CharTokenizer(Tokenizer):
    def __init__(self, case_sensitive: Optional[bool] = False):
        self.case_sensitive = case_sensitive
        self.vocab = ["GO", "END", "UNK"] + list(open("./ressources/charset.txt" if self.case_sensitive else "./ressources/charset_cap_only.txt", 'r').read())
        self.token_to_idx = {token: i for i, token in enumerate(self.vocab)}
        self.go_token_idx = self.token_to_idx["GO"]
        self.end_token_idx = self.token_to_idx["END"]
        self.unk_token_idx = self.token_to_idx["UNK"]
        self.vocab_size = len(self.vocab)

    def tokenize(self, sentences: Iterable[str]) -> List[List[str]]:
        return [list(seq) if self.case_sensitive else list(seq.upper()) for seq in sentences]

    def detokenize(self, tokens: Iterable[Iterable[str]]) -> List[str]:
        return [''.join(seq) for seq in tokens]


def normalize_imgs(imgs):
    return [(img/255.) - 0.5 for img in imgs]


def pad_imgs(imgs):
    bs = len(imgs)
    shapes = [img.shape for img in imgs]
    channels = shapes[0][2]
    max_h = max([s[0] for s in shapes])
    max_w = max([s[1] for s in shapes])
    padded_imgs = np.zeros((bs, channels, max_h, max_w), dtype=np.float64)
    mask = np.ones((bs, max_h, max_w), dtype=bool)
    for n in range(bs):
        for h in range(shapes[n][0]):
            for w in range(shapes[n][1]):
                mask[n, h, w] = False
                for c in range(channels):
                    padded_imgs[n, c, h, w] = imgs[n][h, w, c]
    return padded_imgs, mask


class OCRDataModule(pl.LightningDataModule):
    def __init__(self, hparams, datasets_paths, tokenizer=None):
        super().__init__()
        self.hpar = hparams
        if tokenizer is None:
            self.tokenizer = CharTokenizer()
        else:
            self.tokenizer = tokenizer
        self.datasets_paths = datasets_paths

    def collate_fn(self, batch):
        imgs = [el['raw_img'] for el in batch if el is not None]
        tgts = [el['raw_label'] for el in batch if el is not None]
        imgs = normalize_imgs(imgs)
        imgs, img_mask = pad_imgs(imgs)
        tgts, tgt_mask = self.tokenizer.encode(tgts)

        return {'img': torch.from_numpy(imgs).float(),
                'img_padding_mask': torch.from_numpy(img_mask),
                'tgt': torch.from_numpy(tgts).long(),
                'tgt_padding_mask': torch.from_numpy(tgt_mask)}

    def setup(self, stage=None):
        if stage in (None, 'fit'):
            self.train_datasets = [OCRDataset(path, self.hpar, is_train=True) for path in self.datasets_paths['train']]
            self.val_datasets = [OCRDataset(path, self.hpar, is_train=False) for path in self.datasets_paths['val']]

        if stage == 'validate':
            self.val_datasets = [OCRDataset(path, self.hpar, is_train=False) for path in self.datasets_paths['val']]

        if stage in (None, 'test'):
            self.test_datasets = [OCRDataset(path, self.hpar, is_train=False) for path in self.datasets_paths['test']]

    def train_dataloader(self):
        return DataLoader(ConcatDataset(self.train_datasets), batch_size=self.hpar.bs, shuffle=True, collate_fn=self.collate_fn, num_workers=self.hpar.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(ConcatDataset(self.val_datasets), batch_size=self.hpar.bs, collate_fn=self.collate_fn, num_workers=self.hpar.num_workers)

    def val_dataloaders(self):
        return [DataLoader(dataset, batch_size=self.hpar.bs, collate_fn=self.collate_fn, num_workers=self.hpar.num_workers) for dataset in self.val_datasets]

    def test_dataloader(self):
        return DataLoader(ConcatDataset(self.test_datasets), batch_size=self.hpar.bs, collate_fn=self.collate_fn, num_workers=self.hpar.num_workers)

    def test_dataloaders(self):
        return [DataLoader(dataset, batch_size=self.hpar.bs, collate_fn=self.collate_fn, num_workers=self.hpar.num_workers) for dataset in self.test_datasets]
