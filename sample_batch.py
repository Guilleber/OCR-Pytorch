from datamodule import CharTokenizer, OCRDataModule
from argparse import Namespace
import sys

tokenizer = CharTokenizer()

sentences = ["azerty", "/99 45*.", "a"]

ids, mask = tokenizer.encode(sentences)
print(tokenizer.decode(ids))

hparams = {'grayscale': False,
           'resize': False,
           'width': 100,
           'height': 32,
           'bs': 16}

datamodule = OCRDataModule(Namespace(**hparams), [sys.argv[1]], tokenizer)
datamodule.setup(stage='test')
dataloader = datamodule.test_dataloader()

batch = next(iter(dataloader))
print(tokenizer.decode(batch['tgt'].numpy()))
print([(key, batch[key].size()) for key in batch])
