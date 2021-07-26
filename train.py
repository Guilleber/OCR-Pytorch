import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin

import torch

import argparse
from datetime import datetime
import sys

from datamodule import OCRDataModule, CharTokenizer
from model import SATRNModel
import parameters


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_type', type=str, help="Model type: must match one in parameters.py")
    parser.add_argument('-d', '--datasets', type=str, help="Datasets to use for training. Must match one in parameters.py")
    parser.add_argument('--exp_name', type=str, default='exp')
    parser.add_argument('--load_weights_from', type=str, default=None)
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--augmentation', type=str, default= 'none', help="none, simple or funsd")
    
    parser.add_argument('--bs', type=int, help="mini-batch size", default=32)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--height', type=int, default=32, help="Height to which the image are resized. Ignored if [--resize] is not used.")
    parser.add_argument('--width', type=int, default=100, help="Width to which the image are resized. Ignored if [--resize] is not used.")

    parser.add_argument('--lr', type=float, help="learning rate", default=3e-4)

    parser.add_argument('--grayscale', help="transform images to grayscale", action='store_true')
    parser.add_argument('--resize', help="resize images to [--width] x [--height]", action='store_true')
    parser.add_argument('--save_best_model', action='store_true')
    parser.add_argument('--run_test', action='store_true')
    parser.add_argument('--run_val', action='store_true')
    parser.add_argument('--case_sensitive', action='store_true')

    args = parser.parse_args()

    args.lr = args.lr * args.gpus

    args = argparse.Namespace(**vars(args), **parameters.models[args.model_type])

    # print to error stream as the logs for the standard stream are often full of junk :)
    print("parameters = {}".format(args), file=sys.stderr)
    print("start time = {}".format(datetime.now().strftime("%d/%m/%Y %H:%M")), file=sys.stderr)

    tokenizer = CharTokenizer(case_sensitive=args.case_sensitive)
    args.vocab_size = tokenizer.vocab_size
    args.go_token_idx = tokenizer.go_token_idx
    args.end_token_idx = tokenizer.end_token_idx

    datamodule = OCRDataModule(args, parameters.datasets[args.datasets], tokenizer=tokenizer)
    if args.load_weights_from is None:
        model = SATRNModel(args, tokenizer=tokenizer)
    else:
        model = SATRNModel.load_from_checkpoint(args.load_weights_from, tokenizer=tokenizer)

    # reproducibility
    pl.seed_everything(42)

    # saves best model
    callbacks = []
    if args.save_best_model:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_cer',
                                             dirpath='./saved_models/',
                                             filename=args.exp_name + '-{epoch:02d}-{val_cer:2.2f}',
                                             save_top_k=1,
                                             verbose=True,
                                             mode='min')
        callbacks.append(checkpoint_callback)

    # early stopping
    early_stopping_callback = pl.callbacks.EarlyStopping(monitor='val_cer',
                                               min_delta=0.0,
                                               patience=2,
                                               mode='min')
    callbacks.append(early_stopping_callback)

    trainer = pl.Trainer(gpus=args.gpus,
                         accelerator='ddp',
                         plugins=[DDPPlugin(find_unused_parameters=False)],
                         checkpoint_callback=args.save_best_model,
                         callbacks=callbacks,
                         gradient_clip_val=2.,
                         resume_from_checkpoint=args.resume_from,
                         max_epochs=args.epochs)

    if args.run_test:
        datamodule.setup(stage='test')
        for i, dataloader in enumerate(datamodule.test_dataloaders()):
            print("#test dataset {}".format(parameters.datasets[args.datasets]['test'][i]))
            trainer.test(model, dataloader)
    elif args.run_val:
        datamodule.setup(stage='validate')
        for i, dataloader in enumerate(datamodule.val_dataloaders()):
            print("#val dataset {}".format(parameters.datasets[args.datasets]['val'][i]))
            trainer.test(model, dataloader)
    else:
        trainer.fit(model, datamodule)

    print("end time = {}".format(datetime.now().strftime("%d/%m/%Y %H:%M")), file=sys.stderr)
