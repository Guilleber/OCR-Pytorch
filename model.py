import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

import pytorch_lightning as pl

from modules.PositionalEncoding import PositionalEncoding, PositionalEncoding2d
from modules.TransformerEncoderLayer2d import TransformerEncoderLayer2d


class SATRNModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        #Shallow CNN
        self.shallow_conv = nn.Sequential(nn.Conv2d(1 if hparams.grayscale else 3, hparams.d_model, 3, padding=1),
                                          nn.ReLU(),
                                          nn.dropout(hparams.dropout),
                                          nn.Conv2d(hparams.d_model, hparams.d_model, 3, padding=1))

        #Encoder
        self.encoder_pe = PositionalEncoding2d(hparams.d_model, hparams.dropout)
        encoder_layers = TransformerEncoderLayer2d(hparams.d_model, hparams.nhead, hparams.d_hidden, dropout=hparams.dropout)
        self.encoder = nn.TransformerEncoder(encoder_layers, hparams.nlayers_encoder)

        #Decoder
        self.decoder_emb = nn.Embedding(hparams.vocab_size, hparams.d_model)
        self.decoder_pe = PostionalEncoding(hparams.d_model, hparams.dropout)
        decoder_layers = TransformerDecoderLayer(hparams.d_model, hparams.nhead, hparams.d_hidden, dropout=hparams.dropout)
        self.decoder = nn.TransformerDecoder(decoder_layers, hparams.nlayers_decoder)
        self.lin_out = nn.Linear(hparams.d_model, hparams.vocab_size)
        return

    def generate_square_subsequent_mask(self, L: int):
        mask = (torch.triu(torch.ones(L, L)) == 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def run_encoder(self, img, img_padding_mask=None):
        """
        Shape:
            img: N x C x H x W
            img_padding_mask: N x H x W
        """
        img = self.shallow_conv(img)
        if img_padding_mask is not None:
            img_padding_mask = img_padding_mask.unsqueeze(1)
            img_padding_mask = F.max_pool2d(img_padding_mask, stride=4)
            img_padding_mask = img_padding_mask.squeeze(1)

        img = img.permute(2, 3, 0, 1) # -> H x W x N x C
        img = self.encoder_pe(img)
        img = self.encoder(img, mask=None, src_key_padding_mask=img_padding_mask)
        
        (H, W, N, _) = img.size()

        img = img.view(H*W, N, self.hparams.d_model)
        if img_padding_mask is not None:
            img_padding_mask = img_padding_mask.view(N, H*W)

        return img, img_padding_mask

    def run_decoder(self, memory, dec_input, attn_mask=None, memory_padding_mask=None, dec_input_padding_mask=None):
        """
        Shape:
            memory: H*W x N x C
            dec_input: L x N
            attn_mask: L x L
            memory_padding_mask: N x H*W
            dec_input_padding_mask: N x L
        """

        dec_input = self.decoder_emb(dec_input) # -> L x N x C
        dec_input = self.decoder_pe(dec_input)
        logits = self.decoder(dec_input, img, tgt_mask=attn_mask, memory_key_padding_mask=img_padding_mask, tgt_key_padding_mask=dec_input_padding_mask) # -> L x N x C
        logits = self.lin_out(logits)
        return logits

    def forward(self, img, tgt, img_padding_mask=None, tgt_padding_mask=None):
        """
        Shape:
            img: N x C x H x W
            tgt: N x L
            img_padding_mask: N x H x W
            tgt_padding_mask: N x L
        """
        memory, memory_padding_mask = self.run_encoder(img, img_padding_mask)

        tgt = tgt.transpose(0, 1) # -> L x N
        (L, N) = tgt.size()

        dec_input = torch.cat([self.hparams.go_token_idx * torch.ones(1, N).type_as(tgt), tgt[:-1, :]], dim=0)
        if tgt_padding_mask is not None:
            dec_input_padding_mask = torch.cat([tgt_padding_mask[0:1, :], tgt_padding_mask[:-1, :]], dim=0)
        else:
            dec_input_padding_mask = None
        attn_mask = self.generate_square_subsequent_mask(L) # -> L x L

        logits = run_decoder(memory, dec_input, attn_mask, memory_padding_mask, dec_input_padding_mask)

        logits = logits.transpose(0, 1) # -> N x L x V
        return logits

    def generate(self, img, img_padding_mask, max_length=25):
        """
        Shape:
            img: N x C x H x W
            img_padding_mask: N x H x W
        """
        N = img.size(0)
        memory, memory_padding_mask = self.run_encoder(img, img_padding_mask)

        termination_check = torch.zeros(N)
        dec_input = self.hparams.go_token_idx * torch.ones(1, N)
        if img.is_cuda:
            termination_check = termination_check.cuda()
            dec_input = dec_input.cuda()
        attn_mask = self.generate_square_subsequent_mask(1)
        length = 0

        while torch.sum(termination_check) != termination_check.size(0) and length < max_length:
            logits = run_decoder(memory, dec_input, attn_mask, memory_padding_mask, None)
            pred = logits.argmax(dim=2)

            termination_check = torch.max(termination_check, pred[-1, :] == self.hparams.end_token_idx)
            dec_input = torch.cat([self.hparams.go_token_idx * torch.ones(1, N).type_as(dec_input), pred], dim=0)
            attn_mask = self.generate_square_subsequent_mask(dec_input.size(0))
            length += 1
        return pred.transpose(0, 1)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.lr)

    def training_step(self, batch, batch_idx):
        logits = self(**batch)
        logits = logits.view(-1, logits.size(2))
        tgt = batch['tgt'].view(-1)

        loss = F.cross_entropy(logits, tgt)

        pred = logits.argmax(dim=1)
        acc = torch.sum(pred == tgt)
        nb_ex = tgt.size(0)

        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return {'loss': loss, 'acc': acc, 'nb_ex': nb_ex}

    def training_epoch_end(self, outputs):
        acc = torch.sum([out['acc'] for out in outputs])
        nb_ex = torch.sum([out['nb_ex'] for nb_ex in outputs])
        acc = acc/nb_ex
        self.log('acc', acc)

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            logits = self(**batch)
            logits = logits.view(-1, logits.size(2))
            tgt = batch['tgt'].view(-1)

            loss = F.cross_entropy(logits, tgt)

            pred = logits.argmax(dim=1)
            acc = torch.sum(pred == tgt)
            nb_ex = tgt.size(0)

            self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return {'loss': loss, 'acc': acc, 'nb_ex': nb_ex}

    def validation_epoch_end(self, outputs):
        acc = torch.sum([out['acc'] for out in outputs])
        nb_ex = torch.sum([out['nb_ex'] for nb_ex in outputs])
        acc = acc/nb_ex
        self.log('val_acc', acc)
