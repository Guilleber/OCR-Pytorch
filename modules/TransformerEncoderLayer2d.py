import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerEncoderLayer2d(nn.Module):
    def __init__(self, d_model, nhead, d_hidden, dropout=0.1, depthwise=True):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.d_hidden = d_hidden
        self.dropout = dropout
        self.depthwise = depthwise

        self.self_attn = nn.MultiHeadAttention(d_model, nhead, dropout=dropout)

        #Locality-aware feedforward layer
        if depthwise:
            self.feed_forward = nn.Sequential(nn.Conv2d(d_model, d_hidden, 1),
                                              nn.ReLU(),
                                              nn.Dropout(dropout),
                                              nn.Conv2d(d_hidden, d_hidden, 3, padding=1, groups=d_hidden),
                                              nn.ReLU(),
                                              nn.Dropout(dropout),
                                              nn.Conv2d(d_hidden, d_model, 1))
        else:
            self.feed_forward = nn.Sequential(nn.Conv2d(d_model, d_hidden, 3, padding=1),
                                              nn.ReLU(),
                                              nn.Dropout(dropout),
                                              nn.Conv2d(d_hidden, d_model, 3, padding=1))

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        return

    def forward(self, src, src_mask=None, key_padding_mask=None):
        """
        Shape:
            src: H x W x N x C with C=d_model
            src_mask: None
            src_key_padding_mask: N x H x W
        """

        (H, W, N, C) = src.size()
        assert C == self.d_model

        #Self attention
        src = src.view(H*W, N, C)
        src_key_padding_mask = src_key_padding_mask.view(N, H*W)
        attn = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(attn)
        src = self.norm1(src)

        #Locality-aware feedforward layer
        src = src.view(H, W, N, C)
        src = src.permute(2, 3, 0, 1) # -> N x C x H x W
        ff = self.feed_forward(src)
        src = src + self.dropout2(ff)
        src = src.permute(2, 3, 0, 1) # -> H x W x N x C
        src = self.norm2(src)
        return src
