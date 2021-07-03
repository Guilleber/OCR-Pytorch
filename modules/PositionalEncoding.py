import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: Optional[float] = 0.1, max_len: Optional[int] = 100):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # -> max_len x 1
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(0), :, :]
        return self.dropout(x)


class PositionalEncoding2d(nn.Module):
    def __init__(self, d_model: int, dropout: Optional[float] = 0.1, max_height: Optional[int] = 1000,
                 max_width: Optional[int] = 1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        assert d_model % 2 == 0

        pe = torch.zeros(max_height, max_width, d_model)
        d_model = d_model // 2
        pos_h = torch.arange(0, max_height, dtype=torch.float).unsqueeze(1)
        pos_w = torch.arange(0, max_width, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))

        pe[:, :, 0:d_model:2] = torch.sin(pos_w * div_term).unsqueeze(0).repeat(max_height, 1, 1)
        pe[:, :, 1:d_model:2] = torch.cos(pos_w * div_term).unsqueeze(0).repeat(max_height, 1, 1)
        pe[:, :, d_model::2] = torch.sin(pos_h * div_term).unsqueeze(1).repeat(1, max_width, 1)
        pe[:, :, d_model+1::2] = torch.cos(pos_h * div_term).unsqueeze(1).repeat(1, max_width, 1)
        pe = pe.unsqueeze(2)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(0), :x.size(1), :, :]
        return self.dropout(x)


class A2DPE(nn.Module):
    def __init__(self, d_model: int, dropout: Optional[float] = 0.1, max_height: Optional[int] = 1000,
                 max_width: Optional[int] = 1000):
        """
        Adaptive 2D Positional Encoding from paper "On Recognizing Texts of Arbitrary Shapes with 2D Self-Attention"
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear_alpha = nn.Sequential(nn.Linear(d_model, d_model//2),
                                          nn.ReLU(),
                                          nn.Dropout(dropout),
                                          nn.Linear(d_model//2, 2*d_model),
                                          nn.Sigmoid())

        pe_h = torch.zeros(max_height, max_width, d_model)
        pe_w = torch.zeros(max_height, max_width, d_model)
        pos_h = torch.arange(0, max_height, dtype=torch.float).unsqueeze(1)
        pos_w = torch.arange(0, max_width, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))

        pe_w[:, :, 0::2] = torch.sin(pos_w * div_term).unsqueeze(0).repeat(max_height, 1, 1)
        pe_w[:, :, 1::2] = torch.cos(pos_w * div_term).unsqueeze(0).repeat(max_height, 1, 1)
        pe_h[:, :, 0::2] = torch.sin(pos_h * div_term).unsqueeze(1).repeat(1, max_width, 1)
        pe_h[:, :, 1::2] = torch.cos(pos_h * div_term).unsqueeze(1).repeat(1, max_width, 1)
        
        pe_w = pe_w.unsqueeze(2)
        pe_h = pe_h.unsqueeze(2)
        self.register_buffer('pe_w', pe_w)
        self.register_buffer('pe_h', pe_h)

    def forward(self, x: Tensor) -> Tensor:
        """
        x : H x W x N x C
        """
        H, W, N, C = x.size()

        alpha = self.linear_alpha(x.reshape(-1, N, C).mean(dim=0)) # -> N x 2C
        alpha = alpha.reshape(1, 1, N, 2*C) # -> 1 x 1 x N x 2C
        x = x + alpha[:, :, :, :C]*self.pe_h[:H, :W, :, :] + alpha[:, :, :, C:]*self.pe_w[:H, :W, :, :]
        return self.dropout(x)


class ExperimentalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: Optional[float] = 0.1, max_height: Optional[int] = 1000,
                 max_width: Optional[int] = 1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear_alpha = nn.Sequential(nn.Linear(d_model, d_model//2),
                                          nn.ReLU(),
                                          nn.Dropout(dropout),
                                          nn.Linear(d_model//2, d_model),
                                          nn.Sigmoid())

        pe = torch.zeros(max_height, max_width, d_model, dtype=torch.float)

        assert d_model%2 == 0
        d_model //= 2
        self.d_model = d_model

        pos_h = torch.arange(0, max_height, dtype=torch.float).unsqueeze(1)
        pos_w = torch.arange(0, max_width, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))

        pe[:, :, 0:d_model:2] = (pos_w * div_term).unsqueeze(0).repeat(max_height, 1, 1)
        pe[:, :, 1:d_model:2] = (pos_w * div_term).unsqueeze(0).repeat(max_height, 1, 1)
        pe[:, :, d_model::2] = (pos_h * div_term).unsqueeze(1).repeat(1, max_width, 1)
        pe[:, :, d_model+1::2] = (pos_h * div_term).unsqueeze(1).repeat(1, max_width, 1)
        
        pe = pe.unsqueeze(2) # -> max_H x max_W x 1 x C
        self.register_buffer('pe', pe)
        return

    def forward(self, x: Tensor) -> Tensor:
        """
        x : H x W x N x C
        """
        H, W, N, C = x.size()

        alpha = self.linear_alpha(x.reshape(-1, N, C).mean(dim=0)) # -> N x C
        alpha = alpha.reshape(1, 1, N, C) # -> 1 x 1 x N x C
        pe = alpha*self.pe[:H, :W, :, :] # -> H x W x N x C

        pe[:, :, :, 0:self.d_model:2] = torch.sin(self.pe[:H, :W, :, 0:self.d_model:2])
        pe[:, :, :, 1:self.d_model:2] = torch.cos(self.pe[:H, :W, :, 1:self.d_model:2])
        pe[:, :, :, self.d_model::2] = torch.sin(self.pe[:H, :W, :, self.d_model::2])
        pe[:, :, :, self.d_model+1::2] = torch.cos(self.pe[:H, :W, :, self.d_model+1::2])

        x = x + pe

        return self.dropout(x)
