import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # -> max_len x 1
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x)
        x = x + self.pe[:x.size(0), :, :]
        return self.dropout(x)


class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_height=1000, max_width=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        assert d_model % 2 == 0

        pe = torch.zeros(max_height, max_width, d_model)
        d_model = d_model // 2
        pos_h = torch.arange(0, max_height, dtype=torch.float).unsqueeze(1)
        pos_w = torch.arange(0, max_width, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, :, 0:d_model:2] = torch.sin(pos_w * div_term).unsqueeze(0).repeat(max_height, 1, 1)
        pe[:, :, 1:d_model:2] = torch.cos(pos_w * div_term).unsqueeze(0).repeat(max_height, 1, 1)
        pe[:, :, d_model::2] = torch.sin(pos_h * div_term).unsqueeze(1).repeat(1, max_width, 1)
        pe[:, :, d_model+1::2] = torch.cos(pos_h * div_term).unsqueeze(1).repeat(1, max_width, 1)
        pe = pe.unsqueeze(2)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :x.size(1), :, :]
        return self.dropout(x)
