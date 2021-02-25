import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer

class PositionalEncoding(nn.Module):
    # modified from https://github.com/pytorch/examples/blob/master/word_language_model/model.py
    def __init__(self, d_model, max_len=1024):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model) # l x c
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # l x 1
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # c/2
        pe[:, 0::2] = torch.sin(position * div_term) # l x c/2
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # l x 1 x c
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=8, nhead=2, nhid=8, n_encode_layers=6,
            n_decode_layers=6):
        super(Transformer, self).__init__()
