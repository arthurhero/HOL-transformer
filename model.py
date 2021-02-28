import math
import copy
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

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

class TransformerEncoderLayer(nn.Module):
    '''
    One encoder layer with Pre-LN
    Modified from https://github.com/facebookresearch/detr/blob/master/models/transformer.py
    '''
    def __init__(self, d_model=8, n_head=2, n_hid=8, dropout=0.0, pos_encoder = None):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, n_hid)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(n_hid, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu
        self.pos_encoder = pos_encoder

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None):
        '''
        src - length x batch x channel
        output same size
        '''
        src2 = self.norm1(src)
        if self.pos_encoder is not None:
            q = k = self.pos_encoder(src2) 
        else:
            q = k = src2
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

class TransformerEncoder(nn.Module):
    '''
    input a integer-encoded raw string, output hidden states
    vocab_size - how many chars or words
    d_model - key, query, value channel size
    nhead - number of attention heads
    nhid - fc layer channel size
    n_layers - number of encoder layers
    '''
    def __init__(self, vocab_size, d_model=8, n_head=2, n_hid=8, n_layers=6, pos_encoder=None):
        super(TransformerEncoder, self).__init__()
        self.encoder = nn.Embedding(vocab_size, d_model)
        encoder_layer = TransformerEncoderLayer(d_model, n_head, n_hid, dropout=0.0, pos_encoder=pos_encoder)
        self.transformer_encoder = _get_clones(encoder_layer, n_layers) 
        self.d_model = d_model
        self.n_layers = n_layers

    def forward(self, src,
            src_mask: Optional[Tensor] = None):
        '''
        src - length x batch, int-encoded raw string, 0 padded
        src_mask - byte tensor, length x length
        '''
        key_pad_mask = (src==0).transpose(0,1) # batch x len, True value gets ignored
        src = self.encoder(src) * math.sqrt(self.d_model) # length x batch x channel
        output = src
        for layer in self.transformer_encoder:
            output = layer(output, src_mask,
                           src_key_padding_mask=key_pad_mask)
        return output 

class TransformerDecoderLayer(nn.Module):
    '''
    One decoder layer with Pre-LN
    Modified from https://github.com/facebookresearch/detr/blob/master/models/transformer.py
    '''
    def __init__(self, d_model=8, n_head=2, n_hid=8, dropout=0.0, pos_encoder = None):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, n_hid)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(n_hid, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.relu
        self.pos_encoder = pos_encoder

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None):
        '''
        tgt - target embedding, tlen x batch x channel
        memory - output of encoder, slen x batch x channel
        mask - tlen x slen, byte or bool
        padding_mask - batch x len, byte or bool
        output tlen x batch x channel
        '''
        tgt2 = self.norm1(tgt)
        v = memory.clone()
        if self.pos_encoder is not None:
            q=k=self.pos_encoder(tgt2)
            memory = self.pos_encoder(memory)
        else:
            q=k=tgt2
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=tgt2,
                                   key=memory,
                                   value=v, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

class TransformerDecoder(nn.Module):
    '''
    input embedded target, encoded memory, output predicted string (prob for each word) 
    vocab_size - how many chars or words
    d_model - key, query, value channel size
    nhead - number of attention heads
    nhid - fc layer channel size
    n_layers - number of decoder layers
    cls_num - class number
    '''
    def __init__(self, vocab_size, cls_num, d_model=8, n_head=2, n_hid=8, n_layers=6, pos_encoder=None, embed_tgt=True):
        super(TransformerDecoder, self).__init__()
        self.embed_tgt = embed_tgt
        if embed_tgt:
            self.encoder = nn.Embedding(vocab_size, d_model)
        decoder_layer = TransformerDecoderLayer(d_model, n_head, n_hid, dropout=0.0, pos_encoder=pos_encoder)
        self.transformer_decoder = _get_clones(decoder_layer, n_layers)
        self.decoder = nn.Linear(d_model, cls_num)
        self.d_model = d_model
        self.n_layers = n_layers

    def forward(self, tgt, memory, 
            tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
            mask_tgt = True):
        '''
        tgt - length x batch (x tlen), 0 padded
        memory - slen x batch x channel
        mask - byte tensor, length x length
        pad_mask - batch x length
        '''
        if mask_tgt:
            key_pad_mask = (tgt==0).transpose(0,1) # batch x len, True value gets ignored
        else:
            key_pad_mask = None
        if self.embed_tgt:
            tgt = self.encoder(tgt) * math.sqrt(self.d_model) # length x batch x channel
        output = tgt
        for layer in self.transformer_decoder:
            output = layer(output, memory, tgt_mask, memory_mask,
                           tgt_key_padding_mask = key_pad_mask,
                           memory_key_padding_mask = memory_key_padding_mask)
        output = self.decoder(output.mean(0)) # batch x num_cls
        return output 

def build_pretrain_transformer(vocab_size, pos_max_len,d_model=8):
    pos_encoder = PositionalEncoding(d_model, max_len=pos_max_len)
    model = nn.ModuleDict({
        'encoder': TransformerEncoder(vocab_size, d_model, n_head=2, n_hid=8, n_layers=6, pos_encoder=pos_encoder),
        'decoder': nn.Linear(d_model, vocab_size)
    })
    return model

def build_step_cls_transformer(vocab_size, pos_max_len,d_model=8):
    pos_encoder = PositionalEncoding(d_model, max_len=pos_max_len)
    model = nn.ModuleDict({
        'conj_encoder': TransformerEncoder(vocab_size, d_model, n_head=2, n_hid=8, n_layers=6, pos_encoder=pos_encoder),
        'deps_encoder': TransformerEncoder(vocab_size, d_model, n_head=2, n_hid=8, n_layers=6, pos_encoder=pos_encoder),
        'step_decoder': TransformerDecoder(vocab_size, 2, d_model, n_head=4, n_hid=16, n_layers=24, pos_encoder=pos_encoder)
    })
    return model

def build_step_gen_transformer(vocab_size, pos_max_len,d_model=8):
    pos_encoder = PositionalEncoding(d_model, max_len=pos_max_len)
    model = nn.ModuleDict({
        'conj_encoder': TransformerEncoder(vocab_size, d_model, n_head=2, n_hid=8, n_layers=6, pos_encoder=pos_encoder),
        'deps_encoder': TransformerEncoder(vocab_size, d_model, n_head=2, n_hid=8, n_layers=6, pos_encoder=pos_encoder),
        'step_decoder': TransformerDecoder(vocab_size, vocab_size, d_model, n_head=2, n_hid=8, n_layers=6, pos_encoder=pos_encoder, embed_tgt=False)
    })
    return model

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
