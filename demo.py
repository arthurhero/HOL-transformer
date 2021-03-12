import math
import os
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_utils import DataParser
from model import build_pretrain_transformer, build_step_cls_transformer, build_step_gen_transformer
from run_models import *


if __name__ == '__main__':
    batch_size = 1
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dataparser = DataParser('../holstep', max_len=256, use_tokens=False, verbose=True, saved_vocab='vocab.pkl', saved_train_conj='train_conj.pkl', saved_val_conj='val_conj.pkl', saved_test_conj='test_conj.pkl', saved_max_len=57846)

    # ckpt names
    pt_ckpt = 'pt.ckpt'
    sct_ckpt = 'sct.ckpt'
    sgt_ckpt = 'sgt.ckpt'

    # model size
    d_model = 64
    n_head=4
    n_hid =128
    n_layers=9

    # data generator
    pre_val_gen = dataparser.conj_generator(split='val', batch_size=1, shuffle=True, load_neg_steps=True)
    gen_val_gen= dataparser.conj_generator(split='val', batch_size=1, shuffle=True, load_neg_steps=False)

    # models
    pt = build_pretrain_transformer(dataparser.vocab_size+3, dataparser.max_len,d_model, n_head, n_hid, n_layers).to(device) #
    pt.load_state_dict(torch.load('pt_64_l9_h4.ckpt'))
    sgt = build_step_gen_transformer(dataparser.vocab_size+3, dataparser.max_len, d_model, n_head, n_hid, n_layers).to(device)
    sgt.load_state_dict(torch.load('best_sgt.ckpt'))

    # pretrain
    for i, data in enumerate(pre_val_gen):
        sample, mask, outputs = run_pretrain_transformer(*data, pt, 0.2, device, None, True)
        masked_sample = sample * mask
        _, outputs = torch.max(outputs, 2)
        outputs = (1-mask)*outputs+masked_sample
        sample = sample.transpose(0,1).cpu().detach().numpy()
        masked_sample = masked_sample.transpose(0,1).cpu().detach().numpy()
        outputs = outputs.transpose(0,1).cpu().detach().numpy()
        print(dataparser.integer_decode_statements(sample)[0])
        print()
        print(dataparser.integer_decode_statements(masked_sample)[0])
        print()
        print(dataparser.integer_decode_statements(outputs)[0])
        print()
        break

    '''
    for i, data in enumerate(gen_val_gen):
        conj, deps, outputs = run_step_gen_transformer(*data, sgt, d_model, device, None, True)
        _, outputs = torch.max(outputs, 2)
        conj = conj.transpose(0,1).cpu().detach().numpy()
        deps = deps.transpose(0,1).cpu().detach().numpy()
        outputs = outputs.transpose(0,1).cpu().detach().numpy()
        print(dataparser.integer_decode_statements(conj))
        '''


