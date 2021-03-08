import math
import os
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_utils import DataParser
from model import build_pretrain_transformer, build_step_cls_transformer, build_step_gen_transformer
from run_models import *
from loss import *

def train(model, run_fn, loss_fn, optimizer, num_epochs, train_gen_gen, val_gen_gen, device, ckpt_path, best_ckpt_path, arg):
    
    if os.path.isfile(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path))
        print("Loaded ckpt!")

    best_acc = 0.0
    model = model.to(device)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            step = 0
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            running_loss = 0.0
            running_corrects = 0.0
            running_totals = 0.0

            if phase == 'train':
                dataloader = next(train_gen_gen)
            else:
                dataloader = next(val_gen_gen)
            for i,data in enumerate(dataloader):
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    loss,co,to = run_fn(*data, model, *arg, device, loss_fn)
                    #if phase == 'train' and loss != 0.0:
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item()
                running_corrects += co.item()
                running_totals += to

                if step % 100 == 99:
                    step_loss = running_loss / running_totals 
                    step_acc = running_corrects / running_totals
                    print('Step {}: {:.4f} Acc: {:.4f}'.format(step, step_loss, step_acc))
                    torch.save(model.state_dict(), ckpt_path)
                if phase == 'train':
                    step += 1
        epoch_loss = running_loss / running_totals
        epoch_acc = running_corrects / running_totals
        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

        if phase == 'val' and (epoch_acc > best_acc):
            best_acc = epoch_acc
            torch.save(model.state_dict(), best_ckpt_path)
        print()
    return

if __name__ == '__main__':
    num_epochs = 10
    lr = 10e-4
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dataparser = DataParser('../holstep', max_len=256, use_tokens=False, verbose=True, saved_vocab='vocab.pkl', saved_train_conj='train_conj.pkl', saved_val_conj='val_conj.pkl', saved_test_conj='test_conj.pkl', saved_max_len=57846)

    # ckpt names
    pt_ckpt = 'pt.ckpt'
    sct_ckpt = 'sct.ckpt'
    sgt_ckpt = 'sgt.ckpt'

    # model size
    d_model = 8
    n_head=8
    n_hid=16
    n_layers=6

    # pretrain encoders
    pre_train_gen = dataparser.conj_gen_gen(split='train', batch_size=1, shuffle=True, load_neg_steps=True)
    pre_val_gen = dataparser.conj_gen_gen(split='val', batch_size=1, shuffle=False, load_neg_steps=True)

    pt = build_pretrain_transformer(dataparser.vocab_size+3, dataparser.max_len, d_model, n_head, n_hid, n_layers)
    optimizer = torch.optim.Adam(pt.parameters(),lr=lr,betas=(0.5,0.9))

    train(pt, run_pretrain_transformer, pretrain_loss, optimizer, num_epochs, pre_train_gen, pre_val_gen, device, pt_ckpt, 'best_'+pt_ckpt, [0.2])
    '''

    # train step classifider
    cls_train_gen= dataparser.steps_gen_gen(split='train', batch_size=1, shuffle=True)
    cls_val_gen= dataparser.steps_gen_gen(split='val', batch_size=1, shuffle=False)
    sct = build_step_cls_transformer(dataparser.vocab_size+3, dataparser.max_len, d_model, n_head, n_hid, n_layers)
    '''
    '''
    pt = build_pretrain_transformer(dataparser.vocab_size+3, dataparser.max_len, d_model, n_head, n_hid, n_layers)
    #pt.load_state_dict(torch.load('best_'+pt_ckpt))
    pt.load_state_dict(torch.load(pt_ckpt))
    encoder_state = copy.deepcopy(pt['encoder'].state_dict())
    sct['conj_encoder'].load_state_dict(encoder_state)
    sct['deps_encoder'].load_state_dict(encoder_state)
    '''
    '''
    optimizer = torch.optim.Adam(sct.parameters(),lr=lr,betas=(0.5,0.9))
    train(sct, run_step_cls_transformer, cls_loss, optimizer, num_epochs, cls_train_gen, cls_val_gen, device, sct_ckpt, 'best_'+sct_ckpt, [True])
    '''

