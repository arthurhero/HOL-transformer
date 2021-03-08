import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def pretrain_loss(sample, output, mask):
    '''
    sample - len x batch
    output - len x batch x vocab
    mask - len x batch
    '''
    mask = 1-mask
    mask *= (sample!=0).long()
    if mask.sum()==0:
        print("mask zero")
        return 0.0
    nd_idx = mask.nonzero(as_tuple=True)
    sample = sample[nd_idx]
    output = output[nd_idx]
    loss = nn.CrossEntropyLoss(reduction='sum')(output, sample)
    _, preds = torch.max(output, 1)
    corrects = torch.sum(preds == sample.data)
    return loss, corrects, len(sample)

def cls_loss(outputs, labels):
    '''
    outputs - batch x 2
    labels - batch
    '''
    #print(outputs)
    loss = nn.CrossEntropyLoss(reduction='sum')(outputs, labels)
    _, preds = torch.max(outputs, 1)
    corrects = torch.sum(preds == labels.data)
    #print(loss, corrects/len(labels))
    return loss, corrects, len(labels)

def gen_loss(outputs, step):
    '''
    outputs - tlen x batch x cls_num
    step - [tlen' x num_dep]
    '''
    loss = 0.0
    corrects = 0
    total = 0
    for i in range(len(step)):
        min_loss = None
        out = outputs[:,i] # tlen x cls
        steps = step[i] # tlen' x num_dep
        length = min(out.shape[0], steps.shape[0])
        out = out[:length]
        steps = steps[:length]
        for j in range(steps.shape[1]):
            s = steps[:,j] # tlen'
            mask_len = torch.sum(s != 0)
            s = s[:mask_len]
            o = out[:mask_len]
            l = nn.CrossEntropyLoss(reduction='sum')(o, s)
            if min_loss is None or l<min_loss:
                min_loss = l
                _, preds = torch.max(o, 1)
                corrects_len = torch.sum(preds == s.data)
                total_len = len(s)
        loss += min_loss
        corrects += corrects_len
        total += total_len
    return loss, corrects, total

