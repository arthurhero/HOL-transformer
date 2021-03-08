import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def cut_batch(tensor, batch_size, model):
    '''
    tensor - len x long x *
    cut long into batch_sizes and feed in model
    '''
    lon = tensor.shape[1]
    num = int(math.ceil(float(lon) / batch_size))
    outs = []
    for j in range(num):
        start = j*batch_size
        end = min((j+1)*batch_size, lon)
        ten = tensor[:,start:end] # len x batch x *
        out = model(ten) # len x batch x *
        outs.append(out)
    outs = torch.cat(outs, dim=1) # len x long x *
    return outs 

def run_pretrain_transformer(conj, deps, step, neg_step, pt, percent, device, loss_fn):
    '''
    pretrain using random mask char
    conj - batch x clen
    deps - [num_dep x dlen], len=batch
    step - [num_step x slen], len=batch
    pt - pretrain_transformer
    percent - masking percent
    '''
    # turn into tensors
    conj = torch.LongTensor(conj).to(device).transpose(0,1) # clen x batch
    b=conj.shape[1]
    deps = [torch.LongTensor(dep)[torch.randperm(len(dep))[:min(len(dep),b)]].to(device).transpose(0,1) for dep in deps] # [dlen x batch]
    '''
    step = [torch.LongTensor(s).to(device).transpose(0,1) for s in step] # [slen x num_step]
    neg_step = [torch.LongTensor(s).to(device).transpose(0,1) for s in neg_step] # [slen x num_step]
    '''


    #sample_list = [conj]+deps+step+neg_step
    sample_list = [conj]+deps 
    loss = 0.0
    corrects = 0
    total = 0
    for sample in sample_list:
        #print(sample.shape)
        while 1:
            mask = torch.ones(sample.shape[0]*sample.shape[1]).long().to(device)
            mask[:int(math.ceil(len(mask)*percent))]=0
            mask = mask[torch.randperm(len(mask))]
            mask = mask.reshape(sample.shape)
            masked_sample = sample*mask
            if torch.sum(masked_sample != sample)>0 and masked_sample.sum(0).min() > 0:
                break
            '''
            if torch.sum(masked_sample != sample)==0:
                print("no diff")
                '''
            if torch.sum(masked_sample != sample)>0 and masked_sample.sum(0).min() == 0:
                print("made all 0 row")
            if masked_sample.shape[1]==0:
                print("sample batch 0!")
            if torch.sum(masked_sample != sample)==0 and masked_sample.sum(0).min() == 0:
                print("orig all 0 row")
        hidden = cut_batch(masked_sample, b, pt['encoder']) # len x batch x channel
        output = pt['decoder'](hidden) # len x batch x vocab_size
        res = loss_fn(sample, output, mask)
        '''
        if res == 0.0:
            continue
            '''
        lo, co, to = res
        loss += lo
        corrects += co
        total += to
    if loss == 0.0:
        print(loss)
    return loss, corrects, total

def run_step_cls_transformer(conj, deps, step, labels, sct, use_deps, device, loss_fn):
    '''
    int-encoded inputs, 0 padded, numpy
    conj - batch x clen
    deps - [num_dep x dlen], len=batch
    step - batch x slen
    sct - step_cls_transformer
    use_deps - bool, used deps or not
    '''
    # turn into tensors
    conj = torch.LongTensor(conj).to(device).transpose(0,1) # clen x batch
    step = torch.LongTensor(step).to(device).transpose(0,1) # slen x batch
    b=conj.shape[1]
    if use_deps:
        #deps = [torch.LongTensor(dep).to(device).transpose(0,1) for dep in deps] # [dlen x num_dep]
        deps = [torch.LongTensor(dep)[torch.randperm(len(dep))[:min(len(dep),b)]].to(device).transpose(0,1) for dep in deps] # [dlen x batch]

    encoded_conj = sct['conj_encoder'](conj) # clen x batch x channel

    if use_deps:
        encoded_deps = [cut_batch(dep, b, sct['deps_encoder']) for dep in deps] # [dlen x num_dep x channel]

    memory_len = 256 

    encoded_conj = encoded_conj[:memory_len]
    if use_deps:
        encoded_deps = [dep[:memory_len] for dep in encoded_deps]
        memory = [torch.cat([encoded_conj[:,i:i+1],encoded_deps[i].reshape(-1,1,encoded_deps[i].shape[-1])],dim=0) for i in range(b)] # [8 x (num_dep+1) x c], len=batch
        #memory = [m.reshape(m.shape[0]*m.shape[1],-1).unsqueeze(1) for m in memory]
    else:
        memory = [encoded_conj[:,i:i+1] for i in range(b)] # [8 x 1 x c]

    outputs = []
    for i in range(b):
        outputs.append(sct['step_decoder'](step[:,i:i+1] ,memory[i], mask_tgt=True)[0]) # 1 x cls_num
    outputs = torch.cat(outputs,dim=0) # batch x cls_num
    labels = torch.LongTensor(labels).to(device)
    loss, corrects, total = loss_fn(outputs, labels)
    return loss, corrects, total

def run_step_gen_transformer(conj, deps, step, sgt, d_model, device, loss_fn):
    '''
    conj - batch x clen
    deps - [num_dep x dlen], len=batch
    sgt - step_gen_transformer
    '''
    # turn into tensors
    conj = torch.LongTensor(conj).to(device).transpose(0,1) # clen x batch
    deps = [torch.LongTensor(dep).to(device).transpose(0,1) for dep in deps] # [dlen x num_dep]

    b=conj.shape[1]

    encoded_conj = sgt['conj_encoder'](conj) # clen x batch x channel
    encoded_deps = [cut_batch(dep, b, sgt['deps_encoder']) for dep in deps] # [dlen x num_dep x channel]

    memory_len = 8

    encoded_conj = encoded_conj[:memory_len]
    encoded_deps = [dep[:memory_len] for dep in encoded_deps]
    memory = [torch.cat([encoded_conj[:,i:i+1],encoded_deps[i]],dim=1) for i in range(b)] # [8 x (num_dep+1) x c], len=batch
    memory = [m.reshape(m.shape[0]*m.shape[1],-1).unsqueeze(1) for m in memory]

    tlen = max([dep.shape[0] for dep in deps]+[conj.shape[0]])
    tgt = torch.randn(tlen, b, d_model).to(device)
    outputs = []
    for i in range(b):
        outputs.append(sgt['step_decoder'](tgt[:,i:i+1] ,memory[i], mask_tgt=False)) # tlen x 1 x cls_num
    outputs = torch.cat(outputs,dim=1) # tlen x batch x cls_num
    step = [torch.LongTensor(s).to(device).transpose(0,1) for s in step] # [tlen' x num_dep]
    loss, corrects, total = loss_fn(outputs, step)
    return loss, corrects, total 

if __name__ == '__main__':
    from data_utils import DataParser
    from model import build_pretrain_transformer, build_step_cls_transformer, build_step_gen_transformer
    from loss import *

    dataparser = DataParser('../holstep', use_tokens=False, verbose=True, saved_vocab='vocab.pkl', saved_train_conj='train_conj.pkl', saved_val_conj='val_conj.pkl', saved_test_conj='test_conj.pkl', saved_max_len=57846)

    pre_train_gen = dataparser.conj_generator(split='train', batch_size=1, shuffle=True, load_neg_steps=True)
    pre_val_gen = dataparser.conj_generator(split='val', batch_size=1, shuffle=False, load_neg_steps=True)

    cls_train_gen= dataparser.steps_generator(split='train', batch_size=1, shuffle=True)
    cls_val_gen= dataparser.steps_generator(split='val', batch_size=1, shuffle=False)

    gen_train_gen= dataparser.conj_generator(split='train', batch_size=1, shuffle=True, load_neg_steps=False)
    gen_val_gen= dataparser.conj_generator(split='val', batch_size=1, shuffle=False, load_neg_steps=False)

    d_model = 8
    pt = build_pretrain_transformer(dataparser.vocab_size+1, dataparser.max_len)
    sct = build_step_cls_transformer(dataparser.vocab_size+1, dataparser.max_len)
    sgt = build_step_gen_transformer(dataparser.vocab_size+1, dataparser.max_len, d_model)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    '''
    for i, data in enumerate(cls_train_gen):
        print(i)
        loss = run_step_cls_transformer(*data, sct, device, cls_loss)
        break
    for i, data in enumerate(gen_train_gen):
        print(i)
        loss = run_step_gen_transformer(*data, sgt, d_model, device, gen_loss)
        break
        '''
    for i, data in enumerate(pre_train_gen):
        print(i)
        loss = run_pretrain_transformer(*data, pt, 0.2, device, pretrain_loss)
        break
