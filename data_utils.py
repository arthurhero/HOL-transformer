# Modified from https://github.com/tensorflow/deepmath/blob/master/deepmath/holstep_baselines/data_utils.py

# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utility functions to parse and format the raw HOL statement text files.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import random
import numpy as np

import pickle

logging.basicConfig(level=logging.NOTSET)

class DataParser(object):

  def __init__(self, source_dir, max_len=256, use_tokens=False, verbose=True, saved_vocab = None, saved_train_conj=None,
          saved_val_conj=None, saved_test_conj=None, saved_max_len = 57846):
    #random.seed(1337)
    self.use_tokens = use_tokens
    self.max_len = max_len 
    self.real_max_len = saved_max_len
    if self.use_tokens:
      self.step_markers = {'T'}
 
      def tokenize_fn(line):
        line = line.rstrip()[2:]
        tokens = line.split()
        return tokens
      self.tokenize_fn = tokenize_fn
    else:
      self.step_markers = {'+', '-', 'C', 'A'}
      self.tokenize_fn = lambda x: x.rstrip()[2:]
    self.verbose = verbose
    train_dir = os.path.join(source_dir, 'train')
    test_dir = os.path.join(source_dir, 'test')
    self.train_fnames = [
        os.path.join(train_dir, '%05d' % i) for i in range(1, 9500)]
    self.val_fnames = [
        os.path.join(train_dir, '%05d' % i) for i in range(9500, 10000)]
    self.test_fnames = [
        os.path.join(test_dir, '%04d' % i) for i in range(1, 1412)] 
    if verbose:
      logging.info('Building vocabulary...')
    if saved_vocab is not None and os.path.isfile(saved_vocab):
      with (open(saved_vocab, "rb")) as f:
        self.vocabulary = pickle.load(f)
    else:
      self.vocabulary = self.build_vocabulary()
      f = open("vocab.pkl","wb")
      pickle.dump(self.vocabulary,f)
      f.close()
    if verbose:
      logging.info('Found %s unique tokens.', len(self.vocabulary))
    self.vocab_size = len(self.vocabulary)
    self.vocabulary_index = dict(enumerate(self.vocabulary))
    self.reverse_vocabulary_index = dict(
        [(value, key) for (key, value) in self.vocabulary_index.items()])

    # special chars not included in reverse vocab
    self.vocabulary_index[-1]=' ' # <PAD>
    self.vocabulary_index[self.vocab_size]='<UNK>'
    self.vocabulary_index[self.vocab_size+1]='<CLS>'

    if saved_train_conj is not None and os.path.isfile(saved_train_conj):
      with (open(saved_train_conj, "rb")) as f:
        self.train_conjectures, self.train_step_nums, self.train_conj_names  = pickle.load(f)
    else:
      self.train_conjectures, self.train_step_nums, self.train_conj_names, ml= self.parse_file_list(self.train_fnames)
      if ml > self.real_max_len:
        self.real_max_len = ml
      f = open("train_conj.pkl","wb")
      pickle.dump((self.train_conjectures,self.train_step_nums, self.train_conj_names),f)
      f.close()

    if saved_val_conj is not None and os.path.isfile(saved_val_conj):
      with (open(saved_val_conj, "rb")) as f:
        self.val_conjectures, self.val_step_nums, self.val_conj_names = pickle.load(f)
    else:
      self.val_conjectures, self.val_step_nums, self.val_conj_names, ml = self.parse_file_list(self.val_fnames)
      if ml > self.real_max_len:
        self.real_max_len = ml
      f = open("val_conj.pkl","wb")
      pickle.dump((self.val_conjectures, self.val_step_nums, self.val_conj_names),f)
      f.close()

    if saved_test_conj is not None and os.path.isfile(saved_test_conj):
      with (open(saved_test_conj, "rb")) as f:
        self.test_conjectures, self.test_step_nums, self.test_conj_names = pickle.load(f)
    else:
      self.test_conjectures, self.test_step_nums, self.test_conj_names, ml = self.parse_file_list(self.test_fnames)
      if ml > self.real_max_len:
          self.real_max_len = ml
      f = open("test_conj.pkl","wb")
      pickle.dump((self.test_conjectures, self.test_step_nums, self.test_conj_names),f)
      f.close()

    if verbose:
      logging.info('Real max length is '+str(self.real_max_len))

  def build_vocabulary(self):
    vocabulary = set()
    for fname in self.train_fnames:
      f = open(fname)
      for line in f:
        if line[0] in self.step_markers:
          for token in self.tokenize_fn(line):
            vocabulary.add(token)
      f.close()
    return vocabulary

  def parse_file_list(self, fnames):
    conjectures = {}
    conj_names = []
    step_nums = []
    max_len = 0
    for fname in fnames:
      conjecture, l = self.parse_file(fname)
      if l>max_len:
          max_len = l
      name = conjecture.pop('name')
      conjectures[name] = conjecture
      step_nums.append(len(conjecture['+'])*2)
      conj_names.append(name)
    step_nums = np.asarray(step_nums)
    step_nums = np.cumsum(step_nums)
    return conjectures, step_nums, conj_names, max_len

  def display_stats(self, conjectures):
    dep_counts = []
    dep_lengths = []
    conj_lengths = []
    pos_step_counts = []
    pos_step_lengths = []
    neg_step_counts = []
    neg_step_lengths = []

    logging.info('%s conjectures in total.', len(conjectures))
    for value in conjectures.values():
      deps = value['deps']
      conj = value['conj']
      pos_steps = value['+']
      neg_steps = value['-']
      dep_counts.append(len(deps))
      if deps:
        dep_lengths.append(np.mean([len(x) for x in deps]))
      conj_lengths.append(len(conj))
      pos_step_counts.append(len(pos_steps))
      if pos_steps:
        pos_step_lengths.append(np.mean([len(x) for x in pos_steps]))
      neg_step_counts.append(len(neg_steps))
      if neg_steps:
        neg_step_lengths.append(np.mean([len(x) for x in neg_steps]))
    logging.info('total number of steps: %s',
                 np.sum(pos_step_counts) + np.sum(neg_step_counts))
    logging.info('mean number of positive steps per conjecture: %s',
                 np.mean(pos_step_counts))
    logging.info('mean number of negative steps per conjecture: %s',
                 np.mean(neg_step_counts))
    logging.info('mean conjecture length: %s', np.mean(conj_lengths))
    logging.info('mean number of dependencies: %s', np.mean(dep_counts))
    logging.info('mean dependency length: %s', np.mean(dep_lengths))
    logging.info('mean number of positive steps: %s',
                 np.mean(pos_step_counts))
    logging.info('mean number of negative steps: %s',
                 np.mean(neg_step_counts))
    logging.info('mean positive step length: %s', np.mean(pos_step_lengths))
    logging.info('mean negative step length: %s', np.mean(neg_step_lengths))
    # TODO(fchollet): plot histograms

  def parse_file(self, fname):
    f = open(fname)
    name = f.readline().rstrip()[2:]
    if self.use_tokens:
      # Text representation of conjecture.
      f.readline()
      # Tokenization of conjecture.
      conj = self.tokenize_fn(f.readline())
    else:
      # Text representation of conjecture.
      conj = self.tokenize_fn(f.readline())
    conjecture = {
        'name': name,
        'deps': [],
        '+': [],
        '-': [],
        'conj': conj,
    }
    max_len = len(conj)
    while 1:
      line = f.readline()
      if not line:
        break
      marker = line[0]
      if self.use_tokens:
        line = f.readline()  # Text representation
      content = self.tokenize_fn(line)
      if marker == 'A':
        conjecture['deps'].append(content)
        if len(content)>max_len:
            max_len = len(content)
      elif marker == '+':
        conjecture['+'].append(content)
        if len(content)>max_len:
            max_len = len(content)
      elif marker == '-':
        conjecture['-'].append(content)
        if len(content)>max_len:
            max_len = len(content)
    return conjecture, max_len

  def integer_encode_statements(self, statements):
    max_len = self.max_len
    encoded = np.zeros((len(statements), max_len), dtype='int32')
    for s, statement in enumerate(statements):
      for i, char in enumerate(statement[:max_len]):
        encoded[s, i] = self.reverse_vocabulary_index.get(
            char, self.vocab_size) + 1 # <UNK> maps to vocab_size+1, <PAD> to 0
    return encoded
 
  def integer_decode_statements(self, statements):
      strs = list()
      for s, statement in enumerate(statements):
          s = ''
          for i, char in enumerate(statement):
              s += self.vocabulary_index.get(char-1,'<UNK>')
          strs.append(s.rstrip())
      return strs

  def one_hot_encode_statments(self, statements):
    max_len = 0
    for s, statement in enumerate(statements):
      if len(statement)>max_len:
        max_len = len(statement)
    encoded = np.zeros((len(statements), max_len, len(self.vocabulary) + 1),
                       dtype='float32')
    for s, statement in enumerate(statements):
      for i, char in enumerate(statement[:max_len]):
        j = self.reverse_vocabulary_index.get(char, -1) + 1
        encoded[s, max_len - i - 1, j] = 1
    return encoded

  def draw_step_by_index(self, index=0, split='train'):
    '''
    draw one step with its conjecture and deps and label
    '''
    if split == 'train':
      all_conjectures = self.train_conjectures
      conjecture_names = self.train_conj_names
      step_nums = self.train_step_nums
    elif split == 'val':
      all_conjectures = self.val_conjectures
      conjecture_names = self.val_conj_names
      step_nums = self.val_step_nums
    elif split == 'test':
      all_conjectures = self.test_conjectures
      conjecture_names = self.test_conj_names
      step_nums = self.test_step_nums
    else:
      raise ValueError('`split` must be in {"train", "val", "test"}.')
    conj_idx = (step_nums<=index).astype(int).sum() # step_nums are cumsum
    conj_name = conjecture_names[conj_idx]
    conjecture = all_conjectures[conj_name]
    if index < step_nums[0]:
      step_idx = index
    else:
      step_idx = index-step_nums[conj_idx-1]
    pos_step_num = len(conjecture['+'])
    if step_idx >= pos_step_num:
      step = conjecture['-'][step_idx-pos_step_num]
      label = 0
    else:
      step = conjecture['+'][step_idx]
      label = 1
    deps = conjecture['deps']
    conj = conjecture['conj']
    return step, conj, deps, label

  def draw_conjecture_by_index(self, index=0, split='train', load_neg_steps = False):
    '''
    load the entire proof including conj, deps and steps
    '''
    if split == 'train':
      all_conjectures = self.train_conjectures
      conjecture_names = self.train_conj_names
    elif split == 'val':
      all_conjectures = self.val_conjectures
      conjecture_names = self.val_conj_names
    elif split == 'test':
      all_conjectures = self.test_conjectures
      conjecture_names = self.test_conj_names
    else:
      raise ValueError('`split` must be in {"train", "val", "test"}.')
    
    conj_name = conjecture_names[index]
    conjecture = all_conjectures[conj_name]
    steps = conjecture['+']
    if len(steps)==0:
      return None
    deps=conjecture['deps']
    if len(deps)==0:
      return None
    conj=conjecture['conj']
    if load_neg_steps:
      return steps, conjecture['-'], conj, deps
    else:
      return steps, conj, deps

  def steps_generator(
    self, split='train', encoding='integer', batch_size=8, shuffle=True):
    if split == 'train':
      total = self.train_step_nums[-1]
    elif split == 'val':
      total = self.val_step_nums[-1]
    elif split == 'test':
      total = self.test_step_nums[-1]
    else:
      raise ValueError('`split` must be in {"train", "val", "test"}.')

    if encoding == 'integer':
      encode = lambda x: self.integer_encode_statements(x)
    elif encoding == 'one-hot':
      encode = lambda x: self.one_hot_encode_statments(x)
    else:
      raise ValueError('Unknown encoding:', encoding)

    indices = list(range(total))
    if shuffle:
      random.shuffle(indices)
    cur_idx = 0
    while cur_idx < len(indices):
      conj = []
      deps = []
      step = []
      label = []
      while cur_idx < len(indices) and len(conj)<batch_size:
         s, c, d, l = self.draw_step_by_index(indices[cur_idx], split)
         step.append(s)
         conj.append(c)
         label.append(l)
         deps.append(encode(d))
         cur_idx += 1
      encoded_step = encode(step) # b x 256
      cls_col = np.zeros((encoded_step.shape[0],1))+(self.vocab_size+2) # <CLS>, b x 1
      encoded_step = np.concatenate([cls_col,encoded_step],axis=1) # b x (1+256)
      yield (encode(conj), deps, encoded_step, np.asarray(label))

  def steps_gen_gen(self, split='train', encoding='integer', batch_size=8, shuffle=True):
      while 1:
          yield self.steps_generator(split, encoding, batch_size, shuffle)

  def conj_generator(
      self, split='train', encoding='integer', batch_size=8, shuffle=True, load_neg_steps=False):
    if split == 'train':
      total = len(self.train_conj_names)
    elif split == 'val':
      total = len(self.val_conj_names)
    elif split == 'test':
      total = len(self.test_conj_names)
    else:
      raise ValueError('`split` must be in {"train", "val", "test"}.')

    if encoding == 'integer':
      encode = lambda x: self.integer_encode_statements(x)
    elif encoding == 'one-hot':
      encode = lambda x: self.one_hot_encode_statments(x)
    else:
      raise ValueError('Unknown encoding:', encoding)

    indices = list(range(total))
    if shuffle:
      random.shuffle(indices)
    cur_idx = 0
    while cur_idx < len(indices):
      conj = []
      deps = []
      step = []
      neg_step = []
      while cur_idx < len(indices) and len(conj)<batch_size:
         ret = self.draw_conjecture_by_index(indices[cur_idx], split, load_neg_steps)
         if ret is None:
           cur_idx += 1
           continue
         if load_neg_steps:
           s, ns, c, d = ret 
           neg_step.append(encode(ns))
         else:
           s, c, d = ret 
         step.append(encode(s))
         conj.append(c)
         deps.append(encode(d))
         cur_idx += 1
      if load_neg_steps:
          yield (encode(conj), deps, step, neg_step)
      else:
          yield (encode(conj), deps, step)

  def conj_gen_gen(self, split='train', encoding='integer', batch_size=8, shuffle=True, load_neg_steps=False):
    while 1:
      yield self.conj_generator(split, encoding, batch_size, shuffle, load_neg_steps)

if __name__ == '__main__':
    dataparser = DataParser('../holstep', max_len=256, use_tokens=False, verbose=True, saved_vocab='vocab.pkl', saved_train_conj='train_conj.pkl', saved_val_conj='val_conj.pkl', saved_test_conj='test_conj.pkl', saved_max_len=57846)
    print(dataparser.train_step_nums[-1])
    '''
    mask_train_gen= dataparser.conj_generator(split='train', batch_size=1, shuffle=True, load_neg_steps = True)
    cls_train_gen= dataparser.steps_generator(split='train', batch_size=1, shuffle=True)
    gen_train_gen= dataparser.conj_generator(split='train', batch_size=1, shuffle=True, load_neg_steps = False)
    print(dataparser.vocab_size)
    c,d,s,l = next(cls_train_gen)
    print("conj",dataparser.integer_decode_statements(c)[0])
    print("step",dataparser.integer_decode_statements(s)[0])
    '''
