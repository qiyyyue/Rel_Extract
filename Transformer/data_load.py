# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
from __future__ import print_function
from Transformer.hyperparams import Hyperparams as hp
from sklearn import preprocessing
import tensorflow as tf
import numpy as np
import codecs
import regex

def load_vocab():
    vocab = [line.split()[0] for line in codecs.open(hp.vocab_path, 'r', 'utf-8').read().splitlines() if int(line.split()[1]) >= hp.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def load_relid2target():
    relid2target_dict = {}
    for relid, target in [line.split('\t') for line in codecs.open(hp.relid2target_path, 'r', 'utf-8').readlines() if line]:
        relid2target_dict[relid] = int(target)
    return relid2target_dict

def create_data(sentences, targets):

    word2id, id2word = load_vocab()
    # Index
    x_list, y_list, Sentences, Targets = [], [], [], []

    for sentence, target in zip(sentences, targets):
        x = [word2id.get(word, 1) for word in (sentence + u" </S>").split()] # 1: OOV, </S>: End of Text
        y = [0]*hp.class_num
        y[target] = 1
        if len(x) <= hp.maxlen:
            x_list.append(np.array(x))
            y_list.append(np.array(y))
            Sentences.append(sentence)
            Targets.append(target)
    print(len(x_list), len(y_list))
    print('shape', x_list[0].shape, y_list[0].shape)
    # Pad      
    X = np.zeros([len(x_list), hp.maxlen], np.int32)
    Y = np.zeros([len(y_list), hp.class_num], np.int32)
    for i, (x, y) in enumerate(zip(x_list, y_list)):
        X[i] = np.lib.pad(x, [0, hp.maxlen-len(x)], 'constant', constant_values=(0, 0))
        Y[i] = y
    
    return X, Y, Sentences, Targets

def load_train_data():
    # 构建句子id到关系id映射表
    sent_rel_train = [line.strip().split('\t') for line in codecs.open(hp.sent2rel_train_path, 'r', 'utf-8').readlines() if line]
    sentid2relid = {}

    for sentid, relids in sent_rel_train:
        sentid2relid[sentid] = relids.split()

    #构建句子集合,对应的关系id集合
    train_sentences = []
    train_targets = []
    for train_sentid, _, _, train_sentence in [line.strip().split('\t') for line in codecs.open(hp.train_senteces_path, 'r', 'utf-8').readlines() if line]:
        for relid in sentid2relid[train_sentid]:
            train_sentences.append(train_sentence)
            train_targets.append(int(relid))
        # train_sentences.append(train_sentence)
        # train_targets.append(sentid2relid[train_sentid])

    #print('len', len(train_sentences), len(train_targets))
    X, Y, Sources, Targets = create_data(train_sentences, train_targets)
    return X, Y
    
def load_test_data():
    # 构建句子id到关系id映射表
    sent_rel_train = [line.strip().split('\t') for line in codecs.open(hp.sent2rel_dev_path, 'r', 'utf-8').readlines() if line]
    sentid2relid = {}
    for sentid, relid in sent_rel_train:
        sentid2relid[sentid] = relid
    # 构建句子集合,对应的关系id集合
    train_sentences = []
    train_targets = []
    for train_sentid, _, _, train_sentence in [line.strip().split('\t') for line in codecs.open(hp.dev_senteces_path, 'r', 'utf-8').readlines() if line]:
        for relid in sentid2relid[train_sentid]:
            train_sentences.append(train_sentence)
            train_targets.append(int(relid))

    X, Y, Sources, Targets = create_data(train_sentences, train_targets)
    return X, Y

def get_batch_data():
    # Load data
    X, Y = load_train_data()
    
    # calc total batch count
    num_batch = len(X) // hp.batch_size
    # print(num_batch)

    state = np.random.get_state()
    np.random.shuffle(X)
    np.random.set_state(state)
    np.random.shuffle(Y)

    # # Convert to tensor
    # X = tf.convert_to_tensor(X, tf.int32)
    # Y = tf.convert_to_tensor(Y, tf.int32)

    for i in range(num_batch):
        batch_x = X[i*hp.batch_size: (i + 1)*hp.batch_size]
        batch_y = Y[i*hp.batch_size: (i + 1)*hp.batch_size]
        yield batch_x, batch_y

    # # Create Queues
    # input_queues = tf.train.slice_input_producer([X, Y])
    #
    # # create batch queues
    # x, y = tf.train.shuffle_batch(input_queues,
    #                             num_threads=12,
    #                             batch_size=hp.batch_size,
    #                             capacity=hp.batch_size*64,
    #                             min_after_dequeue=hp.batch_size*32,
    #                             allow_smaller_final_batch=False)
    #
    # #print(x.shape, y.shape, num_batch)
    # return x, y, num_batch # (N, T), (N, class_num), ()

