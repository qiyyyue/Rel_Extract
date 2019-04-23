# -*- coding: utf-8 -*-
from __future__ import print_function

import random

from Transformer.hyperparams import Hyperparams as hp
import tensorflow as tf
import numpy as np
import codecs
import regex

def make_cnt_dict():
    cnt_dict = {0: [0], 1: [1, 10, 4, 33, 12, 16, 30, 11, 31, 32, 13, 19, 17, 34, 18], 2: [7, 5, 2, 3, 29, 23, 26], 3: [21, 20, 6, 27, 14, 8, 24, 18, 22, 15, 25, 9]}
    return cnt_dict

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

    cnt_dict = make_cnt_dict()
    word2id, id2word = load_vocab()
    # Index
    x_list, y_list, Sentences, Targets = [], [], [], []

    for sentence, target in zip(sentences, targets):
        x = [word2id.get(word, 1) for word in (sentence + u" </S>").split()] # 1: OOV, </S>: End of Text
        y = [0]*hp.class_num
        # y[target] = 1
        if target == 0:
            y[0] = 1
        else:
            y[1] = 1
        # # for i, rel_ids in cnt_dict:
        # #     if target in rel_ids:
        # #         y[i] = 1
        # #         break
        if len(x) <= hp.maxlen:
            x_list.append(np.array(x))
            y_list.append(np.array(y))
            Sentences.append(sentence)
            Targets.append(target)
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
    labels = [i for i in range(hp.class_num)]
    X, Y = sampling(X, Y, labels, hp.train_N)
    return X, Y
    
def load_dev_data():
    # 构建句子id到关系id映射表
    sent_rel_train = [line.strip().split('\t') for line in codecs.open(hp.sent2rel_dev_path, 'r', 'utf-8').readlines() if line]
    sentid2relid = {}
    for sentid, relids in sent_rel_train:
        sentid2relid[sentid] = relids.split()
    # 构建句子集合,对应的关系id集合
    train_sentences = []
    train_targets = []
    for train_sentid, _, _, train_sentence in [line.strip().split('\t') for line in codecs.open(hp.dev_senteces_path, 'r', 'utf-8').readlines() if line]:
        for relid in sentid2relid[train_sentid]:
            train_sentences.append(train_sentence)
            train_targets.append(int(relid))

    X, Y, Sources, Targets = create_data(train_sentences, train_targets)
    # labels = [i for i in range(hp.class_num)]
    # X, Y = sampling(X, Y, labels, hp.dev_N)
    return X, Y

def load_test_data():
    # 构建句子id到关系id映射表
    sent_rel_train = [line.strip().split('\t') for line in codecs.open(hp.sent2rel_test_path, 'r', 'utf-8').readlines() if line]
    sentid2relid = {}
    for sentid, relid in sent_rel_train:
        sentid2relid[sentid] = relid
    # 构建句子集合,对应的关系id集合
    test_sentences = []
    test_sent_ids = []
    for test_sent_id, _, _, test_sentence in [line.strip().split('\t') for line in codecs.open(hp.dev_senteces_path, 'r', 'utf-8').readlines() if line]:
        test_sentences.append(test_sentence)
        test_sent_ids.append(test_sent_id)

    word2id, id2word = load_vocab()
    # Index
    x_list = []
    id_list = []

    for test_sentence, test_id in zip(test_sentences, test_sent_ids):
        x = [word2id.get(word, 1) for word in (test_sentence + u" </S>").split()]
        if len(x) <= hp.maxlen:
            x_list.append(np.array(x))
            id_list.append(test_id)

    X = np.zeros([len(x_list), hp.maxlen], np.int32)
    for i, x in enumerate(x_list):
        X[i] = np.lib.pad(x, [0, hp.maxlen - len(x)], 'constant', constant_values=(0, 0))

    return X, id_list

def get_batch_data():
    # Load data
    X, Y = load_train_data()

    # calc total batch count
    num_batch = len(X) // hp.batch_size
    # print(num_batch)

    indices = np.random.permutation(np.arange(len(X)))
    X = X[indices]
    Y = Y[indices]

    # print(X.shape)
    # print(Y.shape)
    # print(np.sum(Y))
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

def sampling(org_X, org_Y, labels, N):
    new_X = []
    new_Y = []

    org_Y = np.argmax(org_Y, axis=-1)

    for label in labels:
        indices = np.where(org_Y==label)
        if len(indices[0]) == 0:
            continue
        label_X = org_X[indices]
        if len(label_X) < N:
            label_X = over_sampling(label_X, N)
        else:
            label_X = under_sampling(label_X, N)
        new_X += list(label_X)
        new_Y += list([label]*N)


    new_X = np.array(new_X)
    new_Y = np.array(new_Y)
    indices = np.random.permutation(np.arange(len(new_X)))
    new_X = new_X[indices]
    new_Y = new_Y[indices]
    new_Y = make_one_hot(new_Y)
    return new_X, new_Y

def under_sampling(org_X, N):
    random.shuffle(org_X)
    return org_X[:N]

def over_sampling(org_X, N):
    while(N > len(org_X)):
        org_X = np.concatenate([org_X, org_X], axis=0)
    random.shuffle(org_X)
    return org_X[:N]

def make_one_hot(org_Y):
    return (np.arange(hp.class_num)==org_Y[:,None]).astype(np.integer)

# labels = [i for i in range(35)]
# org_X, org_Y = load_dev_data()
# print(org_X.shape, org_Y.shape)
# new_X, new_Y = sampling(org_X, org_Y, labels, 3000)
# print(new_X.shape, new_Y.shape)
# X, Y = load_train_data()
# for i in range(hp.class_num):
#     indices = np.where(np.argmax(Y, axis=-1)==i)
#     print(len(indices[0]))
#     print(indices)
#     print('------------')