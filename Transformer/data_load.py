# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
from __future__ import print_function
from Transformer.hyperparams import Hyperparams as hp
import tensorflow as tf
import numpy as np
import codecs
import regex

def load_vocab():
    vocab = [line.split()[0] for line in codecs.open(hp.vocab_path, 'r', 'utf-8').read().splitlines() if int(line.split()[1]) >= hp.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def create_data(sentences, targets):

    word2id, id2word = load_vocab()
    # Index
    x_list, y_list, Sentences, Targets = [], [], [], []
    for sentence, target in zip(sentences, targets):
        x = [word2id.get(word, 1) for word in (sentence + u" </S>").split()] # 1: OOV, </S>: End of Text
        y = tf.one_hot(target, hp.class_num)
        if len(x) <=hp.maxlen:
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
    # de_sents = [regex.sub("[^\s\p{Latin}']", "", line) for line in codecs.open(hp.source_train, 'r', 'utf-8').read().split("\n") if line and line[0] != "<"]
    # en_sents = [regex.sub("[^\s\p{Latin}']", "", line) for line in codecs.open(hp.target_train, 'r', 'utf-8').read().split("\n") if line and line[0] != "<"]
    # 构建句子id到关系id映射表
    sent_rel_train = [line.strip().split() for line in codecs.open(hp.sent2rel_train_path, 'r', 'utf-8').readlines() if line]
    sentid2relid = {}

    for line_sent in sent_rel_train:
        sentid = line_sent[0]
        relids = line_sent[1:]
        sentid2relid[sentid] = relids
    #构建句子集合,对应的关系id集合
    train_sentences = []
    train_targets = []
    for train_sentid, train_sentence in [line.strip().split() for line in codecs.open(hp.train_senteces_path, 'r', 'utf-8').readlines() if line]:
        for relid in sentid2relid[train_sentid]:
            train_sentences.append(train_sentence)
            train_targets.append(relid)
        # train_sentences.append(train_sentence)
        # train_targets.append(sentid2relid[train_sentid])

    X, Y, Sources, Targets = create_data(train_sentences, train_targets)
    return X, Y
    
def load_test_data():
    # 构建句子id到关系id映射表
    sent_rel_train = [line.strip().split() for line in codecs.open(hp.sent2rel_test_path, 'r', 'utf-8').readlines() if line]
    sentid2relid = {}
    for sentid, relid in sent_rel_train:
        sentid2relid[sentid] = relid
    # 构建句子集合,对应的关系id集合
    train_sentences = []
    train_targets = []
    for train_sentid, train_sentence in [line.strip().split() for line in codecs.open(hp.train_senteces_path, 'r', 'utf-8').readlines() if line]:
        for relid in sentid2relid[train_sentid]:
            train_sentences.append(train_sentence)
            train_targets.append(relid)

    X, Y, Sources, Targets = create_data(train_sentences, train_targets)
    return X, Y

def get_batch_data():
    # Load data
    X, Y = load_train_data()
    
    # calc total batch count
    num_batch = len(X) // hp.batch_size
    
    # Convert to tensor
    X = tf.convert_to_tensor(X, tf.int32)
    Y = tf.convert_to_tensor(Y, tf.int32)
    
    # Create Queues
    input_queues = tf.train.slice_input_producer([X, Y])
            
    # create batch queues
    x, y = tf.train.shuffle_batch(input_queues,
                                num_threads=8,
                                batch_size=hp.batch_size, 
                                capacity=hp.batch_size*64,   
                                min_after_dequeue=hp.batch_size*32, 
                                allow_smaller_final_batch=False)

    print(x.shape, y.shape, num_batch)
    return x, y, num_batch # (N, T), (N, num), ()

get_batch_data()
