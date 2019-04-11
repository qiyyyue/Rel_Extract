# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
class Hyperparams:
    '''Hyperparameters'''
    # data
    train_data_base_dir = '../Data/TrainData/open_data/open_data'
    sent2rel_train_path = '../Data/TrainData/open_data/open_data/sent_relation_train.txt'
    train_senteces_path = '../Data/TrainData/open_data/pro_data/sent_train.txt'
    vocab_path = '../Data/TrainData/open_data/pro_data/vocab_cnt.txt'
    class_num = 35


    # training
    batch_size = 32 # alias = N
    lr = 0.0001 # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = '../Model/Transformer_log/' # log directory
    
    # model
    maxlen = 10 # Maximum number of words in a sentence. alias = T.
                # Feel free to increase this if you are ambitious.
    min_cnt = 20 # words whose occurred less than min_cnt are encoded as <UNK>.
    hidden_units = 512 # alias = C
    num_blocks = 6 # number of encoder/decoder blocks
    num_epochs = 20
    num_heads = 8
    dropout_rate = 0.2
    sinusoid = False # If True, use sinusoid. If false, positional embedding.
    
    
    
    