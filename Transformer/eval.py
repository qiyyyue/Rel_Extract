# -*- coding: utf-8 -*-

from __future__ import print_function
import codecs
import os

import tensorflow as tf
import numpy as np

from Transformer.hyperparams import Hyperparams as hp
from Transformer.data_load import load_test_data, load_vocab
from Transformer.train import Rel_Ext_Graph
from nltk.translate.bleu_score import corpus_bleu

def eval(): 
    # Load graph
    g = Rel_Ext_Graph(is_training=False)
    print("Graph loaded")
    
    # Load data
    X, Y = load_test_data()
    word2idx, idx2word = load_vocab()
     
#     X, Sources, Targets = X[:33], Sources[:33], Targets[:33]
     
    # Start session         
    with g.graph.as_default():    
        sv = tf.train.Supervisor()
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ## Restore parameters
            sv.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))
            print("Restored!")

            for i in range(len(X) // hp.batch_size):

                ### Get mini-batches
                x = X[i*hp.batch_size: (i+1)*hp.batch_size]
                y = Y[i*hp.batch_size: (i+1)*hp.batch_size]

                _preds, _acc = sess.run(g.preds, g.acc, {g.x: x, g.y: y})
                print(_preds, _acc)
                     

                                          
if __name__ == '__main__':
    eval()
    print("Done")
    
    