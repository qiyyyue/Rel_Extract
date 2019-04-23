import tensorflow as tf
import numpy as np
import pandas as pd

from Transformer.hyperparams import Hyperparams as hp
from Transformer.modules import *
from Transformer.data_load import load_vocab
from sklearn import metrics

class Transformer_Model(object):

    def __init__(self, _is_traing = True):
        self.x = tf.placeholder(tf.int32, shape=(None, hp.maxlen))
        self.y = tf.placeholder(tf.int32, shape=(None, hp.class_num))
        self.is_train = _is_traing

        self.build_model()

    def build_model(self):
        # Load vocabulary
        word2idx, idx2word = load_vocab()

        # Encoder
        with tf.variable_scope("encoder"):
            ## Embedding
            self.enc = embedding(self.x,
                                 vocab_size=len(word2idx),
                                 num_units=hp.hidden_units,
                                 zero_pad=False,
                                 scale=True,
                                 scope="enc_embed")

            # with tf.variable_scope('enc_embed', reuse=None):
            #     lookup_table = tf.get_variable('lookup_table',
            #                                    dtype=tf.float32,
            #                                    shape=[len(word2idx), hp.hidden_units],
            #                                    initializer=tf.contrib.layers.xavier_initializer())
            #     self.lookup_table = tf.concat((tf.zeros(shape=[1, hp.hidden_units]), lookup_table[1:, :]), 0)
            #     self.enc = tf.nn.embedding_lookup(lookup_table, self.x)
            #     self.enc = self.enc * (hp.hidden_units ** 0.5)

            # self.embd = self.enc

            key_masks = tf.expand_dims(tf.sign(tf.reduce_sum(tf.abs(self.enc), axis=-1)), -1)

            ## Positional Encoding
            if hp.sinusoid:
                self.enc += positional_encoding(self.x,
                                                num_units=hp.hidden_units,
                                                zero_pad=False,
                                                scale=False,
                                                scope="enc_pe")
            else:
                self.enc += embedding(
                    tf.tile(tf.expand_dims(tf.range(tf.shape(self.x)[1]), 0), [tf.shape(self.x)[0], 1]),
                    vocab_size=hp.maxlen,
                    num_units=hp.hidden_units,
                    zero_pad=False,
                    scale=False,
                    scope="enc_pe")

            # self.position = self.enc

            self.enc *= key_masks

            ## Dropout
            self.enc = tf.layers.dropout(self.enc,
                                         rate=hp.dropout_rate,
                                         training=tf.convert_to_tensor(self.is_train))

            ## Blocks
            for i in range(hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    ### Multihead Attention
                    self.enc = multihead_attention(queries=self.enc,
                                                   keys=self.enc,
                                                   num_units=hp.hidden_units,
                                                   num_heads=hp.num_heads,
                                                   dropout_rate=hp.dropout_rate,
                                                   is_training=self.is_train,
                                                   causality=False)
                    # self.mutihead = self.enc
                    ### Feed Forward
                    self.enc = feedforward(self.enc, num_units=[4 * hp.hidden_units, hp.hidden_units])
        # Final linear projection
        self.logits = tf.layers.dense(self.enc, hp.class_num)
        self.logits = tf.reshape(self.logits, [-1, hp.maxlen * hp.class_num])
        self.logits = tf.layers.dense(self.logits, hp.class_num)

        self.preds = tf.to_int32(tf.arg_max(tf.nn.softmax(self.logits, axis=-1), dimension=-1))
        self.y_label = tf.to_int32(tf.argmax(self.y, -1))
        self.true_preds = tf.equal(self.preds, self.y_label)


        self.acc = tf.reduce_mean(tf.cast(tf.equal(self.y_label, self.preds), tf.float32))

        if self.is_train:
            self.y_smoothed = label_smoothing(tf.cast(self.y, dtype=tf.float32))
            self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_smoothed)
            self.loss = tf.reduce_mean(self.loss)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

            # Summary
            tf.summary.scalar('mean_loss', self.loss)
            self.merged = tf.summary.merge_all()