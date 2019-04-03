import tensorflow as tf
import numpy as np
import pandas as pd



class bilstm:

    def __init__(self, _ebd_dim, _hidden_dim, _num_classes, _dropout_rate):
        self.hidden_dim = _hidden_dim
        self.ebd_dim = _ebd_dim
        self.num_classes = _num_classes
        self.dropout_rate = _dropout_rate

    def build(self, is_training = True):

        x = tf.placeholder(dtype=tf.float32, shape=[None, self.ebd_dim], name='x')
        y = tf.placeholder(dtype=tf.float32, shape=[None, self.num_classes], name='y')
        seq_len = tf.placeholder(dtype=tf.int32, shape=[None])

        fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim)
        bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim)

        # dropout
        if is_training:
            fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=(1 - self.dropout_rate))
            bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=(1 - self.dropout_rate))

        outputs, f_states = tf.nn.bidirectional_dynamic_rnn(
            fw_cell,
            bw_cell,
            x,
            dtype=tf.float32,
            sequence_length=seq_len
        )


        w_output, b_output = outputs

        f_outputs = tf.concat([w_output, b_output], -1)

        


