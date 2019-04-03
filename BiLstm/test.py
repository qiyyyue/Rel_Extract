import tensorflow as tf
import numpy as np
import pandas as pd

from tensorflow.examples.tutorials.mnist import input_data



batch_size = 128
is_training = False


embd_dim = 728
classes = 10

hidden_dim = 100

dropout_rate = 0.5


mnist = input_data.read_data_sets("data/mnist/", one_hot=True)

batch_x, batch_y = mnist.train.next_batch(batch_size)
batch_x = np.reshape(batch_x, [batch_size, 28, 28])
print(batch_x.shape)




def bilstm():
    x = tf.placeholder(tf.float32, [batch_size, 28, 28], 'x')
    y = tf.placeholder(tf.float32, [batch_size, classes], 'y')


    fw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim)
    bw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim)

    # dropout
    if is_training:
        fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=(1 - dropout_rate))
        bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=(1 - dropout_rate))

    outputs, f_states = tf.nn.bidirectional_dynamic_rnn(
        fw_cell,
        bw_cell,
        x,
        sequence_length=[28]*batch_size,
        dtype=tf.float32,
    )


    w_output, b_output = outputs

    f_outputs = tf.concat([w_output, b_output], -1)

    w = tf.

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        tmp_f, tmp_out = sess.run([f_outputs, outputs], feed_dict={x: batch_x, y: batch_y})
        tmp_w, tmp_b = tmp_out
        print(tmp_w.shape, tmp_b.shape)

        print(tmp_f.shape)


bilstm()