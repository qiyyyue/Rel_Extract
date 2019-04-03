import tensorflow as tf
import numpy as np
import pandas as pd



class bilstm:

    def __init__(self, _batch_size, _seq_len, _ebd_dim, _hidden_dim, _num_classes, _dropout_rate, _learning_rate, _num_epochs):
        self.batch_size = _batch_size
        self.seq_len = _seq_len
        self.ebd_dim = _ebd_dim
        self.hidden_dim = _hidden_dim
        self.num_classes = _num_classes
        self.dropout_rate = _dropout_rate
        self.learning_rate = _learning_rate
        self.num_epochs = _num_epochs


    def build(self, is_training = True):

        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.seq_len, self.ebd_dim], name='x')
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, self.seq_len, self.num_classes], name='y')
        self.y = tf.reshape(self.y, [-1, self.num_classes])

        length = tf.reduce_sum(tf.sign(self.x), reduction_indices=1)
        length = tf.cast(self.x, tf.int32)
        # forward lstm cell and backward lstm cell
        fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim)
        bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim)

        # dropout
        if is_training:
            fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=(1 - self.dropout_rate))
            bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=(1 - self.dropout_rate))

        # bi lstm
        outputs, f_states = tf.nn.bidirectional_dynamic_rnn(
            fw_cell,
            bw_cell,
            x,
            dtype=tf.float32,
            sequence_length=length
        )

        # out put of forward lstm and backward lst
        w_output, b_output = outputs

        # concat
        f_outputs = tf.concat([w_output, b_output], -1)
        f_outputs = tf.reshape(f_outputs, [-1, 2*self.hidden_dim])
        #
        w = tf.get_variable('w', [2*self.hidden_dim, self.num_classes])
        b = tf.get_variable('b', [self.num_classes])

        logits = tf.add(tf.matmul(f_outputs, w), b)

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
        self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)

        self.correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

    def train(self, sess, save_file, X_train, y_train, X_val, y_val):
        saver = tf.train.Saver()
        num_iterations = int(np.math.ceil(1.0 * len(X_train) / self.batch_size))

        for epoch in range(self.num_epochs):
            # shuffle train in each epoch
            sh_index = np.arange(len(X_train))
            np.random.shuffle(sh_index)
            X_train = X_train[sh_index]
            y_train = y_train[sh_index]
            print ("current epoch: %d" % (epoch))
            for iteration in range(num_iterations):
                _, train_loss = sess.run([self.opt, self.loss], feed_dict = {self.x: X_train, self.y: y_train})

                if iteration%20 == 0:
                    print('loss: ', train_loss)
        


