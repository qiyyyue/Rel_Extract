import tensorflow as tf
import numpy as np
import pandas as pd

from tensorflow.examples.tutorials.mnist import input_data

batch_size = 128
is_training = False
embd_dim = 28
seq_len = 28
classes = 10
hidden_dim = 100
dropout_rate = 0.5


mnist = input_data.read_data_sets("data/mnist/", one_hot=True)

batch_x, batch_y = mnist.train.next_batch(batch_size)
batch_x = np.reshape(batch_x, [batch_size, 28, 28])
