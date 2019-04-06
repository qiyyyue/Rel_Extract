from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Tool_Utils.Embedding.Word2Vec.Prepare_Data import *
import argparse
import collections
import math
import os
import random
import sys
from tempfile import gettempdir
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.contrib.tensorboard.plugins import projector



vocabulary_size = 4000

batch_size = 128
cbow_window = 1
embedding_size = 128
num_sampled = 64

valid_size = 16
valid_window = 100
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

data_path = '../../../Data/CHN/DataSet1'
log_dir = 'log/Cbow_Model'

def word2vec_cbow():
    data, count, char2id, id2char = make_data(os.path.join(data_path, 'word2vec.train'), vocabulary_size)
    batch_gen = generate_cbow_batch(batch_size, cbow_window, data)

    graph = tf.Graph()

    with graph.as_default():
        # Input data.
        with tf.name_scope('inputs'):
            train_inputs = tf.placeholder(tf.int32, shape=[batch_size, 2*cbow_window])
            train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
            valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        # Look up embeddings for inputs.
        with tf.name_scope('embeddings'):
            embeddings = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

        # Construct the variables for the NCE loss
        with tf.name_scope('weights'):
            nce_weights = tf.Variable(
                tf.truncated_normal([vocabulary_size, embedding_size],
                                    stddev=1.0 / math.sqrt(embedding_size)))
        with tf.name_scope('biases'):
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

        embeds = None
        for i in range(2 * cbow_window):
            embedding_i = tf.nn.embedding_lookup(embeddings, train_inputs[:, i])
            emb_x, emb_y = embedding_i.get_shape().as_list()
            if embeds is None:

                embeds = tf.reshape(embedding_i, [emb_x, emb_y, 1])
            else:
                embeds = tf.concat([embeds, tf.reshape(embedding_i, [emb_x, emb_y, 1])], 2)
        avg_embed = tf.reduce_mean(embeds, 2, keep_dims=False)


        # Compute the average NCE loss for the batch.
        # tf.nce_loss automatically draws a new sample of the negative labels each
        # time we evaluate the loss.
        # Explanation of the meaning of NCE loss:
        #   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(
                tf.nn.nce_loss(
                    weights=nce_weights,
                    biases=nce_biases,
                    labels=train_labels,
                    inputs=avg_embed,
                    num_sampled=num_sampled,
                    num_classes=vocabulary_size))

        # Add the loss value as a scalar to summary.
        tf.summary.scalar('loss', loss)

        # Construct the SGD optimizer using a learning rate of 1.0.
        with tf.name_scope('optimizer'):
            optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

        # Compute the cosine similarity between minibatch examples and all
        # embeddings.
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,
                                                  valid_dataset)
        similarity = tf.matmul(
            valid_embeddings, normalized_embeddings, transpose_b=True)

        # Merge all summaries.
        merged = tf.summary.merge_all()

        # Add variable initializer.
        init = tf.global_variables_initializer()

        # Create a saver.
        saver = tf.train.Saver()

        # Step 5: Begin training.
        num_steps = 100001

        with tf.Session(graph=graph) as session:
            # Open a writer to write summaries.
            writer = tf.summary.FileWriter(log_dir, session.graph)

            # We must initialize all variables before we use them.
            init.run()
            print('Initialized')

            average_loss = 0
            for step in xrange(num_steps):
                batch_inputs, batch_labels = next(batch_gen)
                feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

                # Define metadata variable.
                run_metadata = tf.RunMetadata()

                # We perform one update step by evaluating the optimizer op (including it
                # in the list of returned values for session.run()
                # Also, evaluate the merged op to get all summaries from the returned
                # "summary" variable. Feed metadata variable to session for visualizing
                # the graph in TensorBoard.
                _, summary, loss_val = session.run([optimizer, merged, loss],
                                                   feed_dict=feed_dict,
                                                   run_metadata=run_metadata)
                average_loss += loss_val

                # Add returned summaries to writer in each step.
                writer.add_summary(summary, step)
                # Add metadata to visualize the graph for the last run.
                if step == (num_steps - 1):
                    writer.add_run_metadata(run_metadata, 'step%d' % step)

                if step % 2000 == 0:
                    if step > 0:
                        average_loss /= 2000
                    # The average loss is an estimate of the loss over the last 2000
                    # batches.
                    print('Average loss at step ', step, ': ', average_loss)
                    average_loss = 0

                # Note that this is expensive (~20% slowdown if computed every 500 steps)
                if step % 10000 == 0:
                    sim = similarity.eval()
                    for i in xrange(valid_size):
                        valid_word = id2char[valid_examples[i]]
                        top_k = 8  # number of nearest neighbors
                        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                        log_str = 'Nearest to %s:' % valid_word
                        for k in xrange(top_k):
                            close_word = id2char[nearest[k]]
                            log_str = '%s %s,' % (log_str, close_word)
                        print(log_str)
            final_embeddings = normalized_embeddings.eval()

            # Write corresponding labels for the embeddings.
            with open(log_dir + '/metadata.tsv', 'w') as f:
                for i in xrange(vocabulary_size):
                    f.write(id2char[i] + '\n')

            # Save the model for checkpoints.
            saver.save(session, os.path.join(log_dir, 'model.ckpt'))

            # Create a configuration for visualizing embeddings with the labels in
            # TensorBoard.
            config = projector.ProjectorConfig()
            embedding_conf = config.embeddings.add()
            embedding_conf.tensor_name = embeddings.name
            embedding_conf.metadata_path = os.path.join(log_dir, 'metadata.tsv')
            projector.visualize_embeddings(writer, config)
        writer.close()

word2vec_cbow()