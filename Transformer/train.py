# -*- coding: utf-8 -*-

from __future__ import print_function
import tensorflow as tf
from sklearn import metrics

from Transformer.hyperparams import Hyperparams as hp
from Transformer.data_load import get_batch_data, load_vocab, load_dev_data
from Transformer.modules import *
from Transformer.transformer_model import Transformer_Model
import os, codecs
import time
from datetime import timedelta
from tqdm import tqdm

save_dir = '../CheckPionts/Transformer/Transformer_1'
tensorboard_dir = '../tensorboard/Transformer/Transformer_1'
save_path = os.path.join(save_dir, 'model')

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def evaluate(sess, model):

    X, Y = load_dev_data()

    indices = np.random.permutation(np.arange(len(X)))
    X = X[indices]
    Y = Y[indices]

    sum_loss = .0
    sum_acc = .0
    y_val_cls, y_pred_cls = np.zeros(shape=len(X), dtype=np.int32), np.zeros(shape=len(X), dtype=np.int32)
    for i in range(len(X) // hp.batch_size):
        start_id = i * hp.batch_size
        end_id = min((i + 1) * hp.batch_size, len(X))
        ### Get mini-batches
        x = X[i * hp.batch_size: (i + 1) * hp.batch_size]
        y = Y[i * hp.batch_size: (i + 1) * hp.batch_size]

        _loss, _acc = sess.run([model.loss, model.acc], {model.x: x, model.y: y})
        y_val_cls[start_id: end_id], y_pred_cls[start_id: end_id] = sess.run([model.y_label, model.preds], {model.x: x, model.y: y})

        sum_loss += _loss
        sum_acc += _acc
    print("Precision, Recall and F1-Score...")
    print(y_val_cls)
    print(y_pred_cls)
    print(metrics.classification_report(y_val_cls, y_pred_cls))

    # 混淆矩阵
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_val_cls, y_pred_cls)
    print(cm)

    print('val acc is {:.4f}'.format(sum_acc / (len(X) // hp.batch_size)))
    return sum_loss/(len(X) // hp.batch_size), sum_acc/(len(X) // hp.batch_size)

def train():
    g = Transformer_Model(_is_traing = True)
    print("Graph loaded")

    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    tf.summary.scalar("loss", g.loss)
    tf.summary.scalar("accuracy", g.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    # 配置 Saver
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    writer.add_graph(sess.graph)

    start_time = time.time()
    total_batch = 1  # 总批次
    best_acc_val = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    require_improvement = 2000  # 如果超过1000轮未提升，提前结束训练

    for epoch in range(1, hp.num_epochs + 1):
        for x, y in get_batch_data():
            feed_dict = {g.x: x, g.y: y}

            _preds = sess.run([g.preds], {g.x: x, g.y: y})
            _acc, _loss, _ = sess.run([g.acc, g.loss, g.train_op], feed_dict=feed_dict)
            print(_acc, _loss)
            if total_batch % hp.save_per_batch == 0:
                # 每多少轮次将训练结果写入tensorboard scalar
                s = sess.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % hp.print_per_batch == 0:
                # 每多少轮次输出在训练集和验证集上的性能
                loss_train, acc_train = sess.run([g.loss, g.acc], feed_dict=feed_dict)
                loss_val, acc_val = evaluate(sess, g)  # todo

                if acc_val > best_acc_val:
                    # 保存最好结果
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=sess, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

            sess.run(g.train_op, feed_dict=feed_dict)
            total_batch += 1
        #     if total_batch - last_improved > require_improvement:
        #         # 验证集正确率长期不提升，提前结束训练
        #         print("No optimization for a long time, auto-stopping...")
        #         flag = True
        #         break  # 跳出循环
        # if flag:  # 同上
        #     break

    print("Done")




if __name__ == '__main__':                
    train()
    

