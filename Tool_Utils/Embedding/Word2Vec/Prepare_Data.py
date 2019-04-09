import collections
import os
from typing import List

import numpy as np
import pandas as pd
import random
import codecs
import regex
from collections import Counter

# data_path = '../../../Data/CHN/DataSet1'

def get_sentences(file_name: str) -> List[str]:
    rf = open(file_name)
    sentences = []
    sentence = ''
    for line in rf:
        if line == '\n' or len(line) == 0:
            sentences.append(str(sentence))
            sentence = ''
            continue
        sentence += str(line.strip().split(' ')[0])
    return sentences

def trans_data(org_files: List[str], des_file: str) -> None:
    sentences = []
    for org_file in org_files:
        sentences += get_sentences(org_file)
    wf = open(des_file, 'w')
    for sentence in sentences:
        wf.write(sentence + '\n')
    wf.flush()
    wf.close()

def load_data(file_name: str) -> List[str]:
    rf = open(file_name)
    sentences = ''
    for sentence in rf:
        sentences += sentence
    return sentences

def make_data(file_name: str, max_count) -> List[dict]:
    sentences = load_data(file_name)
    count = [['UNK', -1]]
    count.extend(collections.Counter(sentences).most_common(max_count - 1))
    dictionary = {}
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = []
    unk_count = 0
    for word in sentences:
        index = dictionary.get(word, 0)
        if index == 0:  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary

def generate_skip_batch(batch_size: int, num_skips: int, skip_window: int, data: List[int]):
    data_index = 0
    while True:
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        span = 2 * skip_window + 1  # [ skip_window target skip_window ]
        buffer = collections.deque(maxlen=span)  # pylint: disable=redefined-builtin
        if data_index + span > len(data):
            data_index = 0
        buffer.extend(data[data_index:data_index + span])
        data_index += span
        for i in range(batch_size // num_skips):
            context_words = [w for w in range(span) if w != skip_window]
            words_to_use = random.sample(context_words, num_skips)
            for j, context_word in enumerate(words_to_use):
                batch[i * num_skips + j] = buffer[skip_window]
                labels[i * num_skips + j, 0] = buffer[context_word]
            if data_index == len(data):
                buffer.extend(data[0:span])
                data_index = span
            else:
                buffer.append(data[data_index])
                data_index += 1
        # Backtrack a little bit to avoid skipping words in the end of a batch
        data_index = (data_index + len(data) - span) % len(data)
        yield batch, labels

def generate_cbow_batch(batch_size: int, cbow_window: int, data: List[int]):
    data_index = 0
    while True:
        span = 2 * cbow_window + 1
        # 去除中心word: span - 1
        batch = np.ndarray(shape=(batch_size, span - 1), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

        buffer = collections.deque(maxlen=span)
        for _ in range(span):
            buffer.append(data[data_index])
            # 循环选取 data中数据，到尾部则从头开始
            data_index = (data_index + 1) % len(data)

        for i in range(batch_size):
            # target at the center of span
            target = cbow_window
            # 仅仅需要知道context(word)而不需要word
            target_to_avoid = [cbow_window]

            col_idx = 0
            for j in range(span):
                # 略过中心元素 word
                if j == span // 2:
                    continue
                batch[i, col_idx] = buffer[j]
                col_idx += 1
            labels[i, 0] = buffer[target]
            # 更新 buffer
            buffer.append(data[data_index])
            data_index = (data_index + 1) % len(data)
        yield batch, labels

def make_word2cnt(in_fname, out_fname):
    text = codecs.open(in_fname, 'r', 'utf-8').read()
    text = regex.sub("\n", " ", text)
    words = text.split()
    word2cnt = Counter(words)
    with codecs.open(out_fname, 'w', 'utf-8') as fout:
        fout.write("{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n".format("<PAD>", "<UNK>", "<S>", "</S>"))
        for word, cnt in word2cnt.most_common(len(word2cnt)):
            print(str(word), str(cnt))
            fout.write(u"{}\t{}\n".format(word, cnt))

data_path = '../../../Data/TrainData/open_data'
in_fname = os.path.join(data_path, 'corpus_segment.txt')
out_fname = os.path.join(data_path, 'word2cnt')

make_word2cnt(in_fname, out_fname)

# data, count, char2id, id2char = make_data(os.path.join(data_path, 'word2vec.train'), 4000)
# print(data)
# print(count)
# print(char2id)
# print(id2char)
# print(len(char2id), len(id2char))

