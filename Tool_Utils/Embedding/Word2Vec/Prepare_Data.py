import collections
import os
from typing import List

import numpy as np
import pandas as pd
import random

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


def generate_batck(batch_size: int, num_skips: int, skip_window: int, data: List[int]):
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



# data, count, char2id, id2char = make_data(os.path.join(data_path, 'word2vec.train'), 4000)
# print(data)
# print(count)
# print(char2id)
# print(id2char)
# print(len(char2id), len(id2char))

