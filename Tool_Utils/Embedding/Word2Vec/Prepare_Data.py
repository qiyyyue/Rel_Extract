import os
from typing import List

import numpy as np
import pandas as pd


data_path = '../../../Data/CHN/DataSet1'

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
    sentences = []
    for sentence in rf:
        sentences.append(sentence.strip())
    return sentences

def make_data(file_name: str) -> List[dict]:
    sentences = load_data(file_name)
    char2id = dict()
    char2id['UNK'] = 0
    for sentence in sentences:
        for char in sentence:
            if char not in char2id:
                char2id[char] = len(char2id)
    id2char = dict(zip(char2id.values(), char2id.keys()))
    return sentences, char2id, id2char

sentences, char2id, id2char = make_data(os.path.join(data_path, 'word2vec.train'))



