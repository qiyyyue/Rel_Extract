import codecs
import re

import jieba.posseg as psg
import os
from collections import Collection, Counter
from typing import List

org_base_path = '../../Data/TrainData/open_data/open_data'
des_base_path = '../../Data/TrainData/open_data/pro_data'

file_list = ['sent_test.txt', 'sent_dev.txt', 'sent_train.txt']


def repalce_person_name(org_file: str, des_file: str):
    rf = codecs.open(org_file, 'r', encoding='utf8')
    wf = codecs.open(des_file, 'w', encoding='utf8')

    for line in rf:
        sent_id, p1, p2, sentence = line.strip().split('\t')
        sentence = sentence.replace(p1, 'PN').replace(p2, 'PN')
        wf.write(u'{}\t{}\t{}\t{}\n'.format(sent_id, p1, p2, sentence))
for file in file_list:
    org_file = os.path.join(org_base_path, file)
    des_file = os.path.join(des_base_path, file)
    repalce_person_name(org_file, des_file)