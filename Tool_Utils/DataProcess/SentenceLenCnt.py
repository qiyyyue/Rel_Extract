import codecs
import sys
from collections import Counter

data_path = '../../Data/TrainData/open_data/open_data/sent_train.txt'

sentence_len = [len(line.strip().split('\t')[-1].split()) for line in codecs.open(data_path, 'r', 'utf-8').readlines() if line]

len_cnt = Counter(sentence_len).most_common(len(sentence_len))

wf = codecs.open('../../Data/TrainData/open_data/len2cnt', 'w', 'utf-8')
for senten_len, cnt in len_cnt:
    print(senten_len, cnt)
    wf.write(u'{}\t{}\n'.format(senten_len, cnt))