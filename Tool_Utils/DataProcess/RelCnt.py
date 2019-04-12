import codecs
import sys
from collections import Counter

data_path = '../../Data/TrainData/open_data/open_data/sent_relation_train.txt'

rels_lines = [line.strip().split('\t')[-1].split() for line in codecs.open(data_path, 'r', 'utf-8').readlines() if line]
rels = []
for rels_line in rels_lines:
    for rel in rels_line:
        rels.append(rel)

rel_cnt = Counter(rels).most_common(len(rels))
print(len(rel_cnt))
wf = codecs.open('../../Data/TrainData/open_data/rel2cnt.txt', 'w', 'utf-8')
for rel, cnt in rel_cnt:
    print(rel, cnt)
    wf.write(u'{}\t{}\n'.format(rel, cnt))