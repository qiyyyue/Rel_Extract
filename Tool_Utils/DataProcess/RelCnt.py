import codecs
import sys
from collections import Counter

data_path = '../../Data/TrainData/open_data/open_data/sent_relation_dev.txt'

rels_lines = [line.strip().split('\t')[-1].split() for line in codecs.open(data_path, 'r', 'utf-8').readlines() if line]
rels = []
for rels_line in rels_lines:
    for rel in rels_line:
        rels.append(rel)

rel_cnt = Counter(rels).most_common(len(rels))

# max_cnt = max([cnt for _, cnt in rel_cnt])
sum_cnt = sum([cnt for _, cnt in rel_cnt])
# print(max_cnt, sum_cnt)
# print(int(max_cnt)/sum_cnt)

print(len(rel_cnt))
wf = codecs.open('../../Data/TrainData/open_data/dev_rel2cnt.txt', 'w', 'utf-8')
for rel, cnt in rel_cnt:
    print(rel, cnt, '%.5f'%(cnt/sum_cnt))
    wf.write(u'{}\t{}\t{:.5f}\n'.format(rel, cnt, cnt/sum_cnt))