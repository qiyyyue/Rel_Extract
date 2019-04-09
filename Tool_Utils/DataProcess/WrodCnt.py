import codecs
import os
from collections import Collection, Counter
from typing import List

data_path = '../../Data/TrainData/open_data/open_data'

file_list = ['sent_test.txt', 'sent_dev.txt', 'sent_train.txt']


word_list = []


def word_cnt(file_name: str) -> List[str]:
    rf = open(file_name, 'r', encoding='utf8')
    tmp_word_list = []
    for line in rf:
        try:
            ents = line.strip().split('\t')[1:3]
            tmp_word_list += ents
            #print(ents)
        except Exception as e:
            print('erro line: ' + line)
            pass
    return tmp_word_list

for file in file_list:
    file_name = os.path.join(data_path, file)
    word_list += word_cnt(file_name)

word_cnt_dict = Counter(word_list)


save_file = os.path.join(data_path, 'entity_cnt.txt')
wf = codecs.open(save_file, 'w', encoding='utf8')
for word, cnt in word_cnt_dict.most_common(len(word_cnt_dict)):
    print(str(word), str(cnt))
    wf.write(u'{}\t{}\n'.format(word, cnt))



