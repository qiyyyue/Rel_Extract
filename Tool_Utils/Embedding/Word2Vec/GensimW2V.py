from gensim.models import word2vec


data_path = '../../../Data/TrainData/open_data/corpus_segment.txt'
save_path = '../../../Model/Word2Vec/Tmp1/Word2VecModel'
print('loading sentences')
sentences = word2vec.LineSentence(data_path)
print('train model')
model = word2vec.Word2Vec(sentences, hs=1,min_count=1,window=3,size=300)
print('save model')
model.save(save_path)