import sys
from gensim.models import Word2Vec
from out import read_dataset



def main(data_path):
    all_lines=read_dataset(data_path)
    print('Splitting lines in the dataset')
    all_lines = [line.strip().split() for line in all_lines]
    print('Training word2vec model')

    w2v = Word2Vec(all_lines, size=100, window=5, min_count=1, workers=4)
    w2v.save('Assignment3/data/w2v.model')

    w2v1=Word2Vec.load('/20863354_s647liu/Assignment3/data/w2v.model')
    print(w2v1.wv.most_similar('good',topn=20))
    print(w2v1.wv.most_similar('bad',topn=20))


if __name__ == '__main__':
    main(sys.argv[1])
