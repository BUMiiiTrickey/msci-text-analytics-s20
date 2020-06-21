from gensim.models import Word2Vec

def main():

    w2v =Word2Vec.load('/20863354_s647liu/Assignment3/data/w2v.model')
    print('Most similar words to GOOD')
    print(w2v.wv.most_similar('good',topn=20))
    print('Most similar words to BAD')
    print(w2v.wv.most_similar('bad',topn=20))




if __name__ == '__main__':
    print(main())
