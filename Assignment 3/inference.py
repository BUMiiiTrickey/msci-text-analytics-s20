import sys
from gensim.models import Word2Vec


def main(data_path):
    with open('20863354_s647liu/as3/data/sample_words.txt') as f:
        sample_text = f.readlines()
    sample_text = [w.strip() for w in sample_text]
    w2v = Word2Vec.load('20863354_s647liu/as3/data/w2v.model')
    return ['{} => {}'.format(w, [o[0] for o in w2v.most_similar([w2v[w]], topn=20)[1:]]) for w in sample_text]


if __name__ == '__main__':
    words = main(sys.argv[1])
    print('\n\n'.join(words))
