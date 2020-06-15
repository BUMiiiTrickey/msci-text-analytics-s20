import pickle
import os
import sys
import csv
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def read_csv(data_path):
    with open(data_path) as f:
        data = f.readlines()
    return [' '.join(line.strip().split(',')) for line in data]


def load_data(data_dir):
    x_train = read_csv(os.path.join(data_dir, 'train.csv'))
    x_val = read_csv(os.path.join(data_dir, 'val.csv'))
    x_test = read_csv(os.path.join(data_dir, 'test.csv'))
    labels = read_csv(os.path.join(data_dir, 'labels.csv'))
    labels = [int(label) for label in labels]
    y_train = labels[:len(x_train)]
    y_val = labels[len(x_train): len(x_train) + len(x_val)]
    y_test = labels[-len(x_test):]
    return x_train, x_val, x_test, y_train, y_val, y_test


def train_uni(x_train, y_train):
    count_vect = CountVectorizer()
    x_train_count = count_vect.fit_transform(x_train)
    tfidf_transformer = TfidfTransformer()
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_count)
    clf = MultinomialNB().fit(x_train_tfidf, y_train)
    return clf, count_vect, tfidf_transformer

def train_bi(x_train, y_train):
    count_vect1 = CountVectorizer(ngram_range=(2, 2))
    x_train_count1 = count_vect1.fit_transform(x_train)
    tfidf_transformer1 = TfidfTransformer()
    x_train_tfidf1 = tfidf_transformer1.fit_transform(x_train_count1)
    clf1 = MultinomialNB().fit(x_train_tfidf1, y_train)
    return clf1, count_vect1, tfidf_transformer1

def train_uni_bi(x_train, y_train):
    count_vect2 = CountVectorizer(ngram_range=(1, 2))
    x_train_count2 = count_vect2.fit_transform(x_train)
    tfidf_transformer2 = TfidfTransformer()
    x_train_tfidf2 = tfidf_transformer2.fit_transform(x_train_count2)
    clf2 = MultinomialNB().fit(x_train_tfidf2, y_train)
    return clf2, count_vect2, tfidf_transformer2

def train_uni_ns(x_train, y_train):
    stop_words=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
           "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
           'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
           'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
           'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
           'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
           'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
           'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
           'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
           'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
           'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
           'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
           'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
           'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm',
           'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn',
           "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven',
           "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't",
           'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't",
           'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]


    count_vect3 = CountVectorizer(stop_words=stop_words)
    x_train_count3 = count_vect3.fit_transform(x_train)

    tfidf_transformer3 = TfidfTransformer()
    x_train_tfidf3 = tfidf_transformer3.fit_transform(x_train_count3)

    clf3 = MultinomialNB().fit(x_train_tfidf3, y_train)
    return clf3, count_vect3, tfidf_transformer3

def train_bi_ns(x_train, y_train):
    stop_words=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
           "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
           'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
           'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
           'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
           'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
           'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
           'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
           'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
           'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
           'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
           'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
           'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
           'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm',
           'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn',
           "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven',
           "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't",
           'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't",
           'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    count_vect4 = CountVectorizer(stop_words=stop_words, ngram_range=(2, 2))
    x_train_count4 = count_vect4.fit_transform(x_train)

    tfidf_transformer4 = TfidfTransformer()
    x_train_tfidf4 = tfidf_transformer4.fit_transform(x_train_count4)

    clf4 = MultinomialNB().fit(x_train_tfidf4, y_train)
    return clf4, count_vect4, tfidf_transformer4

def train_uni_bi_ns(x_train, y_train):
    stop_words=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
           "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
           'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
           'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
           'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
           'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
           'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
           'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
           'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
           'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
           'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
           'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
           'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
           'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm',
           'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn',
           "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven',
           "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't",
           'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't",
           'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

    count_vect5 = CountVectorizer(stop_words=stop_words,ngram_range=(1, 2))
    x_train_count5 = count_vect5.fit_transform(x_train)

    tfidf_transformer5 = TfidfTransformer()
    x_train_tfidf5 = tfidf_transformer5.fit_transform(x_train_count5)

    clf5 = MultinomialNB().fit(x_train_tfidf5, y_train)
    return clf5, count_vect5, tfidf_transformer5


def evaluate(x, y, clf, count_vect, tfidf_transformer):
    x_count = count_vect.transform(x)
    x_tfidf = tfidf_transformer.transform(x_count)
    preds = clf.predict(x_tfidf)
    return {'accuracy': accuracy_score(y, preds)}


def testing(x_test, clf, count_vect, tfidf_transformer):
    x_count = count_vect.transform(x_test)
    x_tfidf = tfidf_transformer.transform(x_count)
    preds_uni = clf.predict(x_tfidf)
    return preds_uni


def main(data_dir):
    x_train, x_val, x_test, y_train, y_val, y_test = load_data(data_dir)
    clf, count_vect, tfidf_transformer = train_uni(x_train, y_train)
    clf1, count_vect1, tfidf_transformer1 = train_bi(x_train, y_train)
    clf2, count_vect2, tfidf_transformer2 = train_uni_bi(x_train, y_train)
    clf3, count_vect3, tfidf_transformer3 = train_uni_ns(x_train, y_train)
    clf4, count_vect4, tfidf_transformer4 = train_bi_ns(x_train, y_train)
    clf5, count_vect5, tfidf_transformer5 = train_uni_bi_ns(x_train, y_train)

    scores = {}

    print('Validating')
    scores['uni'] = evaluate(x_test, y_test, clf, count_vect, tfidf_transformer)
    scores['bi'] = evaluate(x_test, y_test, clf1, count_vect1, tfidf_transformer1)
    scores['uni_bi'] = evaluate(x_test, y_test, clf2, count_vect2, tfidf_transformer2)
    scores['uni_ns'] = evaluate(x_test, y_test, clf3, count_vect3, tfidf_transformer3)
    scores['bi_ns'] = evaluate(x_test, y_test, clf4, count_vect4, tfidf_transformer4)
    scores['uni_bi_ns'] = evaluate(x_test, y_test, clf5, count_vect5, tfidf_transformer5)

    print('Testing')
    preds_uni = testing(x_test, clf, count_vect, tfidf_transformer)
    preds_bi = testing(x_test, clf1, count_vect1, tfidf_transformer1)
    preds_uni_bi = testing(x_test, clf2, count_vect2, tfidf_transformer2)
    preds_uni_ns = testing(x_test, clf3, count_vect3, tfidf_transformer3)
    preds_bi_ns = testing(x_test, clf4, count_vect4, tfidf_transformer4)
    preds_uni_bi_ns = testing(x_test, clf5, count_vect5, tfidf_transformer5)

    filename = 'mnb_uni.pkl'
    with open(filename, 'wb')as file:
        pickle.dump(clf, file)
    with open(filename, 'rb')as file:
        pickle_model = pickle.load(file)

    filename = 'mnb_bi.pkl'
    with open(filename, 'wb')as file:
        pickle.dump(clf1, file)
    with open(filename, 'rb')as file:
        pickle_model = pickle.load(file)

    filename = 'mnb_uni_bi.pkl'
    with open(filename, 'wb')as file:
        pickle.dump(clf2, file)
    with open(filename, 'rb')as file:
        pickle_model = pickle.load(file)

    filename = 'mnb_uni_ns.pkl'
    with open(filename, 'wb')as file:
        pickle.dump(clf3, file)
    with open(filename, 'rb')as file:
        pickle_model = pickle.load(file)

    filename = 'mnb_bi_ns.pkl'
    with open(filename, 'wb')as file:
        pickle.dump(clf4, file)
    with open(filename, 'rb')as file:
        pickle_model = pickle.load(file)

    filename = 'mnb_uni_bi_ns.pkl'
    with open(filename, 'wb')as file:
        pickle.dump(clf5, file)
    with open(filename, 'rb')as file:
        pickle_model = pickle.load(file)

    return scores,preds_uni,preds_bi,preds_uni_bi,preds_uni_ns,preds_bi_ns,preds_uni_bi_ns

if __name__ == '__main__':
    print(main(sys.argv[1]))
