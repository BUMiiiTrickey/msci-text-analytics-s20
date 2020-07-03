import sys
import numpy as np
from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
from keras.models import load_model


def load_data(data_dir):
    with open('as4/data/sample_sentences.txt')as f:
        lines=f.readlines()
    voc_size=1000
    onehot1=[one_hot(words,voc_size)for words in lines]
    sent_lenth=494
    embedded_docs=pad_sequences(onehot1,padding='pre',maxlen=sent_lenth)
    x_pred=np.array(embedded_docs)

    return x_pred

def main(data_dir,model_code):
    if model_code == 'sigmoid':
        model = load_model('as4/data/nn_sigmoid.model')
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    elif model_code == 'relu':
        model = load_model('as4/data/nn_relu.model')
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    else:
        model = load_model('as4/data/nn_tanh.model')
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


    x_pred=load_data(data_dir)
    pred =model.predict_classes(x_pred)
    print(pred)

    return pred

if __name__ == '__main__':
    main(sys.argv[1],sys.argv[2])
