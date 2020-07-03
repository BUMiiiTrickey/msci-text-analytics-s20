import os
import sys
import numpy as np
from tensorflow.python.keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout, Activation, Flatten
from keras.regularizers import l2
from keras.utils import to_categorical
from tensorflow.python.keras.preprocessing import sequence
from keras.callbacks import EarlyStopping



TRAINING_SIZE = 50000
DIGITS = 3
REVERSE = True

MAXLEN = DIGITS + 1 + DIGITS

# LOAD WORD EMBEDDING
embeddings_dict = {};

fd = open(os.path.join('', 'as4/data/embedding_word2vec.txt'),
            encoding="utf-8")
for line in fd:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:])
    embeddings_dict[word] = coefs

 # LOAD DATASET
with open('as4/data/train.csv') as f:
    train_texts = f.readlines()
with open('as4/data/val.csv') as f:
    val_texts = f.readlines()
with open('as4/data/test.csv') as f:
    test_texts = f.readlines()
with open('as4/data/out.csv') as f:
    texts = f.readlines()
with open('as4/data/labels.csv') as f:
    labels = f.readlines()
labels = [int(label) for label in labels]
y_train1 = labels[:len(train_texts)]
y_val1 = labels[len(train_texts): len(train_texts) + len(val_texts)]
y_test1 = labels[-len(test_texts):]

y_train = to_categorical(np.asarray(y_train1))
y_val = to_categorical(np.asarray(y_val1))
y_test = to_categorical(np.asarray(y_test1))



print('converting the word embedding into tokenized vector')
tokenizer_obj = Tokenizer()
tokenizer_obj.fit_on_texts(texts)
train_sequences = tokenizer_obj.texts_to_sequences(train_texts)
val_sequences = tokenizer_obj.texts_to_sequences(val_texts)
test_sequences = tokenizer_obj.texts_to_sequences(test_texts)


TOP_K = 20000
MAX_SEQUENCE_LENGTH = 500
max_length = len(max(train_texts, key=len))
if max_length > MAX_SEQUENCE_LENGTH:
    max_length = MAX_SEQUENCE_LENGTH
pad_train = sequence.pad_sequences(train_sequences, maxlen=max_length)
pad_val = sequence.pad_sequences(val_sequences, maxlen=max_length)
pad_test = sequence.pad_sequences(test_sequences, maxlen=max_length)


embeddings_matrix = np.random.uniform(-0.05, 0.05, size=(
len(tokenizer_obj.word_index) + 1, len(coefs)))

for word, i in tokenizer_obj.word_index.items():
    try:
        embeddings_vector = embeddings_dict[word]
    except KeyError:
        embeddings_vector = None
    if embeddings_vector is not None:
        embeddings_matrix[i] = embeddings_vector
del embeddings_dict

print('Shape of train tensor:', pad_train.shape)
print('Shape of val tensor:', pad_val.shape)
print('Shape of test tensor:', pad_test.shape)

def main(data_dir):

    sig_model = Sequential()
    sig_model.add(Embedding(input_dim=len(tokenizer_obj.word_index) + 1,
                        output_dim=len(coefs), input_length=max_length,
                        weights=[embeddings_matrix], trainable=False, name='word_embedding_layer'))

    sig_model.add(Dense(64, activation='sigmoid', kernel_regularizer=l2(0.001)))
    sig_model.add(Flatten())
    sig_model.add(Dropout(0.4))
    sig_model.add(Dense(2, activation='softmax', kernel_regularizer=l2(0.001), name='output_layer'))
    sig_model.summary()
    sig_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    es=EarlyStopping(monitor='val_loss',mode='min',verbose=1)

    sig_model.fit(pad_train, y_train, batch_size=1024, epochs=10, validation_data=(pad_val, y_val),callbacks=[es])

    score, acc = sig_model.evaluate(pad_test,y_test,
                                batch_size=1024)
    print("Accuracy on the test set = {0:4.3f}".format(acc))
    sig_model.save('as4/data/nn_sigmoid.model')


    relu_model = Sequential()
    relu_model.add(Embedding(input_dim=len(tokenizer_obj.word_index) + 1,
                        output_dim=len(coefs), input_length=max_length,
                        weights=[embeddings_matrix], trainable=False, name='word_embedding_layer'))

    relu_model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
    relu_model.add(Flatten())
    relu_model.add(Dropout(0.4))
    relu_model.add(Dense(2, activation='softmax', kernel_regularizer=l2(0.001), name='output_layer'))
    relu_model.summary()
    relu_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    es=EarlyStopping(monitor='val_loss',mode='min',verbose=1)
    relu_model.fit(pad_train, y_train, batch_size=1024, epochs=10, validation_data=(pad_val, y_val),callbacks=[es])

    score, acc = relu_model.evaluate(pad_test,y_test,
                                batch_size=1024)
    print("Accuracy on the test set = {0:4.3f}".format(acc))
    relu_model.save('nn_relu.model')


    tanh_model = Sequential()
    tanh_model.add(Embedding(input_dim=len(tokenizer_obj.word_index) + 1,
                        output_dim=len(coefs), input_length=max_length,
                        weights=[embeddings_matrix], trainable=False, name='word_embedding_layer'))

    tanh_model.add(Dense(64, activation='tanh', kernel_regularizer=l2(0.001)))
    tanh_model.add(Flatten())
    tanh_model.add(Dropout(0.4))
    tanh_model.add(Dense(2, activation='softmax', kernel_regularizer=l2(0.001), name='output_layer'))
    tanh_model.summary()
    tanh_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    es=EarlyStopping(monitor='val_loss',mode='min',verbose=1)
    tanh_model.fit(pad_train, y_train, batch_size=1024, epochs=10, validation_data=(pad_val, y_val),callbacks=[es])

    score, acc = tanh_model.evaluate(pad_test,y_test,
                                batch_size=1024)
    print("Accuracy on the test set = {0:4.3f}".format(acc))
    tanh_model.save('nn_tanh.model')

if __name__ == '__main__':
    main(sys.argv[1])
