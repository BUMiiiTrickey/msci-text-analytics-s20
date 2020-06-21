import sys
import random


def read_dataset(data_path):
    with open("20863354_s647liu/Assignment3/data/pos.txt") as f:
        pos_lines = f.readlines()
    with open("20863354_s647liu/Assignment3/data/neg.txt") as f:
        neg_lines = f.readlines()
    return(pos_lines + neg_lines)


def main(data_path):
    all_lines = read_dataset(data_path)
    total_lines = len(all_lines)

    vocab = {}
    csv_data = ''
    train_data = ''
    val_data = ''
    test_data = ''
    labels_data = ''

    train_size = int(0.8 * total_lines)
    val_size = int(0.1 * total_lines)

    random.shuffle(all_lines)

    for idx, line in enumerate(all_lines):
        sentence = line[0].strip().split()
        label = line[1]

        no_punctuation_documents = []
        for i in sentence:
            no_punctuation_documents.append(i.translate(str.maketrans('', '', '!"#$%&()*+/:;<=>@[\\]^`{|}~\t\n')))

        csv_line = '{}\n'.format(','.join(no_punctuation_documents))
        csv_data += csv_line
        labels_data += '{}\n'.format(label)

        if idx < train_size:
            train_data += csv_line
        elif idx >= train_size and idx < train_size + val_size:
            val_data += csv_line
        else:
            test_data += csv_line

        for word in sentence:
            word = word.lower()
            if word not in vocab:
                vocab[word] = 0
            vocab[word] += 1



if __name__ == '__main__':
    main(sys.argv[1])
