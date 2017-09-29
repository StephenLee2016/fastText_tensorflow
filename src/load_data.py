__author__ = 'jrlimingyang@jd.com'

import codecs
from gensim import corpora
from collections import defaultdict
import numpy as np

def invert_dict(d):
    return dict([(v, k) for k,v in d.items()])

def stats_words_labels(train_path, test_path):
    words = []
    labels = []
    f = codecs.open(train_path, encoding='utf-8')
    for line in f.readlines():
        line = line.strip('\n').strip()
        tokens = line.split()
        current_labels = []
        current_words = []
        for token in tokens:
            if token.startswith('__label__'):
                current_labels.append(token)
            else:
                if token != '1':
                    current_words.append(token)
        words.append(current_words)
        labels.append(current_labels)
    f.close()
    f = codecs.open(test_path, encoding='utf-8')
    for line in f.readlines():
        line = line.strip('\n').strip()
        tokens = line.split()
        current_labels = []
        current_words = []
        for token in tokens:
            if token.startswith('__label__'):
                current_labels.append(token)
            else:
                if token != '1':
                    current_words.append(token)
        words.append(current_words)
        labels.append(current_labels)
    f.close()

    dic_words = corpora.Dictionary(words)
    max_words = max([len(text) for text in words])
    min_words = min([len(text) for text in words])
    print ('words dict: '+ str(len(dic_words)))
    print (str(max_words) + ' ' + str(min_words))
    dic_words.save('E:\\PycharmProject\\FastText_Tensorflow\\data\\words.dict')
    dic_labels = corpora.Dictionary(labels)
    max_labels = max([len(text) for text in labels])
    min_labels = min([len(text) for text in labels])
    print ('labels dict: '+ str(len(dic_labels)))
    print (str(max_labels)+ ' ' + str(min_labels))
    dic_labels.save('E:\\PycharmProject\\FastText_Tensorflow\\data\\labels.dict')

def transfer_label(labels):
    label_size = 736
    vec = np.zeros(label_size)
    for ids in labels:
        vec[ids] = 1.0
    return vec

def create_ngram_set(input_list, ngram_value=2):
    '''
        提取输入的n-gram特征

        create_ngram_set([我, 是, 中国, 人], ngram_value=2)
            output --> {(我,是),(是,中国),(中国,人)}
    '''
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))

def add_ngram(sequences, token_indice, ngram_range=2):
    '''
        将n-gram特征的计数也当做特征插到句子中

        add_ngram(sequences, token_indice, ngram_range=2)
        > sequences = [[我, 是, 中国, 人], [我, 是, 男人, 怎么, 啦]]
        > token_indice {(我, 是): 1337, (是, 中国): 42, (怎么, 啦): 2016}
            output --> [[我, 是, 中国, 人, 1337, 42], [我, 是, 男人, 怎么, 啦, 1337, 2016]]
    '''
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for i in range(len(new_list) - ngram_range + 1):
            for ngram_value in range(2, ngram_range + 1):
                ngram = tuple(new_list[i: i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)
    return new_sequences



def get_data(file, ngram_range):
    words_dict = invert_dict(corpora.Dictionary.load('E:\\PycharmProject\\FastText_Tensorflow\\data\\words.dict'))
    labels_dict = invert_dict(corpora.Dictionary.load('E:\\PycharmProject\\FastText_Tensorflow\\data\\labels.dict'))
    paths_length = np.load('E:\\PycharmProject\\FastText_Tensorflow\\data\\paths_length.npy')
    train_data = []
    train_label = []
    train_label_num = []
    labels = []
    train_paths_length = []
    f = codecs.open(file, encoding='utf-8')
    ngram_set = set()
    for line in f.readlines():
        line = line.strip('\n').strip()
        tokens = line.split()
        current_labels = []
        current_words = []
        for token in tokens:
            if token.startswith('__label__'):
                current_labels.append(labels_dict[token])
            else:
                if token != '1':
                    current_words.append(words_dict[token])

        current_path_length = [paths_length[label] for label in current_labels]
        train_paths_length.append(current_path_length)
        train_label_num.append(len(current_labels))
        labels.extend(current_labels)
        train_data.append(current_words)
        train_label.append(current_labels)
        # 添加n-gram特征

        for i in range(2, ngram_range+1):
            set_of_ngram = create_ngram_set(current_words, ngram_value=i)
            ngram_set.update(set_of_ngram)

    max_features = len(words_dict)
    start_index = max_features + 1
    token_indice = {v:k + start_index for k, v in enumerate(ngram_set)}
    print (len(token_indice.keys()))
    indice_token = {token_indice[k]: k for k in token_indice}

    max_features = np.max(len(indice_token.keys())) + 1

    train_data = add_ngram(train_data, token_indice, ngram_range)

    #     current_path_length = [paths_length[label] for label in current_labels]
    #     train_paths_length.append(current_path_length)
    #     train_label_num.append(len(current_labels))
    #     labels.extend(current_labels)
    #     train_data.append(current_words)
    #     train_label.append(current_labels)
    #
    return train_data, train_label, labels, train_label_num, train_paths_length

def load_cooking_data():
    stats_words_labels('E:\\PycharmProject\\FastText_Tensorflow\\data\\cooking.train', 'E:\\PycharmProject\\FastText_Tensorflow\\data\\cooking.valid')
    train_data, train_label, train_counts, train_label_num, train_path_length = get_data('E:\\PycharmProject\\FastText_Tensorflow\\data\\cooking.train', ngram_range=2)
    test_data, test_label, test_counts, test_label_num, test_path_length = get_data('E:\\PycharmProject\\FastText_Tensorflow\\data\\cooking.valid', ngram_range=2)
    train_counts.extend(test_counts)
    counts = [train_counts.count(item) for item in set(train_counts)]
    return train_data, train_label, train_label_num, train_path_length, \
           test_data, test_label, test_label_num, test_path_length, counts