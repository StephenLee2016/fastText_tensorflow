__author__ = 'jrlimingyang@jd.com'

import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import GlobalMaxPooling1D
from keras.datasets import imdb

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
                ngram = tuple(new_list[i:i+ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)

    return new_sequences

# 设置超参数
ngram_range = 1
max_features = 20000
maxlen = 400
batch_size = 32
embedding_dims = 50
epochs = 5

print ('Loading data ...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print (len(x_train), 'train sequences')
print (len(x_test), 'test sequences')
print ('Average train sequence length: {}'.format(np.mean(list(map(len, x_train)), dtype=int)))
print ('Average test sequence length: {}'.format(np.mean(list(map(len, x_test)), dtype=int)))

if ngram_range > 1:
    print ('Adding {}=gram features'.format(ngram_range))
    # create set of unique n-gram from the training set.
    ngram_set = set()
    for input_list in x_train:
        for i in range(2, ngram_range + 1):
            set_of_ngram = create_ngram_set(input_list, ngram_value=i)
            ngram_set.update(set_of_ngram)

    # Dictionary mapping n-gram token to a unique integer.
    # Integer values are greater than max_features in order
    # to avoid collision with existing features
    start_index = max_features + 1
    token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
    indice_token = {token_indice[k]: k for k in token_indice}

    # max_features is the highest integer that could be found in the dataset
    max_features = np.max(list(indice_token.keys())) + 1

    # Augmenting x_train and x_test with n-gram features
    x_train = add_ngram(x_train, token_indice, ngram_range)
    x_test = add_ngram(x_test, token_indice, ngram_range)
    print ('Average train sequence length: {}'.format(np.mean(list(map(len, x_train)), dtype=int)))
    print ('Average test sequence length: {}'.format(np.mean(list(map(len, x_test)), dtype=int)))

print ('Pad sequences (sample x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print ('x_train shape:', x_train.shape)
print ('x_test shape:', x_test.shape)

print ('Build model ...')
model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen))

# we add a GlobalAveragePooling1D, which will average the embeddings
# of all words in the document
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))