# coding=utf8
import pandas as pd
import matplotlib.pyplot as plt
import re

import tensorboard
from keras import optimizers
from konlpy import jvm
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from keras.models import load_model
import keras


def reviewTokenize(data):
    data = data.dropna(how='any')
    data['document'] = data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")
    return data


def storeTrainData(data, train_data):
    X_train = []

    # 한글과 공백을 제외하고 모두 제거
    okt = Okt()
    stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']
    for sentence in data['document']:
        temp_X = []
        if type(sentence) is str:
            temp_X = okt.morphs(sentence, stem=True)  # 토큰화
            temp_X = [word for word in temp_X if not word in stopwords]  # 불용어 제거
            X_train.append(temp_X)
            continue
        else:
            print(sentence)
    return X_train


def encodingData(X_train, X_test):
    max_words = 35000
    tokenizer = Tokenizer(num_words=max_words)  # 상위 35,000개의 단어만 보존
    tokenizer.fit_on_texts(X_train)
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)
    return X_train, X_test


def matchLength(X_train, X_test):
    max_len = 30
    X_train = pad_sequences(X_train, maxlen=max_len)
    X_test = pad_sequences(X_test, maxlen=max_len)
    return X_train, X_test


def modelingData(X_train, y_train, max_words=35000):
    model = Sequential()
    model.add(Embedding(max_words, 100))
    model.add(LSTM(128))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    tb_hist = keras.callbacks.TensorBoard(log_dir='../graph', histogram_freq=0, write_graph=True,
                                          write_images=True)
    model.fit(X_train, y_train, epochs=10, batch_size=60, validation_split=0.2, callbacks=[tb_hist])

    print("\n 테스트 정확도: %.4f\n" % (model.evaluate(X_test, y_test)[1]))
    predictEmotion(model, last_data, last_pre_data)
    return model


def predictEmotion(model, last_data, last_pre_data):
    predicts = model.predict_classes(X_test[:])
    predics = model.predict(X_test[:])
    search = 0
    all_data = 0
    hit_data = 0
    for i in range(len(predics)):
        if last_pre_data[i] == 1:
            print("긍정" + str(last_pre_data[i]) + "\n")
        else:
            print("부정" + str(last_pre_data[i]) + "\n")
        print("값이 뭔가요?" + str(predicts[i]) + "\n")
        if predics[i] > 0.5:
            print(str(last_data[i]) + "  긍정  " + str(predics[i]))
        else:
            print(str(last_data[i]) + "  부정  " + str(predics[i]))
        if last_pre_data[i] == predicts[i]:
            search += 1
            hit_data += 1
        all_data += 1
        print("\n")
        if i % 10 == 0:
            print("몇개 맞췄나요? " + str(search) + "/ 10\n")
            search = 0
    percentage = hit_data / all_data * 100
    print("모든 데이터 맞춘 수: " + str(hit_data) + "/" + str(all_data))
    print("\n 맞춘 퍼센테이지 : " + str(percentage) + "\n")


if __name__ == '__main__':
    train_data = pd.read_table('../data/ratings_train.txt')
    test_data = pd.read_table('../data/ratings_test.txt')
    jvm.init_jvm()
    last_data = np.array(test_data['document'])
    last_pre_data = np.array(test_data['label'])
    train_data = reviewTokenize(train_data)
    test_data = reviewTokenize(test_data)
    trained_data = storeTrainData(train_data, train_data)
    tested_data = storeTrainData(test_data, test_data)
    X_train, X_test = encodingData(trained_data, tested_data)
    X_train, X_test = matchLength(X_train, X_test)
    y_train = np.array(train_data['label'])
    y_test = np.array(test_data['label'])
    model = modelingData(X_train, y_train, 35000)
    model.save('../models/first_model.h5')
