from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
import numpy as np
from konlpy.tag import Okt
import json
import os
from pprint import pprint
import nltk
import keras

def read_data(filename):
    with open(filename, 'r', encoding="utf-8") as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        # txt 파일의 헤더(id document label)는 제외하기
        data = data[1:]
    return data


def tokenize(doc):
    # norm은 정규화, stem은 근어로 표시하기를 나타냄
    okt = Okt()
    return ['/'.join(t) for t in okt.pos(doc, norm=True, stem=True)]


def term_frequency(doc):
    return [doc.count(word) for word in selected_words]


def predict_pos_neg(review):
    token = tokenize(review)
    tf = term_frequency(token)
    data = np.expand_dims(np.asarray(tf).astype('float32'), axis=0)
    score = float(model.predict(data))
    if(score > 0.5):
        print("[{}]는 {:.2f}% 확률로 긍정 리뷰이지 않을까 추측해봅니다.^^\n".format(review, score * 100))
    else:
        print("[{}]는 {:.2f}% 확률로 부정 리뷰이지 않을까 추측해봅니다.^^;\n".format(review, (1 - score) * 100))


def read_and_tokenize():
    train_data = read_data('../data/ratings_train.txt')
    test_data = read_data('../data/ratings_test.txt')
    if os.path.isfile('train_docs.json'):
        with open('train_docs.json', encoding="utf-8") as f:
            train_docs = json.load(f)
        with open('test_docs.json', encoding="utf-8") as f:
            test_docs = json.load(f)
    else:
        train_docs = [(tokenize(row[1]), row[2]) for row in train_data]
        test_docs = [(tokenize(row[1]), row[2]) for row in test_data]
        # JSON 파일로 저장
        with open('train_docs.json', 'w', encoding="utf-8") as make_file:
            json.dump(train_docs, make_file, ensure_ascii=False, indent="\t")
        with open('test_docs.json', 'w', encoding="utf-8") as make_file:
            json.dump(test_docs, make_file, ensure_ascii=False, indent="\t")
    return train_docs, test_docs


def modeling(x_train, y_train):
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=optimizers.RMSprop(lr=0.001),
                  loss=losses.binary_crossentropy,
                  metrics=[metrics.binary_accuracy])
    tb_hist = keras.callbacks.TensorBoard(log_dir='../graph', histogram_freq=0, write_graph=True,
                                          write_images=True)
    model.fit(x_train, y_train, epochs=25, batch_size=512, callbacks=[tb_hist])
    return model


def array_to_float(data):
    x_train = np.asarray(data).astype('float32')
    return x_train


def change_frequency(docs):
    train_x = [term_frequency(d) for d, _ in docs]
    train_y = [c for _, c in docs]
    return train_x, train_y


def make_selected_words(train_docs):
    tokens = [t for d in train_docs for t in d[0]]
    text = nltk.Text(tokens, name='NMSC')
    selected_words = [f[0] for f in text.vocab().most_common(10000)]
    return selected_words


train_docs, test_docs = read_and_tokenize()
selected_words = make_selected_words(train_docs)
train_x, train_y = change_frequency(train_docs)
print(train_y)
test_x, test_y = change_frequency(test_docs)
model = modeling(array_to_float(train_x), array_to_float(train_y))
results = model.evaluate(array_to_float(test_x), array_to_float(test_y))
model.save('../models/epoch_25_model.h5')

predict_pos_neg("올해 최고의 영화! 세 번 넘게 봐도 질리지가 않네요.")
predict_pos_neg("배경 음악이 영화의 분위기랑 너무 안 맞았습니다. 몰입에 방해가 됩니다.")
predict_pos_neg("주연 배우가 신인인데 연기를 진짜 잘 하네요. 몰입감 ㅎㄷㄷ")
predict_pos_neg("믿고 보는 감독이지만 이번에는 아니네요")
predict_pos_neg("주연배우 때문에 봤어요")
