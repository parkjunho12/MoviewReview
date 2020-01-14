from konlpy.tag import Okt
import pandas as pd
okt = Okt()


import json
import os
from pprint import pprint

def tokenize(doc):
    # norm은 정규화, stem은 근어로 표시하기를 나타냄
    return ['/'.join(t) for t in okt.pos(doc, norm=True, stem=True)]

if os.path.isfile('train_docs.json'):
    with open('train_docs.json') as f:
        train_docs = json.load(f)
    with open('test_docs.json') as f:
        test_docs = json.load(f)
tokens = [t for d in train_docs for t in d[0]]


import nltk
text = nltk.Text(tokens, name='NMSC')


selected_words = [f[0] for f in text.vocab().most_common(10000)]

def term_frequency(doc):
    return [doc.count(word) for word in selected_words]

test_x = [term_frequency(d) for d, _ in test_docs]
test_y = [c for _, c in test_docs]


import numpy as np

x_test = np.asarray(test_x).astype('float32')

y_test = np.asarray(test_y).astype('float32')

import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics

from tensorflow.keras.models import load_model 


model = load_model('../first_model.h5')

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
             loss=losses.binary_crossentropy,
             metrics=[metrics.binary_accuracy])

results = model.evaluate(x_test, y_test)


def predict_pos_neg(review, label):
    token = tokenize(review)
    tf = term_frequency(token)
    data = np.expand_dims(np.asarray(tf).astype('float32'), axis=0)
    score = float(model.predict(data))
    if(score > 0.5):
        print("[{}]는 {:.2f}% 확률로 긍정 리뷰이지 않을까 추측해봅니다.^^\n".format(review, score * 100))
        if str(label) == "1":
            return 1
        else:
            return 0 
    else:
        print("[{}]는 {:.2f}% 확률로 부정 리뷰이지 않을까 추측해봅니다.^^;\n".format(review, (1 - score) * 100))
        if str(label) == "0":
            return 1
        else:
            return 0



count_hit = predict_pos_neg("너무 재미 없어 ㄹㅇ 개노잼..",0)
for i in range(10):
	count_hit += predict_pos_neg("재미지다 ㄹㅇ 재밌어",1)
print(count_hit)


