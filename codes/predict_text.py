from konlpy.tag import Okt
import pandas as pd
import argparse
import numpy as np

from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics

from tensorflow.keras.models import load_model 
import json
import os
from pprint import pprint
import nltk


parser = argparse.ArgumentParser()
parser.add_argument("--text_path", type=str, default="../data/test.txt", help="predict words")
args = parser.parse_args()


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
        

    
def read_data(filename):
    with open(filename, 'r', encoding="utf-8") as f:
        data = [line for line in f.read().splitlines()]
    return data
        
        
        
with open('train_docs.json', encoding="utf-8") as f:
    train_docs = json.load(f)


tokens = [t for d in train_docs for t in d[0]]
text = nltk.Text(tokens, name='NMSC')

selected_words = [f[0] for f in text.vocab().most_common(10000)]

model = load_model('../first_model.h5')

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
             loss=losses.binary_crossentropy,
             metrics=[metrics.binary_accuracy])
datas = read_data(args.text_path)
for data in datas:
    predict_pos_neg(str(data))


