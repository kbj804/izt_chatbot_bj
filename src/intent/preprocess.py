# -*- encoding:utf-8 -*-

import os

from gensim.models import FastText
from konlpy.tag import Okt
import sys

path = os.getcwd()
print(path)
sys.path.append(path)

from src.intent.classifier import intent_mapping
from src.intent.configs import IntentConfigs
from src.util.tokenizer import tokenize
configs = IntentConfigs()

data = configs.data
vector_size = configs.vector_size


def preprocess_data(tokenizing):
    data['intent'] = data['intent'].map(intent_mapping)

    if tokenizing:
        count = 0
        for i in data['question']:
            data.replace(i, tokenize(i), regex=True, inplace=True)
            if count % 50 == 0:
                print("CURRENT COLLECT : ", count)
            count += 1

    encode = []
    decode = []
    for q, i in data.values:
        encode.append(q)
        decode.append(i)
    # encode: Question, decode: Intent Idx
    return {'encode': encode, 'decode': decode}


def train_vector_model(train_data_list, train):
    if train:
        mecab = Okt()
        str_buf = train_data_list['encode']
        joinString = ' '.join(str_buf)
        pos1 = mecab.pos(joinString)
        pos2 = ' '.join(list(map(lambda x: '\n' if x[1] in ['Punctuation'] else x[0], pos1))).split('\n')
        morphs = list(map(lambda x: mecab.morphs(x), pos2))
        print("BUILD MODEL")
        model = FastText(size=vector_size,
                         window=3,
                         workers=8,
                         min_count=1,
                         sg=1, #skipgram 모델의 성능이 더 좋다고 알려져있음
                         iter=1000)
        model.build_vocab(morphs)
        print("BUILD COMPLETE")

        print("TRAIN START")
        model.train(morphs, total_examples=model.corpus_count,
                    epochs=model.epochs,
                    compute_loss=True)
        if not os.path.exists('./fasttext'):
            os.makedirs('./fasttext')

        model.save('./fasttext/model_test')
        print("TRAIN COMPLETE")
        return model
    else:
        return FastText.load('./fasttext/model_test')


if __name__ == '__main__':

    train_data_list = preprocess_data(tokenizing=configs.tokenizing)
    train_vector_model(train_data_list,train=configs.train_fasttext)
