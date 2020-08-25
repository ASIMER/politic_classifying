import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec
from gensim.utils import simple_preprocess

version = 17

model_name = "models/teached_d2w_v{}".format(version)

textes = pd.read_csv("data/cleaned_textes_v{}.csv".format(version))
model = Doc2Vec.load(model_name)
textes = textes.truncate(after=1919)
print(textes['class'].describe())
print(textes.head())

tokens = []
for i in range(len(textes)):
    tokenized_text = simple_preprocess(textes.text[i])
    tokens.append(tokenized_text)

textes['tokens'] = tokens

class_0 = 0
class_1 = 0
for i in range(len(textes['text'])):
    if len(textes['text'])<=i:
        break
    if textes.values[i][1] == 0:
        class_0 += len(textes['tokens'][i])
    if textes.values[i][1] == 1:
        class_1 += len(textes['tokens'][i])
print("class 0: ", class_0)
print("class 1: ", class_1)