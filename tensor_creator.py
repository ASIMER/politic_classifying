import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec
from gensim.utils import simple_preprocess

version = 18

model_name = "models/teached_d2w_v{}".format(version)

textes = pd.read_csv("data/cleaned_textes_v{}.csv".format(version))
model = Doc2Vec.load(model_name)
textes = textes.truncate(after=1919)
print(textes['class'].describe())
print(textes.head())
print("i=2: ", textes.text[2])
tokenized_text = simple_preprocess(textes.text[2])
print("i=2 tokenized_text: ", tokenized_text)
vector = model.infer_vector(tokenized_text).tolist()
print(len(vector))

tokens = []
for i in range(len(textes)):
    tokenized_text = simple_preprocess(textes.text[i])
    vector = model.infer_vector(tokenized_text).tolist()
    tokens.append(vector)


textes['tokens'] = tokens

print(textes.head())

print("tokens: ", textes.tokens[2])
test_text = simple_preprocess(textes.text[2])

textes.to_json("data/teach_data_v{}.json".format(version), orient="table")