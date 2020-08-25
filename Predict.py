from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
import os

model_num = 56
version = 18
part = 4
model_name = "models/teached_d2w_v{}".format(version)


policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)
print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

model = load_model('models/model_{}/teached_model_part_{}'.format(model_num, part))
d2v_model = Doc2Vec.load(model_name)

while True:
    #Text input
    predict_text = input("Введіть текст для розпізнавання: ")

    #Text preprocess
    tokenized_text = simple_preprocess(predict_text)

    #Vector presentation
    vector = d2v_model.infer_vector(tokenized_text).tolist()
    test_text = np.asarray([vector])
    test_text = np.expand_dims(test_text, -1)

    #Text class prediction
    y_pred = model.predict(test_text)
    y_pred = [np.argmax(_) for _ in y_pred]
    if y_pred[0] == 0:
        print("Політичний напрям тексту Консерватизм")
    else:
        print("Політичний напрям тексту Лібералізм")