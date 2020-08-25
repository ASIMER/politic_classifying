from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import pandas as pd
import numpy as np
import os
from datetime import datetime

model_num = 58
version = 18


policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)
print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

data = pd.read_json("data/teach_data_v{}.json".format(version), orient="table")
data = data.sample(frac=1)
data_tokens, data_classes = data.tokens.to_list(), data['class'].to_list()

model_name = "models/teached_d2w_v{}".format(version)


# split samples to pieces
def get_test_data(data_tokens, data_classes, pieces, part):
    chunk_size = len(data_tokens) / pieces
    start = int(chunk_size * part);

    #test part index selection
    if part == pieces:
        end = len(data_tokens)
    else:
        end = int(start + chunk_size)
    print(start, end)

    train_text = data_tokens[:start][:] + data_tokens[end:][:]
    train_labels = data_classes[:start][:] + data_classes[end:][:]
    test_text = data_tokens[start:end][:]
    test_labels = data_classes[start:end][:]

    train_text = np.asarray(train_text)
    train_labels = np.asarray(train_labels)
    test_text = np.asarray(test_text)
    test_labels = np.asarray(test_labels)

    train_text = train_text.astype('float32')
    test_text = test_text.astype('float32')

    train_text = np.expand_dims(train_text, -1)
    train_labels = np.expand_dims(train_labels, -1)
    train_labels = np.hstack([train_labels, 1-train_labels])

    test_text = np.expand_dims(test_text, -1)
    test_labels = np.expand_dims(test_labels, -1)
    test_labels = np.hstack([test_labels, 1-test_labels])

    return (train_text, train_labels, test_text, test_labels)


#model = load_model('model/teached_model_lstm')

# 5-fold cross validation.
total_acc = 0.0
total_pre = 0.0
total_rec = 0.0
accuracy_part = []
precision_part = []
recall_part = []
time_part = []
for part in range(0, 5):
    start_time = datetime.now()
    x_train, y_train, x_test, y_test = get_test_data(data_tokens, data_classes, 5, part)

    #model learning part
    model = Sequential()
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(LSTM(1024))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print('Fitting:')
    model.fit(x_train, y_train, batch_size=128, epochs=50)

    print('Fitting done on part', part)

    model.save('models/model_{}/teached_model_part_{}'.format(model_num, part))


    y_pred = model.predict(x_test)
    y_pred = [np.argmax(_) for _ in y_pred]
    y_true = [np.argmax(_) for _ in y_test]


    class_count = y_true.count(0) #total class count
    class_pred_count = y_pred.count(0) #total class predicts
    class_pred = [y_pred[i] == y_true[i] for i in range(len(y_true))] #class predicted true\false
    class_pred_true = 0
    for i in range(len(class_pred)):
        if class_pred[i] == True and y_true[i] == 0:
            class_pred_true += 1

    precision = class_pred_true/class_pred_count
    total_pre += precision
    precision_part.append(precision)

    recall = class_pred_true/class_count
    total_rec += recall
    recall_part.append(recall)

    accuracy = np.average([y_pred[i] == y_true[i] for i in range(len(y_pred))])
    accuracy_part.append(accuracy)
    total_acc += accuracy

    time_work = datetime.now() - start_time
    time_part.append(time_work)

    print("accuracy on part",part, accuracy)
    print("precision on part",part, precision)
    print("recall on part",part, recall)
    print("time on part",part, time_work)

print("\n\nTotal accuracy: ", total_acc/5)
print("Accuracy per part: ", accuracy_part)

print("Total precision: ", total_pre/5)
print("Precision per part: ", precision_part)

print("Total recall: ", total_rec/5)
print("Recall per part: ", recall_part)

print("Time per part: ", time_part)