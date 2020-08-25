from tensorflow.keras.models import load_model
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import numpy as np
from gensim.models.doc2vec import Doc2Vec
from gensim.utils import simple_preprocess
import os
from PyQt5 import QtWidgets, QtCore
from predictdesign import Ui_MainWindow  # импорт нашего сгенерированного файла
import sys

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


class mywindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(mywindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        # подключение клик-сигнал к слоту btnClicked
        self.ui.PredButton.clicked.connect(self.btnClicked)

    def btnClicked(self):
        translate = QtCore.QCoreApplication.translate

        predict_text = self.ui.textEdit.toPlainText()
        print(predict_text)
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
            predicted_text = "Політична спрямованість тексту: Консерватизм"
            # Меняем текст
            self.ui.PredText.setText(translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:14pt;\">{}</span></p></body></html>".format(predicted_text)))
        else:
            predicted_text = "Політична спрямованість тексту: Лібералізм"
            # Меняем текст
            self.ui.PredText.setText(translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:14pt;\">{}</span></p></body></html>".format(predicted_text)))




app = QtWidgets.QApplication([])
application = mywindow()
application.show()

sys.exit(app.exec())