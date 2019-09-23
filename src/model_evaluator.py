import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras.preprocessing import sequence
from sklearn.metrics import classification_report, confusion_matrix


class ModelEvaluator:
    def __init__(self, algorithm):
        self.algorithm = algorithm
        if self.algorithm == 'lstm':
            self.xtest = np.load('data/processed/x_test' + self.algorithm + '.npy',allow_pickle=True)
        elif self.algorithm == 'random_forest':
            self.xtest = np.loadtxt('data/processed/x_test' + self.algorithm + '.txt',
                                    delimiter='/t', dtype='str')
        self.ytest = np.loadtxt('data/processed/y_test' + self.algorithm + '.txt',dtype=np.int)
        self.vectorizer = None
        self.model = None
        self.predictions = None
        self.new_data_path = './'
        self.X_new_text = None
        self.max_review_length = 500

    def load_weights(self):
        if self.algorithm == 'random_forest':
            self.vectorizer = joblib.load('model_weights/vectorizer_train.sav')
            self.model = joblib.load('model_weights/' + self.algorithm + '.sav')
        elif self.algorithm == 'lstm':
            self.model = joblib.load('model_weights/' + self.algorithm + '.sav')

    def print_classification_report(self):
        if self.algorithm == 'random_forest':
            self.xtest = self.vectorizer.transform(self.xtest)
        elif self.algorithm == 'lstm':
            self.xtest = sequence.pad_sequences(self.xtest, maxlen=self.max_review_length)
        self.predictions = np.round(self.model.predict(self.xtest)).astype(np.int)
        print("Classification Report {} ".format(classification_report(self.ytest, self.predictions)))

    def plot_confusion_matrix(self, no_of_classes):

        # confusion matrix
        df_cm = pd.DataFrame(confusion_matrix(self.ytest, self.predictions), range(no_of_classes),
                             range(no_of_classes))
        sns.set(font_scale=1.4)  # for label size
        plt.subplots(figsize=(7, 5))
        sns.heatmap(df_cm, annot=True)
