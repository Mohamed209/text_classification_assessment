"""
module to train and tune machine learning model for text classification
"""

import joblib
import numpy as np
import pandas as pd
from keras.layers import Dense, LSTM, Embedding
from keras.models import Sequential
from keras.preprocessing import sequence
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted


class ModelTrainingSession:
    """
    class to train , tune and save best model
    """

    def __init__(self):
        self.processed_data_path = './'
        self.df = None
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=3000)
        self.X = []
        self.y = []
        self.classifier = None
        self.parameters = {}
        self.grid_search = None
        self.tfidfstring = 'tfidf'  # label for pipeline
        self.randomforestring = 'rf'  # label for pipeline
        self.pipeline = None
        self.bestmodel = None
        self.bestscore = 0
        self.bestparams = {}
        self.algorithm = ''
        self.max_review_length = 500
        self.top_words = 5000
        self.embedding_vector_length = 32

    def load_data(self, processedpath):
        if processedpath == 'imdb':
            from keras.datasets import imdb
            (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=self.top_words)
            self.X = np.concatenate((X_train, X_test), axis=0)
            self.y = np.concatenate((y_train, y_test), axis=0)

        else:
            self.processed_data_path = processedpath
            self.df = pd.read_csv(self.processed_data_path)
            self.df = self.df.dropna()
            self.X = self.df['text'].values
            self.y = self.df['label'].values

    def represent_text_BOW(self,
                           ngram=(1, 2),
                           model='tf-idf',
                           max_features=None,
                           mindf=5,
                           maxdf=0.7):
        """
        classical bag of words representations for text
        :param ngram: N-gram tokens to be considered
        :param model: vectorization technique (TF-IDF , Count , ...)
        :return:
        """
        if model == 'tf-idf':
            self.vectorizer = TfidfVectorizer(
                ngram_range=ngram,
                max_features=max_features,
                min_df=mindf,
                max_df=maxdf,
                stop_words=stopwords.words('english'))
            self.X = self.vectorizer.fit_transform(self.df['text'].values).toarray()
            self.y = self.df['label'].values
        else:
            # TODO : support other vectorization methods
            print("only tf-idf is supported")

    @staticmethod
    def train_test_split(features, labels, test_size):
        X_train, X_test, y_train, y_test = train_test_split(features,
                                                            labels,
                                                            test_size=test_size,
                                                            random_state=0)
        return X_train, X_test, y_train, y_test

    def grid_search_pipeline(self, parameters,
                             algorithm='random_forest',
                             text_represent='tf-idf'):
        self.algorithm = algorithm
        if self.algorithm == 'random_forest' and text_represent == 'tf-idf':
            self.classifier = RandomForestClassifier()
            self.pipeline = Pipeline(steps=[(self.tfidfstring, self.vectorizer),
                                            (self.randomforestring, self.classifier)],
                                     verbose=True)
            self.parameters = parameters
            self.grid_search = GridSearchCV(self.pipeline, self.parameters, n_jobs=-1)
            # joblib.dump(self.vectorizer, 'model_weights/vectorizer.sav')
        # Todo add more algorithms

    def start_search(self, xtrain, ytrain, xtest, ytest):
        """
        train machine learning algorithm to classify text
        :param classifier_algorithm:
        :return:
        """
        self.grid_search.fit(xtrain, ytrain)
        # joblib.dump(self.vectorizer, 'model_weights/vectorizer.sav')
        self.bestmodel = self.grid_search.best_estimator_
        self.bestparams = self.grid_search.best_params_
        self.bestscore = self.grid_search.best_score_
        preds = self.bestmodel.predict(xtest)
        print("Finished Training with test set Accuracy : {}".format(accuracy_score(ytest, preds)))
        print("Best params : {} ".format(self.bestparams))

    def train_best_params(self, xtrain, ytrain, xtest, ytest, algorithm):
        self.algorithm = algorithm
        if self.algorithm == 'random_forest':
            xtrain = self.vectorizer.fit_transform(xtrain).toarray()
            xtest = self.vectorizer.transform(xtest).toarray()
            self.bestmodel = RandomForestClassifier(n_estimators=200,
                                                    n_jobs=-1,
                                                    verbose=True)
            self.bestmodel.fit(xtrain, ytrain)
            preds = self.bestmodel.predict(xtest)
            print("Finished Training with test set Accuracy : {}".format(accuracy_score(ytest, preds)))
        elif self.algorithm == 'lstm':
            xtrain = sequence.pad_sequences(xtrain, maxlen=self.max_review_length)
            xtest = sequence.pad_sequences(xtest, maxlen=self.max_review_length)
            self.bestmodel = Sequential()
            self.bestmodel.add(Embedding(self.top_words,
                                         self.embedding_vector_length,
                                         input_length=self.max_review_length))
            self.bestmodel.add(LSTM(100))
            self.bestmodel.add(Dense(1, activation='sigmoid'))
            self.bestmodel.compile(loss='binary_crossentropy',
                                   optimizer='adam',
                                   metrics=['accuracy'])
            print(self.bestmodel.summary())
            self.bestmodel.fit(xtrain,
                               ytrain,
                               validation_data=(xtest, ytest),
                               epochs=3,
                               batch_size=64)
            print("Finished Training with test set Accuracy : {}".format(self.bestmodel.evaluate(xtest,
                                                                                                 ytest)[1]))
        # model .evaluate

    def save_training_data(self, xtest, ytest):
        """
        function used to save training data like xtest , ytest for model evaluation ,
        models weights and vectorizer objects
        :param xtest:
        :param ytest:
        :return:
        """
        # save x_test and y_test for model evaluation
        if self.algorithm == 'lstm':
            np.save('data/processed/x_test' + self.algorithm + '.npy', xtest)
        elif self.algorithm == 'random_forest':
            np.savetxt('data/processed/x_test' + self.algorithm + '.txt', xtest, fmt='%s')
        np.savetxt('data/processed/y_test' + self.algorithm + '.txt', ytest, fmt='%d')
        joblib.dump(self.bestmodel, 'model_weights/' + self.algorithm + '.sav', compress=1)
        if check_is_fitted(self.vectorizer, '_tfidf'):
            joblib.dump(self.vectorizer, 'model_weights/vectorizer_train.sav', compress=1)
