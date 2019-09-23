"""
data_preprocessor is a module used for loading , cleaning , preprocessing and augmenting text data set
"""
import re
import string

import nlpaug.augmenter.word as naw
import pandas as pd
from nltk.stem import WordNetLemmatizer


class DataPreprocessor:
    """
    class has methods to clean , augment and
    create a text representation as input to machine learning model
    """

    stemmer = WordNetLemmatizer()

    def __init__(self):
        self.raw_data_path = './'
        self.processed_data_path = './'
        self.df = None
        self.augmented_df = pd.DataFrame(columns=['text', 'label'])
        self.no_of_augmentations = 5
        self.augmentor = naw.WordNetAug(aug_p=0.5)

    def load_data(self, path):
        """
        data set loader
        :param path: path to dataset
        :return: pandas dataframe
        """
        self.raw_data_path = path
        self.df = pd.read_csv(self.raw_data_path,
                              delimiter='\t',
                              header=None).rename(columns={0: 'text', 1: 'label'})

    def save_proceesed_data(self, df, path):
        """
        save processed dataset
        :param path: where to save clean data
        :return:
        """
        self.processed_data_path = path
        # self.augmented_df=self.augmented_df.dropna()
        # print(self.augmented_df.isnull().any())
        df.to_csv(self.processed_data_path)

    def clean_text(self):
        """
        function to clean text from punctuation , special chars , extra spaces and numbers
        :return:
        """
        # 1. lower case all characters
        self.df['text'] = self.df['text'].str.lower()
        # 2. numbers may distract the model , we shall remove it
        self.df['text'] = self.df['text'].apply(lambda x: re.sub(r'\d+', '', x))
        # 3. remove punctation and special characters
        self.df['text'] = self.df['text'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
        # 4. remove double or many tabs
        self.df['text'] = self.df['text'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())

    def augment_text(self):
        """
        apply random replacements for words with other that have same meaning
        :return:
        """
        for i in range(len(self.df)):
            for j in range(self.no_of_augmentations):
                text = self.augmentor.augment(self.df['text'].iloc[i])
                label = int(self.df['label'].iloc[i])
                tempdf = pd.DataFrame(list(zip([text], [label])), columns=['text', 'label'])
                self.augmented_df = self.augmented_df.append(tempdf)

    @staticmethod
    def lemmatize(sentence):
        """
        revert the words back to its roots
        :return: lemmatized string
        """
        document = sentence.split()
        document = [DataPreprocessor.stemmer.lemmatize(word) for word in document]
        document = ' '.join(document)
        return document

    @staticmethod
    def stem_text(df):
        """
        stem all augmented text
        :return:
        """
        df['text'] = df['text'].apply(lambda x: DataPreprocessor.lemmatize(x))
