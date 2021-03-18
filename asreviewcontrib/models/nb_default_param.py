from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from os import environ

from asreview.models.classifiers.base import BaseTrainClassifier

class NaiveBayesDefaultParamsModel(BaseTrainClassifier):
    name = "nb_example"

    def __init__(self):

        super(NaiveBayesDefaultParamsModel, self).__init__()
        self._model = MultinomialNB()
        self._vectorizer = TfidfVectorizer()
    
    def fit(self, X, y):
        watson_data = []
        for i, text in enumerate(X):
            watson_data.append({'text': text, 'class': y[i]})
        pd.DataFrame(watson_data).to_csv('watson_data.csv', index=False, header=False)
        
        #for text in X:
        #    print(text)
        #for class_label in y:
        #    print(class_label)
        X = self._vectorizer.fit_transform(X)
        return self._model.fit(X, y)
    
    def predict_proba(self, X):
        X = self._vectorizer.transform(X)
        return self._model.predict_proba(X)

from asreview.models.feature_extraction.base import BaseFeatureExtraction

class NaiveBayesFeatureExtration(BaseFeatureExtraction):
    name = "nb_feat"

    def __init__(self, split_ta=0, use_keywords=0):

        super(NaiveBayesFeatureExtration, self).__init__()
        self._vectorizer = TfidfVectorizer()

    def fit(self, texts):
        #print(texts)
        #self._vectorizer.fit(texts)
        pass
        
    def transform(self, texts):
        return texts #self._vectorizer.transform(texts)
    
    def full_hyper_space(self):
        return {}, {}
