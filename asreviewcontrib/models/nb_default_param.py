from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

from asreview.models.classifiers.base import BaseTrainClassifier

class NaiveBayesDefaultParamsModel(BaseTrainClassifier):
    name = "nb_example"

    def __init__(self):

        super(NaiveBayesDefaultParamsModel, self).__init__()
        self._model = MultinomialNB()
        self._vectorizer = TfidfVectorizer()
    
    def fit(self, X, y):
        print(X)
        X = self._vectorizer.fit_transform(X)
        print(X)
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
        print(texts)
        self._vectorizer.fit(texts)
        
    def transform(self, texts):
        return self._vectorizer.transform(texts)
    
    def full_hyper_space(self):
        return {}, {}
