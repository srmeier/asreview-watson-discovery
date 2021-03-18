from sklearn.naive_bayes import MultinomialNB

from asreview.models.classifiers.base import BaseTrainClassifier

class NaiveBayesDefaultParamsModel(BaseTrainClassifier):
    name = "nb_example"

    def __init__(self):

        super(NaiveBayesDefaultParamsModel, self).__init__()
        self._model = MultinomialNB()
    
    def fit(self, X, y):
        print(X)
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(X)
        return self._model.fit(X, y)
    
    def predict_proba(self, X):
        return self._model.predict_proba(X)

from asreview.models.feature_extraction.base import BaseFeatureExtraction

class NaiveBayesFeatureExtration(BaseFeatureExtraction):
    name = "nb_feat"

    def __init__(self):

        super(NaiveBayesFeatureExtration, self).__init__()

    @abstractmethod
    def transform(self, texts):
        return texts
    
    def full_hyper_space(self):
        return {}, {}
