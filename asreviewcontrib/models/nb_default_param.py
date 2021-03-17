from sklearn.naive_bayes import MultinomialNB

from asreview.models.classifiers.base import BaseTrainClassifier

class NaiveBayesDefaultParamsModel(BaseTrainClassifier):
    """Naive Bayes classifier

    The Naive Bayes classifier with the default SKLearn parameters.
    """

    name = "nb_example"

    def __init__(self):

        super(NaiveBayesDefaultParamsModel, self).__init__()
        self._model = MultinomialNB()

from asreview.models.feature_extraction.base import BaseFeatureExtraction

class NaiveBayesFeatureExtration(BaseFeatureExtraction):
    name = "nb_feat"

    def __init__(self):

        super(NaiveBayesFeatureExtration, self).__init__()

    @abstractmethod
    def transform(self, texts):
        return texts
    
    def full_hyper_space(self):
        from hyperopt import hp
        hyper_choices = {}
        hyper_space = {}
        return hyper_space, hyper_choices
