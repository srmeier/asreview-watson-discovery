from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from os import environ

from asreview.models.classifiers.base import BaseTrainClassifier

import json
from time import sleep
from ibm_watson import NaturalLanguageClassifierV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

class NaiveBayesDefaultParamsModel(BaseTrainClassifier):
    name = "nb_example"

    def __init__(self):

        super(NaiveBayesDefaultParamsModel, self).__init__()
        self._model = MultinomialNB()
        self._vectorizer = TfidfVectorizer()
        self._apikey = environ['WATSON_API_KEY']
        self._url = environ['WATSON_URL']
        self._classifier_id = None
    
    def fit(self, X, y):
        watson_data = []
        for i, text in enumerate(X):
            watson_data.append({'text': text.replace('\n', '')[0:1024], 'class': y[i]})
        pd.DataFrame(watson_data).to_csv('watson_data.csv', index=False, header=False)
        
        authenticator = IAMAuthenticator(f'{self._apikey}')
        natural_language_classifier = NaturalLanguageClassifierV1(authenticator=authenticator)
        natural_language_classifier.set_service_url(f'{self._url}')

        with open('watson_data.csv', 'rb') as training_data:
            classifier = natural_language_classifier.create_classifier(training_data=training_data, training_metadata='{"name": "TutorialClassifier", "language": "en"}').get_result()
        
        self._classifier_id = classifier['classifier_id']
        
        while classifier['status'] == 'Training':
            classifier = natural_language_classifier.get_classifier(self._classifier_id).get_result()
            sleep(1)
        
        #for text in X:
        #    print(text)
        #for class_label in y:
        #    print(class_label)
        #X = self._vectorizer.fit_transform(X)
        #return self._model.fit(X, y)
    
    def predict_proba(self, X):
        authenticator = IAMAuthenticator(f'{self._apikey}')
        natural_language_classifier = NaturalLanguageClassifierV1(authenticator=authenticator)
        natural_language_classifier.set_service_url(f'{self._url}')

        text_collection = []
        for text in X:
            text_collection.append({'text': text.replace('\n', '')[0:1024]})

        y = []
        chunk_size = 10
        for chunk_text in [text_collection[i:i + chunk_size] for i in range(0, len(text_collection), chunk_size)]:
            classes = natural_language_classifier.classify_collection(self._classifier_id, chunk_text).get_result()
            for record in classes['collection']:
                for class_label in record['classes']:
                    confidence = [0., 0.]
                    if int(class_label['class_name']) == 1:
                        confidence[1] = class_label['confidence']
                    elif int(class_label['class_name']) == 0:
                        confidence[0] = class_label['confidence']
                    y.append(confidence)
        
        return y
        
        #X = self._vectorizer.transform(X)
        #return self._model.predict_proba(X)

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
