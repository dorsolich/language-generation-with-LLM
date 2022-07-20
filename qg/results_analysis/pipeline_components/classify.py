from sklearn.base import BaseEstimator, TransformerMixin
from qg.results_analysis.objects.Classifier import ClassifierObject
from qg.config.config import get_logger
_logger = get_logger(logger_name=__file__)

class QuestionsClassifier(BaseEstimator, TransformerMixin):
    
    def __init__(self, device, test):
        self.device = device
        self.test = test

    def fit(self, X, y=None):
        return self

    def transform(self, X: dict) -> dict:
        _logger.info(f"Classifying questions...")

        classifier = ClassifierObject(self.device)
        
        classifier.classify( 
            data_loader = X["data_loader"], 
            model = X["model"],
            test = self.test,
        )

        
        for key in classifier.__dict__:
            X[key] = classifier.__dict__[key]
        return X