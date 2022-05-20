from sklearn.base import BaseEstimator, TransformerMixin
from qg.transformers_models.objects.Validator import ValidatorObject
from qg.config.config import get_logger
_logger = get_logger(logger_name=__file__)
class ModelValidator(BaseEstimator, TransformerMixin):
    
    def __init__(self, device, n_epochs, test, metric, task, max_length_target = 100):
        self.device = device
        self.n_epochs = n_epochs
        self.test = test
        self.metric = metric
        self.task = task
        self.max_length_target = max_length_target

    def fit(self, X, y=None):
        return self

    def transform(self, X: dict) -> dict:
        _logger.info(f"Validating model with metric {self.metric}...")
        validator = ValidatorObject(self.device)
        
        if self.task == "SequenceClassification":
            validator.evaluate_cls_model( 
                data_loader = X["data_loader"], 
                model = X["model"],
                epochs = self.n_epochs,
                test = self.test,
                metric = self.metric
            )
        elif self.task == "QuestionGeneration":
            validator.evaluate_qg_model(
                data_loader = X["data_loader"],
                epochs = self.n_epochs,
                model = X["model"],
                test = self.test,
                metric = self.metric,
                max_length_target = self.max_length_target,
                tokenizer = X["tokenizer"],
            )

        
        for key in validator.__dict__:
            X[key] = validator.__dict__[key]
        return X