from sklearn.base import BaseEstimator, TransformerMixin
from qg.transformers_models.objects.Preprocessor import DataProcessorObject
from qg.config.config import get_logger
_logger = get_logger(logger_name=__file__)

class DataProcessor(BaseEstimator, TransformerMixin):
    
    def __init__(self, preprocess_setting):
        self.preprocess_setting = preprocess_setting

    def fit(self, X, y=None):
        return self

    def transform(self, X: dict) -> dict:
        _logger.info(f"Processing dataset on setting {self.preprocess_setting}...")

        processor = DataProcessorObject(
                        sep_token = " <hl> ",
                        eos_token = " </s> ",
                        setting = self.preprocess_setting
                        )
        processed_train_data = processor.process(X["dataset"])
        processed_train_data = processor.filter_examples(processed_train_data)

        print(processed_train_data[0]["context"])
        print(processed_train_data[0]["question"])

        X["dataset"] = processed_train_data
        return X