from sklearn.base import BaseEstimator, TransformerMixin
from qg.transformers_models.objects.Preprocessor import DataProcessorObject
from qg.config.config import get_logger
_logger = get_logger(logger_name=__file__)
import os

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


        if self.preprocess_setting != "AQPL":
            _logger.info(f""" 
            i = 130304
            {processed_train_data[130304]["context"]}
            {processed_train_data[130304]["question"]}
            i = 130308
            {processed_train_data[130308]["context"]}
            {processed_train_data[130308]["question"]}
            """)
        else:
            try:
                _logger.info(f"""
                i = -2
                {processed_train_data[-2]["context"]}
                {processed_train_data[-2]["question"]}
                """)

            except IndexError as e:
                _logger.error("""Failed to preprocess AQPL. 
                This happens because it is at least the second time you run the AQPL
                setting in this environment. The first time you did it, the dataset was saved in memory cache. 
                Next times you run it, it loads from cache and fails to generate a temporal variable that needs to process the data.
                Please create a new environment and run the command again, or clean the cache...""")
                os._exit(0)

        X["dataset"] = processed_train_data
        return X