
import imp
from sklearn.base import BaseEstimator, TransformerMixin
from datasets import load_dataset
from qg.transformers_models.objects.LearningQDataExtractor import LearningQDataExtractorObject
from qg.config.config import get_logger
_logger = get_logger(logger_name=__file__)

class DatasetLoader(BaseEstimator, TransformerMixin):
    
    def __init__(self, dataset, split):
        self.dataset = dataset
        self.split = split

    def fit(self, X, y=None):
        return self

    def transform(self, X: dict) -> dict:

        if self.dataset == "LearningQ":
            extractor = LearningQDataExtractorObject(zipfile_name = "qg/LearningQ_data/LearningQ.zip")
            extractor.extract_data(data_path = "data/khan/khan_labeled_data", task = "classification")
            extractor.transform_classification_data()
            dataset = extractor.formatted_data
            X["dataset"] = dataset[self.split]
            _logger.info(f"{self.split} set of length {len(X['dataset']['text'])} correctly loaded to the pipeline")

        elif self.dataset == "squad_v2":
            X["dataset"] = load_dataset('squad_v2', split=self.split)
            _logger.info(f"{self.split} set of length {len(X['dataset'])} correctly loaded to the pipeline")

        else:
            raise ValueError('Select dataset="squad_v2" or "LearningQ" and split="train" or "validation"')
            
        return X