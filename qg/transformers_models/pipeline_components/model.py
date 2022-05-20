from sklearn.base import BaseEstimator, TransformerMixin
from transformers import AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoConfig,BertForSequenceClassification
import torch
import os


class PreTrainedModelDownloader(BaseEstimator, TransformerMixin):

    def __init__(self, task, model):
        self.task = task
        self.model = model

    def fit(self, X, y=None):
        return self

    def transform(self, X: dict) -> dict:

        if "SequenceClassification" == self.task:
            # config = AutoConfig.from_pretrained(self.model)
            model = AutoModelForSequenceClassification.from_pretrained(self.model)

        elif "QuestionGeneration" == self.task:
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model)

        else:
            raise ValueError('Select a task: "SequenceClassification", "Seq2Seq"')
        
        X["model"] = model
        return X



class TrainedModelUploader(BaseEstimator, TransformerMixin):

    def __init__(self, model, model_name, model_dir, device):
        self.model = model
        self.model_name = model_name
        self.model_dir = model_dir
        self.device = device

    def fit(self, X, y=None):
        return self

    def transform(self, X: dict) -> dict:

        file_name = f'{self.model_name}.pt'
        PATH = os.path.join(self.model_dir, file_name)
        model = torch.load(PATH, map_location=self.device)
        
        X["model"] = model
        return X