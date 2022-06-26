from sklearn.base import BaseEstimator, TransformerMixin
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, RandomSampler
from qg.transformers_models.objects.Encoder import EncoderObject, LearningQEncoderObject, QuestionsClassificationEncoder

from qg.config.config import get_logger
_logger = get_logger(logger_name=__file__)

class PreTrainedTokenizerDownloader(BaseEstimator, TransformerMixin):

    def __init__(self, model):
        self.model = model

    def fit(self, X, y=None):
        return self

    def transform(self, X: dict) -> dict:
        tokenizer = AutoTokenizer.from_pretrained(self.model)
        X["tokenizer"] = tokenizer
        return X


class Encoder(BaseEstimator, TransformerMixin):

    def __init__(self, 
                device, 
                max_length_source, 
                batch_size,
                max_length_target = None,
                dataset = "squad_v2"
                ):
        self.device = device
        self.max_length_source = max_length_source
        self.max_length_target = max_length_target
        self.batch_size = batch_size
        self.dataset = dataset

    def fit(self, X, y=None):
        return self

    def transform(self, X: dict) -> dict:

        # Question Generation task...
        if self.dataset == "squad_v2":
            
            encoder = EncoderObject(device=self.device)
            encoder.encode(
                data = X["dataset"],
                tokenizer = X["tokenizer"],
                max_length_source = self.max_length_source,
                max_length_target = self.max_length_target,
                truncation = True,
                padding = "max_length",
            )
            encoded_dataset = encoder.encoded_dataset

        # Question Classification task...
        elif self.dataset == "LearningQ":

            text = X["dataset"]["text"]
            labels = X["dataset"]["labels"]

            dataset = LearningQEncoderObject(
                tokenizer = X["tokenizer"],
                max_length_source = self.max_length_source, 
                text = text, 
                labels = labels,
            )
            encoded_dataset = dataset

        # Classify generated questions tasks...
        elif self.dataset == "generated_questions":

            questions = X["dataset"]

            dataset = QuestionsClassificationEncoder(
                tokenizer = X["tokenizer"],
                max_length_source = self.max_length_source, 
                questions = questions
            )
            encoded_dataset = dataset


        X["encoded_dataset"] = encoded_dataset
        return X


class DataLoaderComponent(BaseEstimator, TransformerMixin):

    def __init__(self, batch_size, task):
        self.batch_size = batch_size
        self.task = task

    def fit(self, X, y=None):
        return self

    def transform(self, X: dict) -> dict:

        if self.task == "SequenceClassification":
            train_dataloader = DataLoader(X["encoded_dataset"], 
                                        batch_size=self.batch_size, 
                                        shuffle=True
                                        )
                                        
        elif self.task == "QuestionGeneration":
            train_sampler = RandomSampler(X["encoded_dataset"])
            train_dataloader = DataLoader(X["encoded_dataset"], 
                                        sampler=train_sampler, 
                                        batch_size=self.batch_size
                                        )
        X["data_loader"] = train_dataloader
        return X