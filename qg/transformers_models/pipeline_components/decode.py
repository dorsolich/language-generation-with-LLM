from sklearn.base import BaseEstimator, TransformerMixin
from qg.transformers_models.objects.Encoder import EncoderObject
from qg.transformers_models.objects.Decoder import DecoderObject
from tqdm import tqdm

class Decoder(BaseEstimator, TransformerMixin):

    def __init__(self, 
        device, 
        context_max_length, 
        question_max_length,
        num_beams,
        test,
        ):
        self.device = device
        self.context_max_length = context_max_length
        self.question_max_length = question_max_length
        self.num_beams = num_beams
        self.test = test


    def fit(self, X, y=None):
        return self

    def transform(self, X: dict) -> dict:
        encoder = EncoderObject(device=self.device)
        decoder = DecoderObject(device=self.device)

        for i, example in tqdm(enumerate(X["dataset"])):
            if decoder.decode_example(example=example):
                encoder.encode(
                                tokenizer = X["tokenizer"],
                                data = example["context"],
                                one_example = True,
                                max_length_source = self.context_max_length,
                                truncation = True, 
                                padding = "max_length",
                                return_tensors = "pt"
                                )
                decoder.decode(
                                model = X["model"], 
                                tokenizer = X["tokenizer"],
                                encodings = encoder.encoded_example,
                                num_beams = self.num_beams,  
                                question_max_length = self.question_max_length, 
                                repetition_penalty = 2.5,
                                length_penalty = 1,
                                early_stopping = True,
                                use_cache = True,
                                num_return_sequences = 1,
                                do_sample = False
                                )
            if self.test and i!=0:
                break
        
        X["source_texts"] = decoder.source_texts
        X["target_texts"] = decoder.target_texts
        X["model_outputs"] = decoder.model_outputs
        return X
