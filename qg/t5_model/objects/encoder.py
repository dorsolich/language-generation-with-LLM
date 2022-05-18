import time
import torch
import os
from tqdm import tqdm
from qg.utils import format_time

from qg.config.config import get_logger
_logger = get_logger(logger_name=__name__)

class Encoder:
    def __init__(self, device):
        self.device = device

    def encode(self,
            tokenizer, 
            data,
            one_example = False,
            max_length_source = 512, 
            max_length_target = 32, 
            truncation = True, 
            pad_to_max_length = True,
            padding = "max_length",
            return_tensors = "pt"
        ):
        
        
        self.tokenizer = tokenizer
        self.max_length_source = max_length_source
        self.max_length_target = max_length_target
        self.truncation = truncation
        self.padding = padding
        self.pad_to_max_length = pad_to_max_length 

        if one_example: # this will be used in the generation step
            self.encoded_example = self.tokenizer(data, truncation=truncation, max_length=max_length_source, padding=padding, return_tensors=return_tensors)
        else: # this will be used in he training process
            _logger.info("Tokenizing...")
            self.encoded_dataset = data.map(self._encode_features, batched=True)
           
    def _encode_features(self, example_batch):
        
        source_encoding = self.tokenizer.batch_encode_plus(
            example_batch["context"],
            max_length = self.max_length_source,
            padding = self.padding,
            pad_to_max_length = self.pad_to_max_length,
            truncation = self.truncation, 
        )
        target_encoding = self.tokenizer.batch_encode_plus(
            example_batch["question"],
            max_length = self.max_length_target,
            padding = self.padding,
            pad_to_max_length = self.pad_to_max_length,
            truncation = self.truncation, 
        )

        encodings = {
            "source_ids": source_encoding["input_ids"], 
            "target_ids": target_encoding["input_ids"],
            "attention_mask": source_encoding["attention_mask"],
        }

        return encodings

    def TensorDataset(self):
        columns = ["source_ids", "target_ids", "attention_mask"]
        self.encoded_dataset.set_format(type="torch", columns=columns)
        return self