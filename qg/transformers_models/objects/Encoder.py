import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset
from qg.config.config import get_logger
_logger = get_logger(logger_name=__file__)



class EncoderObject:
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
        self.return_tensors = return_tensors

        # this will be used in the question generation step
        if one_example: 
            self.encoded_example = self.tokenizer(data, 
                                        truncation=truncation, 
                                        max_length=max_length_source, 
                                        padding=padding, 
                                        return_tensors=return_tensors
                                    )
        
        # this will be used in the training process with a transformer dataset
        else:
            self.encoded_dataset = data.map(self._encode_batched_features, batched=True)
            columns = ["input_ids", "target_ids", "attention_mask"]
            self.encoded_dataset.set_format(type="torch", columns=columns)

    def _encode_batched_features(self, batched_example):
        """Encodes a huggingface dataset for a text-to-text task, using
        the method ".map()"

        Args:
            batched_example: batched dataset

        Returns:
            dictionary: encoded dataset with keys: "input_ids", "target_ids",
            "attention_mask"
        """
        
        source_encoding = self.tokenizer.batch_encode_plus(
            batched_example["context"],
            max_length = self.max_length_source,
            padding = self.padding,
            pad_to_max_length = self.pad_to_max_length,
            truncation = self.truncation, 
        )
        target_encoding = self.tokenizer.batch_encode_plus(
            batched_example["question"],
            max_length = self.max_length_target,
            padding = self.padding,
            pad_to_max_length = self.pad_to_max_length,
            truncation = self.truncation, 
        )

        encodings = {
            "input_ids": source_encoding["input_ids"], 
            "target_ids": target_encoding["input_ids"],
            "attention_mask": source_encoding["attention_mask"],
        }
        return encodings

class LearningQEncoderObject(torch.utils.data.Dataset):
    """https://huggingface.co/transformers/v4.4.2/custom_datasets.html

    Args:
        torch (_type_): _description_
    """
    def __init__(self, 
        tokenizer: object, 
        max_length_source: int, 
        text: list, 
        labels: list,
        ):

        encodings = tokenizer(
            text,
            max_length = max_length_source,
            truncation = True,
            padding = True,
        )

        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['target_ids'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class QuestionsClassificationEncoder(torch.utils.data.Dataset):
    """https://huggingface.co/transformers/v4.4.2/custom_datasets.html

    Args:
        torch (_type_): _description_
    """
    def __init__(self, 
        tokenizer: object,
        max_length_source: int,
        questions: list,
        ):

        encodings = tokenizer(
            questions,
            max_length = max_length_source,
            truncation = True,
            padding = True,
        )
        self.questions = questions
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.questions)