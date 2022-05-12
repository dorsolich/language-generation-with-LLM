
import time
import torch
import os
from tqdm import tqdm
from qg.t5_model.utils import format_time
from qg.config.config import get_logger
_logger = get_logger(logger_name=__name__)


class Encoder:
    def __init__(self, device):
        self.device = device

        # initializing trainer variables
        self.epoch_loss_values = []
        self.epoch_training_time = []

    def tokenize(self, 
            dataset, 
            tokenizer, 
            max_length_source = 512, 
            max_length_target = 32, 
            truncation = True, 
            pad_to_max_length = True,
            padding = "max_length",
        ):
        
        _logger.info("Tokenizing...")
        self.tokenizer = tokenizer
        self.max_length_source = max_length_source
        self.max_length_target = max_length_target
        self.truncation = truncation
        self.padding = padding
        self.pad_to_max_length = pad_to_max_length 

        self.dataset = dataset.map(self._encode_features, batched=True)
           
    def _encode_features(self, example_batch):
        
        source_encoding = self.tokenizer.batch_encode_plus(
            example_batch['context'],
            max_length = self.max_length_source,
            padding = self.padding,
            pad_to_max_length = self.pad_to_max_length,
            truncation = self.truncation, 
        )
        target_encoding = self.tokenizer.batch_encode_plus(
            example_batch['question'],
            max_length = self.max_length_target,
            padding = self.padding,
            pad_to_max_length = self.pad_to_max_length,
            truncation = self.truncation, 
        )

        encodings = {
            'source_ids': source_encoding['input_ids'], 
            'target_ids': target_encoding['input_ids'],
            'attention_mask': source_encoding['attention_mask'],
        }

        return encodings

    def TensorDataset(self):
        columns = ["source_ids", "target_ids", "attention_mask"]
        self.dataset.set_format(type='torch', columns=columns)
        return self


    def train_model(self, 
            model, 
            dataset,
            optimizer, 
            scheduler, 
            epochs,
            n_rows = 5
        ):
        
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        init_training_time = time.time()

        for epoch in tqdm(range(epochs)):
            _logger.info(f"Training epoch {epoch+1} / {epochs}.")
            init_epoch_time = time.time()
            self.total_loss = 0
            self.model.train()

            for i, batch in tqdm(enumerate(dataset)):
                # if i < n_rows:
                #     self._training_step(batch)
                # else:
                #     break
                # empy cache
                torch.cuda.empty_cache()
                self._training_step(batch)

            avg_epoch_loss = self.total_loss / len(dataset)
            self.epoch_loss_values.append(avg_epoch_loss)
            self.epoch_training_time.append(format_time(time.time() - init_epoch_time))
        
        self.total_training_time = format_time(time.time() - init_training_time)

        return self

    def replace_padding_token(self, labels):
        labels[labels == self.tokenizer.pad_token_id] = -100
        self.labels = labels

        return self

    def _training_step(self, batch):
        self.replace_padding_token(batch["target_ids"])

        self.model.zero_grad()
        outputs = self.model(
                        input_ids = batch["source_ids"].to(self.device), 
                        attention_mask = batch["attention_mask"].to(self.device), 
                        labels = self.labels.to(self.device)
                    )
        loss = outputs[0]
        self.batch_loss = loss.item()
        self.total_loss += self.batch_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

        return self

    def save_model(self, model_name, dir):
        PATH = os.path.join(dir, model_name)
        torch.save(self.model, PATH)
        _logger.info(f"model saved in path {PATH}")

        return self