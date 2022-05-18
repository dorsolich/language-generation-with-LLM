
import time
import torch
import os
from tqdm import tqdm
from qg.t5_model.utils import format_time
from qg.config.config import get_logger
_logger = get_logger(logger_name=__name__)


class Trainer:
    def __init__(self, device):
        self.device = device

        # initializing trainer variables
        self.epoch_loss_values = []
        self.epoch_training_time = []

    def train_model(self, 
            model, 
            tokenizer,
            data_loader,
            optimizer, 
            scheduler, 
            epochs,
            test = False,
        ):
        
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        init_training_time = time.time()

        for epoch in tqdm(range(epochs)):
            _logger.info(f"Training epoch {epoch+1} / {epochs}. Initializing training loop of batch: {len(data_loader)}")
            init_epoch_time = time.time()
            self.total_loss = 0
            self.model.train()

            for i, batch in tqdm(enumerate(data_loader)):
                # empy cache
                torch.cuda.empty_cache()
                if test:
                    if i < 5:
                        self._training_step(batch)
                    else:
                        break
                else:
                    self._training_step(batch)

            avg_epoch_loss = self.total_loss / len(data_loader)
            self.epoch_loss_values.append(avg_epoch_loss)
            self.epoch_training_time.append(format_time(time.time() - init_epoch_time))
        
        self.total_training_time = format_time(time.time() - init_training_time)

        return self

    def _replace_padding_token(self, labels):
        labels[labels == self.tokenizer.pad_token_id] = -100
        self.labels = labels

        return self

    def _training_step(self, batch):
        self._replace_padding_token(batch["target_ids"])

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