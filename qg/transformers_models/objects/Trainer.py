
from cProfile import label
import time
import torch
from torch import nn
import os
from tqdm import tqdm
from qg.config.config import get_logger
_logger = get_logger(logger_name=__file__)

import datetime

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


class TrainerObject:
    def __init__(self, device):
        self.device = device

        # initializing trainer variables
        self.epoch_loss_values = []
        self.epoch_training_time = []
        self.batch_loss_values = []

    def train_model(self, 
            model, 
            tokenizer,
            data_loader,
            task,
            optimizer, 
            scheduler, 
            epochs,
            test = False,
        ):
        
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.task = task
        init_training_time = time.time()

        for epoch in tqdm(range(epochs)):
            _logger.info(f"Training epoch {epoch+1} / {epochs}. Initializing training loop of batch: {len(data_loader)}")
            
            init_epoch_time = time.time()
            self.epoch_total_loss = 0
            self.model.train()

            for i, batch in tqdm(enumerate(data_loader)):
                torch.cuda.empty_cache()
                self._training_step(batch)

                if test and i>5:
                    print("Training stopped")
                    break

            avg_epoch_loss = self.epoch_total_loss / len(data_loader)
            self.epoch_loss_values.append(avg_epoch_loss)
            self.epoch_training_time.append(format_time(time.time() - init_epoch_time))
        
        self.total_training_time = format_time(time.time() - init_training_time)
        

        return self



    def _training_step(self, batch):

        if self.task == "SequenceClassification":
            
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["target_ids"].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
            # _, preds = torch.max(outputs.logits, dim=1)
            # self.correct_predictions += torch.sum(preds == labels)


        elif self.task == "QuestionGeneration":
            
            self._replace_padding_token(batch["target_ids"])
            
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            targets = self.targets.to(self.device)

            self.model.zero_grad()
            outputs = self.model(
                            input_ids = input_ids, 
                            attention_mask = attention_mask, 
                            labels = targets
                        )


        loss = outputs[0]
        self.batch_loss = loss.item()
        self.epoch_total_loss += self.batch_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        self.batch_loss_values.append(self.batch_loss)

        return self


    def _replace_padding_token(self, targets):
        targets[targets == self.tokenizer.pad_token_id] = -100
        self.targets = targets

        return self


    def save_model(self, model_name, dir):
        PATH = os.path.join(dir, model_name)
        torch.save(self.model, PATH)
        _logger.info(f"model saved in path {PATH}")

        return self