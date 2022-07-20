
import time
import torch
from torch import nn
import os
from tqdm import tqdm
from datasets import load_metric

from qg.utils import format_time
from qg.config.config import get_logger
_logger = get_logger(logger_name=__file__)


class TrainerObject:
    def __init__(self, device):
        self.device = device

        # initializing trainer variables
        # training_epoch_loss_values: average loss value of each epoch (sum of batch loss / len(data_loader))
        self.training_epoch_loss_values = []
        # training_batch_loss_values: loss value of each batch of for all the epochs
        self.training_batch_loss_values = []
        # training_epoch_accuracy_values: accuracy of each epoch (preds y VS real y in all the batches in one epoch)
        self.training_epoch_accuracy_values = [] # only for cls
        
        self.epoch_training_time = []

        self.pred_y = []
        self.true_y = []

    def train_model(self, 
            model, 
            tokenizer,
            data_loader,
            task,
            optimizer, 
            scheduler, 
            epochs,
            evaluation_metric,
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
            
            if self.task == "SequenceClassification":  
                self.metric = load_metric(evaluation_metric)

            init_epoch_time = time.time()
            self.epoch_total_loss = 0
            self.model.train()

            for i, batch in tqdm(enumerate(data_loader)):
                torch.cuda.empty_cache()
                self._training_step(batch)

                if test and i>5:
                    break
            
            # calculating epoch loss
            avg_epoch_loss = self.epoch_total_loss / len(data_loader)
            self.training_epoch_loss_values.append(avg_epoch_loss)
            self.epoch_training_time.append(format_time(time.time() - init_epoch_time))

            if self.task == "SequenceClassification":  
                # calculating epoch additional metric
                score = self.metric.compute()
                self.training_epoch_accuracy_values.append(score) # only for cls
        
        self.total_training_time = format_time(time.time() - init_training_time)
        
        return self



    def _training_step(self, batch):
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        # targets are different depending on the task...

        if self.task == "SequenceClassification":            
            targets = batch["target_ids"].to(self.device)

        elif self.task == "QuestionGeneration":
            self._replace_padding_token(batch["target_ids"])
            targets = self.targets.to(self.device)

        self.model.zero_grad()
        outputs = self.model(
                        input_ids = input_ids, 
                        attention_mask = attention_mask, 
                        labels = targets
                    )
        # calculating batch loss
        loss = outputs[0]
        self.batch_loss = loss.item()
        self.epoch_total_loss += self.batch_loss
        loss.backward()
        # updating optimizer and scheduler
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        self.training_batch_loss_values.append(self.batch_loss)

        if self.task == "SequenceClassification":
            # calculating additional batch metric
            logits = outputs[1] ## logits
            predictions = torch.argmax(logits, dim=-1)
            self.metric.add_batch(predictions=predictions, references=targets)
            list_predictions = predictions.detach().cpu().numpy().tolist()
            list_targets = targets.to('cpu').numpy().tolist()
            assert type(list_predictions) == list
            assert type(list_targets) == list
            self.pred_y.extend(list_predictions)
            self.true_y.extend(list_targets)

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