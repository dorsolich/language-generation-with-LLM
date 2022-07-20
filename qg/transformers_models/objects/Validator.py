from datasets import load_metric
import torch
from tqdm import tqdm
from qg.config.config import get_logger
_logger = get_logger(logger_name=__file__)

class ValidatorObject:

    def __init__(self, device) -> None:
        self.device = device

        self.validation_batch_loss_values = []
        self.validation_epoch_loss_values = []
        self.validation_epoch_accuracy_values = []


    def evaluate_cls_model(self, data_loader, epochs, model, test, metric):
        
        self.pred_y = []
        self.true_y = []
        
        self.model = model

        for epoch in tqdm(range(epochs)):
            self.model = self.model.eval()
            # https://huggingface.co/metrics
            self.metric = load_metric(metric)

            self.epoch_total_loss = 0
            
            for i, batch in enumerate(data_loader):
                self._evaluation_step(batch)

                if test and i>5:
                    break

            avg_epoch_loss = self.epoch_total_loss / len(data_loader)
            self.validation_epoch_loss_values.append(avg_epoch_loss)

            
            # calculating epoch additional metric
            score = self.metric.compute()
            self.validation_epoch_accuracy_values.append(score) # only for cls

        return self



    def _evaluation_step(self, batch):

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        targets = batch["target_ids"].to(self.device)

        with torch.no_grad():
        
            outputs = self.model(
                input_ids = input_ids,
                attention_mask = attention_mask,
                labels = targets
            )

        loss = outputs[0] ## loss
        batch_loss = loss.item()
        logits = outputs[1] ## logits

        predictions = torch.argmax(logits, dim=-1)
        self.metric.add_batch(predictions=predictions, references=targets)

        list_predictions = predictions.detach().cpu().numpy().tolist()
        list_targets = targets.to('cpu').numpy().tolist()
        assert type(list_predictions) == list
        assert type(list_targets) == list

        self.pred_y.extend(list_predictions)
        self.true_y.extend(list_targets)

        self.epoch_total_loss += batch_loss
        self.validation_batch_loss_values.append(batch_loss)
    
        return self