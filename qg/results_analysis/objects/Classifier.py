from datasets import load_metric
import torch
import numpy as np
from tqdm import tqdm
from qg.config.config import get_logger
_logger = get_logger(logger_name=__file__)

class ClassifierObject:

    def __init__(self, device) -> None:
        self.device = device

    def classify(self, data_loader, model, test):
        
        self.pred_y = []
        
        self.model = model

        # self.model = self.model.eval()
            
        for i, batch in tqdm(enumerate(data_loader)):
            self._classification_step(batch)

            if test and i>5:
                break

        return self



    def _classification_step(self, batch):

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        with torch.no_grad():
        
            outputs = self.model(
                input_ids = input_ids,
                attention_mask = attention_mask,
            )

        logits = outputs.logits ## logits

        predictions = torch.argmax(logits, dim=-1)

        list_predictions = predictions.detach().cpu().numpy().tolist()
        assert type(list_predictions) == list

        self.pred_y.extend(list_predictions)
    
        return self