from datasets import load_metric
import torch
import numpy as np
from tqdm import tqdm
from qg.config.config import get_logger
_logger = get_logger(logger_name=__file__)

class ValidatorObject:

    def __init__(self, device) -> None:
        self.device = device

        self.batch_loss_values = []
        self.epochs_avg_loss_values = []


    def evaluate_cls_model(self, data_loader, epochs, model, test, metric):
        
        self.pred_y = []
        self.true_y = []
        
        # https://huggingface.co/metrics
        self.metric = load_metric(metric)
        self.model = model

        for epoch in tqdm(range(epochs)):
            self.model = self.model.eval()
            
            # self.accumulated_batch_flat_accuracy = 0
            # self.accumulated_batch_accuracy = 0
            self.epoch_total_loss = 0

            
            
            for i, batch in enumerate(data_loader):
                self._evaluation_step(batch)

                if test and i>5:
                    break

            avg_epoch_loss = self.epoch_total_loss / len(data_loader)
            self.epochs_avg_loss_values.append(avg_epoch_loss)

        self.validation_loss = np.mean(np.array(self.epochs_avg_loss_values))
        score = self.metric.compute()
        self.score = score

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
        list_targets = targets.detach().cpu().numpy().tolist()

        self.pred_y.extend(list_predictions)
        self.true_y.extend(list_targets)

        self.epoch_total_loss += batch_loss
        self.batch_loss_values.append(batch_loss)
    
        return self

    # def evaluate_qg_model(self, data_loader, epochs, model, test, metric, max_length_target, tokenizer):

    #     model.eval()
        
    #     self.metric = load_metric(metric)

    #     for epoch in range(epochs):
    #         for i, batch in tqdm(enumerate(data_loader)):
    #             with torch.no_grad():
    #                 generated_target_ids = model.generate(
    #                     batch["input_ids"],
    #                     attention_mask=batch["attention_mask"],
    #                     max_length=max_length_target,
    #                 )
    #             for generated_target_id in generated_target_ids:
    #                 decoded_outputs = tokenizer.decode(
    #                                                 generated_target_id,
    #                                                 skip_special_tokens=True,
    #                                                 clean_up_tokenization_spaces=True
    #                                                 )
    #                 batched_questions = []
    #                 for encoded_question in batch["target_ids"]:
    #                     decode_question = tokenizer.decode(
    #                                                     encoded_question,
    #                                                     skip_special_tokens=True,
    #                                                     clean_up_tokenization_spaces=True
    #                                                     )
    #                     batched_questions.append(decode_question)
                        

    #                 self.metric.add_batch(predictions=batched_questions, references=decoded_outputs)

                
    #             if self.test:
    #                 if i > 5:
    #                     break

    #         results = self.metric.compute()
    #         print(f"epoch {epoch}, BLEU score: {results['score']:.2f}")
        
    #     score = self.metric.compute()
    #     self.score = score
    #     return self