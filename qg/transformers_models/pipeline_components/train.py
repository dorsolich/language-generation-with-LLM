from sklearn.base import BaseEstimator, TransformerMixin
from transformers import AdamW, get_linear_schedule_with_warmup

from qg.transformers_models.objects.Trainer import TrainerObject


class Trainer(BaseEstimator, TransformerMixin):
    
    def __init__(self,
            device,
            task,
            learning_rate,
            adam_epsilon,
            n_epochs,
            test,
            model_name,
            results_dir,

        ):
        self.device = device
        self.learning_rate = learning_rate
        self.adam_epsilon = adam_epsilon
        self.n_epochs = n_epochs
        self.test = test
        self.model_name = model_name
        self.results_dir = results_dir
        self.task = task

    def fit(self, X, y=None):
        return self

    def transform(self, X: dict) -> dict:
        model = X["model"]
        model.to(self.device)
        optimizer = AdamW(
                        model.parameters(),
                        lr = self.learning_rate,
                        eps = self.adam_epsilon,
                        weight_decay = 0.01
                        )

        total_steps = len(X["data_loader"]) * self.n_epochs
        scheduler = get_linear_schedule_with_warmup(
                                                optimizer,
                                                num_warmup_steps = 0.1*total_steps,
                                                num_training_steps = total_steps
                                                )            
        trainer = TrainerObject(device=self.device)
        trainer.train_model(
                        tokenizer = X["tokenizer"],
                        model = model, 
                        data_loader = X["data_loader"],
                        task = self.task,
                        optimizer = optimizer, 
                        scheduler = scheduler, 
                        epochs = self.n_epochs,
                        test = self.test,
                        )
        trainer.save_model(
                        model_name = f'{self.model_name}.pt',
                        dir = self.results_dir
                        )

        X["epoch_loss_values"] = trainer.epoch_loss_values
        X["batch_loss_values"] = trainer.batch_loss_values
        X["epoch_training_time"] = trainer.epoch_training_time
        X["total_training_time"] = trainer.total_training_time
        
        if self.task == "SequenceClassification":
            X["model"] = model
        
        return X