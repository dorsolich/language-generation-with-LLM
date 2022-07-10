from datasets import load_metric
from qg.config.config import get_logger
_logger = get_logger(logger_name=__file__)

class ComputeRougeObject:

    def __init__(self) -> None:
        pass

    def compute_rouge_scores(self, data, scores):
        ### COMPUTING ROUGE AND F1 SCORE ###
        ####################################

        rouge = load_metric("rouge")
        # splitting data in batches, so rouge metric doesn't run out of memory...
        prev_i = 0
        for i in range(0, len(data["predictions"]), 200):
            batched_predictions = data["predictions"][prev_i:i]
            batched_references = data["predictions"][prev_i:i]
            prev_i = i

            predictions = [batched_predictions]
            references = [[[ref] for ref in batched_references]]
            rouge.add_batch(predictions=predictions, references=references)
        
        # adding the last batch...
        if len(data["predictions"]) - i < 200:
            batch_pred = data["predictions"][i:]
            batch_ref = data["predictions"][i:]

            predictions = [batch_pred]
            references = [[[ref] for ref in batch_ref]]
            rouge.add_batch(predictions=predictions, references=references)
        
        results = rouge.compute()
        scores["rouge"] = results

        self.scores = scores
        return self