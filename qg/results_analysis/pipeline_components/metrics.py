import numpy as np
import spacy
import json
import os
from sklearn.base import BaseEstimator, TransformerMixin
from qg.results_analysis.objects.ComputeRouge import ComputeRougeObject
from qg.results_analysis.objects.ComputeBleu import ComputeBleuObject
from qg.config.config import get_logger, PACKAGE_ROOT
_logger = get_logger(logger_name=__file__)
try:
    nlp = spacy.load("en_core_web_lg")
except:
    os.system("python -m spacy download en_core_web_lg")
    nlp = spacy.load("en_core_web_lg")


class QuestionsMetrics(BaseEstimator, TransformerMixin):
    
    def __init__(self, results_folders, splits):
        self.results_folders = results_folders
        self.splits = splits

    def fit(self, X, y=None):
        return self

    def transform(self, X: dict) -> dict:
                
        for split in self.splits:
            for folder in self.results_folders:

                data = X[f"{split}_{folder}"]["data"]
                scores = X[f"{split}_{folder}"]["scores"]

                # with open(PACKAGE_ROOT/f"qg/transformers_models/experiment_{folder}/mapped_{split}_questions.json", encoding="utf-8") as f:
                #     data = json.load(f)
                # with open(PACKAGE_ROOT/f"qg/transformers_models/experiment_{folder}/scores_{split}_questions.json", encoding="utf-8") as f:
                #     scores = json.load(f)

                ### COMPUTING ROUGE AND F1 SCORE ###
                ####################################
                _logger.info(f"Computing ROUGE score for {folder}, {split}...")
                rouge = ComputeRougeObject()
                rouge.compute_rouge_scores(data=data, scores=scores)
                scores = rouge.scores


                ### COMPUTING BLEU SCORES ####
                ##############################
                _logger.info(f"Computing BLEU scores for {folder}, {split}...")

                bleu_settings = ["tokens", "lemmas"]
                bleu = ComputeBleuObject()

                for bleu_setting in bleu_settings:
                    bleu.compute_scores(data["predictions"], data["references"], setting=bleu_setting)

                    for key in bleu.__dict__:
                        # adding bleu scores to scores dict...
                        key_name = f"{key}_{bleu_setting}"
                        scores[key_name] = bleu.__dict__[key]
                        # calculating average bleu scores and adding it to scores dict...
                        average_key_name = f"average_{key_name}"
                        scores[average_key_name] = np.mean(bleu.__dict__[key])

                with open(PACKAGE_ROOT/f"qg/transformers_models/experiment_{folder}/scores_{split}_questions.json", "w", encoding="utf-8") as f:
                        json.dump(scores, f, ensure_ascii=False, indent=4)
                _logger.info(f'Metrics results saved in: {PACKAGE_ROOT/f"qg/transformers_models/experiment_{folder}/scores_{split}_questions"}.')


        return X