from logging import raiseExceptions
import sys
import pathlib
PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(PACKAGE_ROOT))
import numpy as np
import spacy
import pandas as pd
import json
import os
from datasets import load_metric, load_dataset
from qg.results_analysis.ComputeBleu import ComputeBleuObject
from qg.config.config import get_logger, PACKAGE_ROOT
_logger = get_logger(logger_name=__file__)
try:
    nlp = spacy.load("en_core_web_lg")
except:
    os.system("python -m spacy download en_core_web_lg")
    nlp = spacy.load("en_core_web_lg")

results_folders = ["AA", "AQPL", "basic", "OQPL"]
splits = ["train", "validation"]

for split in splits:
    for folder in results_folders:

        with open(PACKAGE_ROOT/f"qg/transformers_models/t5small_batch32_{folder}/mapped_{split}_questions.json", encoding="utf-8") as f:
            data = json.load(f)
        with open(PACKAGE_ROOT/f"qg/transformers_models/t5small_batch32_{folder}/scores_{split}_questions.json", encoding="utf-8") as f:
            scores = json.load(f)

        # test_pref = data["predictions"][:200]
        # test_ref = data["references"][:200]


        ### COMPUTING ROUGE AND F1 SCORE ###
        ####################################
        _logger.info(f"Computing ROUGE score for {folder}, {split}...")

        rouge = load_metric("rouge")
        prev_i = 0
        for i in range(0, len(data["predictions"]), 200):
            batched_predictions = data["predictions"][prev_i:i]
            batched_references = data["predictions"][prev_i:i]
            prev_i = i

            predictions = [batched_predictions]
            references = [[[ref] for ref in batched_references]]
            rouge.add_batch(predictions=predictions, references=references)
            
        if len(data["predictions"]) - i < 200:
            batch_pred = data["predictions"][i:]
            batch_ref = data["predictions"][i:]

            predictions = [batched_predictions]
            references = [[[ref] for ref in batched_references]]
            rouge.add_batch(predictions=predictions, references=references)
        
        results = rouge.compute()
        
        scores["rouge"] = results

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

        with open(PACKAGE_ROOT/f"qg/transformers_models/t5small_batch32_{folder}/scores_{split}_questions.json", "w", encoding="utf-8") as f:
                json.dump(scores, f, ensure_ascii=False, indent=4)