import sys
import pathlib
import numpy as np
import spacy
import pandas as pd
import json
import os
from datasets import load_dataset
from sklearn.base import BaseEstimator, TransformerMixin
from qg.results_analysis.objects.MapQuestions import MapReferencesWithGenerateQuestions
from qg.results_analysis.objects.CosineSimilarity import CosineSimilarityObject
from qg.config.config import get_logger, PACKAGE_ROOT
_logger = get_logger(logger_name=__file__)
try:
    nlp = spacy.load("en_core_web_lg")
except:
    os.system("python -m spacy download en_core_web_lg")
    nlp = spacy.load("en_core_web_lg")


class QuestionsMapperAndSimilarity(BaseEstimator, TransformerMixin):
    
    def __init__(self, results_folders, splits):
        self.results_folders = results_folders
        self.splits = splits

    def fit(self, X, y=None):
        return self

    def transform(self, X: dict) -> dict:
        """First, makes some questions wrangling. The target is to gather 
        all the reference questions that happens in the raw data-set for each unique source text,
        and to map it with the model-generated question for each unique source text.
        Then, place the reference questions and the model-generated question in a data dictionary

        Secondly, the cosine similarity is computed between each model-generated question and the
        reference questions that belong to each unique source text. This one-to-one reference-generated
        questions-pair and its cosine similarity is reported as the final output.

        Args:
            X (dict): A dictionary containing the generated questions,
            source texts and reference questions for each data-processing setting
            and each data-set split.

        Returns:
            dict: A dictionary containing the set of generated questions and the
            reference questions that have the highest cosine similarity for each 
            data-processing setting and each data-set split
        """


        for split in self.splits:
            for folder in self.results_folders:

                try: # save implementation in case AQPL counld not be run...

                    gen_questions = X[f"{split}_{folder}"]["gen_questions"] # Uploading generated questions...
                    source_texts = X[f"{split}_{folder}"]["source_texts"] # Uploading source texts...
                    ref_questions = X[f"{split}_{folder}"]["ref_questions"] # Uploading ref questions...
                    dataset = load_dataset('squad_v2', split=split) # Uploading raw dataset...

                    _logger.info(f"Wrangling {folder}, {split}...")

                    # The target is to gather all the reference questions that happens in the raw dataset for each unique source text
                    # and to gather the model generated question for each unique source text
                    # and place the reference questions and the model generate question in a data dictionary
                    mapper = MapReferencesWithGenerateQuestions(setting=folder)
                    generated_questions, target_questions = mapper.map_references_with_predictions(dataset, source_texts, ref_questions, gen_questions)
                    
                    # Filtering questions by Cosine Similarity
                    similarity = CosineSimilarityObject(setting=folder)
                    data, scores = similarity.filter_by_similarity(generated_questions, target_questions)


                    with open(PACKAGE_ROOT/f"qg/transformers_models/experiment_{folder}/mapped_{split}_questions.json", "w", encoding="utf-8") as f:
                            json.dump(data, f, ensure_ascii=False, indent=4)

                    with open(PACKAGE_ROOT/f"qg/transformers_models/experiment_{folder}/scores_{split}_questions.json", "w", encoding="utf-8") as f:
                            json.dump(scores, f, ensure_ascii=False, indent=4)

                    X[f"{split}_{folder}"] = {}
                    X[f"{split}_{folder}"]["data"] = data
                    X[f"{split}_{folder}"]["scores"] = scores

                except:
                    _logger.info(f"{folder} - {split} data not found")
                    continue

        return X