from logging import raiseExceptions
import sys
import pathlib
PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(PACKAGE_ROOT))
import numpy as np
import spacy
import json
import os
from datasets import load_dataset
from qg.config.config import get_logger, PACKAGE_ROOT
from qg.results_analysis.MapQuestions import MapReferencesWithGenerateQuestions
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
        # Uploading generated questions...
        with open(PACKAGE_ROOT/f"qg/transformers_models/t5small_batch32_{folder}/{split}_model_outputs.txt", encoding="utf-8") as f:
            gen_questions = f.readlines()
            gen_questions = [gen_question.strip('\n') for gen_question in gen_questions]

        # Uploading source texts...
        with open(PACKAGE_ROOT/f"qg/transformers_models/t5small_batch32_{folder}/{split}_source_texts.txt", encoding="utf-8") as f:
            source_texts = f.readlines()
            source_texts = [source_text.strip('\n') for source_text in source_texts]

        # Uploading ref questions...
        with open(PACKAGE_ROOT/f"qg/transformers_models/t5small_batch32_{folder}/{split}_target_texts.txt", encoding="utf-8") as f:
            ref_questions = f.readlines()
            ref_questions = [ref_question.strip('\n') for ref_question in ref_questions]
        # Uploading raw dataset...
        dataset = load_dataset('squad_v2', split=split)

        _logger.info(f"Wrangling {folder}, {split}")

        # The target is to gather all the reference questions that happens in the raw dataset for each unique source text
        # and to gather the model generated question for each unique source text
        # and place the reference questions and the model generate question in a data dictionary
        mapper = MapReferencesWithGenerateQuestions()
        generated_questions, target_questions = mapper.map_references_with_predictions(dataset, source_texts, ref_questions, gen_questions)
        data, scores = mapper.select_references_by_similarity(generated_questions, target_questions)



        with open(PACKAGE_ROOT/f"qg/transformers_models/t5small_batch32_{folder}/mapped_{split}_questions.json", "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

        with open(PACKAGE_ROOT/f"qg/transformers_models/t5small_batch32_{folder}/scores_{split}_questions.json", "w", encoding="utf-8") as f:
                json.dump(scores, f, ensure_ascii=False, indent=4)