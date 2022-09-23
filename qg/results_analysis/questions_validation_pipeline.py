import sys
import pathlib
PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(PACKAGE_ROOT))

import os
import json
import torch
import numpy as np
import random
from sklearn.pipeline import Pipeline
from qg.results_analysis.pipeline_components.map_questions import QuestionsMapperAndSimilarity
from qg.results_analysis.pipeline_components.metrics import QuestionsMetrics
from qg.config.config import get_logger, device, PACKAGE_ROOT
_logger = get_logger(logger_name=__file__)

seed_val = 1
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


results_folders = ["AA", "AQPL", "basic", "OQPL"]
splits = ["train", "validation"]

questions_validation_pipeline = Pipeline(
    [
        (
            "QuestionsMapperAndSimilarity",
            QuestionsMapperAndSimilarity(results_folders=results_folders, splits=splits)
        ),
        (
            "QuestionsMetrics",
            QuestionsMetrics(results_folders=results_folders, splits=splits)

        )
    ]
)

        
if __name__ == '__main__':
    _logger.info(f"""{questions_validation_pipeline} Running. 
    """)
    results_folders = ["AA", "AQPL", "basic", "OQPL"]
    splits = ["train", "validation"]

    X = {}

    for split in splits:
            for folder in results_folders:

                try:
    
                    # Uploading generated questions...
                    with open(PACKAGE_ROOT/f"qg/transformers_models/experiment_{folder}/{split}_model_outputs.txt", encoding="utf-8") as f:
                        gen_questions = f.readlines()
                        gen_questions = [gen_question.strip('\n') for gen_question in gen_questions]

                    # Uploading source texts...
                    with open(PACKAGE_ROOT/f"qg/transformers_models/experiment_{folder}/{split}_source_texts.txt", encoding="utf-8") as f:
                        source_texts = f.readlines()
                        source_texts = [source_text.strip('\n') for source_text in source_texts]

                    # Uploading ref questions...
                    with open(PACKAGE_ROOT/f"qg/transformers_models/experiment_{folder}/{split}_target_texts.txt", encoding="utf-8") as f:
                        ref_questions = f.readlines()
                        ref_questions = [ref_question.strip('\n') for ref_question in ref_questions]

                    X[f"{split}_{folder}"] = {}
                    X[f"{split}_{folder}"]["gen_questions"] = gen_questions
                    X[f"{split}_{folder}"]["source_texts"] = source_texts
                    X[f"{split}_{folder}"]["ref_questions"] = ref_questions

                except:
                    _logger.info(f"{folder} - {split} data not found")
                    continue

    ret = questions_validation_pipeline.transform(X)

    _logger.info(f"Task finished.")