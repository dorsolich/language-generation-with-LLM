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
from qg.transformers_models.pipeline_components.dataset import DatasetLoader
from qg.transformers_models.pipeline_components.encode import (
    PreTrainedTokenizerDownloader,
    Encoder,
    DataLoaderComponent
)
from qg.transformers_models.pipeline_components.model import TrainedModelUploader
from qg.results_analysis.pipeline_components.classify import QuestionsClassifier
from qg.transformers_models.arguments.args_questions_classification import questions_cls_parser
from qg.config.config import get_logger, device, PACKAGE_ROOT
_logger = get_logger(logger_name=__file__)
args = questions_cls_parser.parse_args()
RESULTS_DIR = PACKAGE_ROOT/"qg"/"transformers_models"/f"experiment_{args.preprocess_setting}"
DATA_DIR = RESULTS_DIR/f"mapped_{args.dataset_split}_questions.json"
CLS_MODEL_DIR = PACKAGE_ROOT/"qg"/"transformers_models"/f"{args.classifier_folder}"

seed_val = args.seed
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)



questions_classification_pipeline = Pipeline(
    [
        (
            "DatasetUploader",
            DatasetLoader(
                dataset=args.dataset, 
                split=args.dataset_split,
                data_dir = DATA_DIR
                ),
        ),
        (
            "PreTrainedTokenizerDownloader",
            PreTrainedTokenizerDownloader(model=args.model)
        ),
        (
            "Encoder",
            Encoder(
                device = device,
                max_length_source = args.max_length_source,
                batch_size = args.batch_size,
                dataset = args.dataset
            )
        ),
        (
            "DataLoaderComponent",
            DataLoaderComponent(task="SequenceClassification", batch_size=args.batch_size)
        ),
        (
            "TrainedModelUploader",
            TrainedModelUploader(
                model = args.model,
                model_name = args.model_name,
                model_dir = CLS_MODEL_DIR,
                device = device,
            )
        ),
        (
            "QuestionsClassifier",
            QuestionsClassifier(
                device=device,
                test=args.test
            )
        )
    ]
)

        
if __name__ == '__main__':
    _logger.info(f"""Running. 
    Test = {args.test}
    batch_size = {args.batch_size}
    model = {args.model}
    model_name = {args.model_name},
    results folder = {RESULTS_DIR}
    data = {DATA_DIR}
    """)

    X = {}
    y = questions_classification_pipeline.transform(X)
    
    results = {}
    results["device"] = device
    results["len_dataset"] = len(y["dataset"]) # processed dataset
    results["example_question"] = y["dataset"][0]

    ignore = ["model", "dataset", "encoded_dataset", "data_loader", "tokenizer", "metric",
            "scheduler", "optimizer", "epoch_total_loss", "batch_loss"]
    for arg in args.__dict__:
            results[arg] = args.__dict__[arg]
    for key in y:
        if key not in ignore:
            results[key] = y[key]
    
    file_name = f"classification_{args.dataset_split}_questions.json"
    PATH = os.path.join(RESULTS_DIR, file_name)
    
    with open(PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

    _logger.info(f"Task finished. Results saved in: {PATH}")