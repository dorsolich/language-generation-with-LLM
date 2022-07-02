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
from qg.transformers_models.pipeline_components.validate import ModelValidator
from qg.transformers_models.arguments.args_cls_validation import cls_validation_parser
from qg.config.config import get_logger, device, today, now, PACKAGE_ROOT
_logger = get_logger(logger_name=__file__)
args = cls_validation_parser.parse_args()
RESULTS_DIR = PACKAGE_ROOT/"qg"/"transformers_models"/args.results_folder

seed_val = args.seed
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)



cls_training_pipeline = Pipeline(
    [
        (
            "DatasetUploader",
            DatasetLoader(dataset=args.dataset, split=args.dataset_split),
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
                max_length_target = args.max_length_target,
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
                model_dir = RESULTS_DIR,
                device = device,
            )
        ),
        (
            "ModelValidator",
            ModelValidator(
                task="SequenceClassification", device=device, metric="accuracy", n_epochs=args.n_epochs, test=args.test
                )
        )
    ]
)

        
if __name__ == '__main__':
    _logger.info(f"""Running. 
    Test = {args.test}
    Epochs = {args.n_epochs}
    batch_size = {args.batch_size}
    model = {args.model}
    model_name = {args.model_name},
    results folder = {RESULTS_DIR}
    """)

    X = {}
    y = cls_training_pipeline.transform(X)
    
    results = {}
    results["device"] = device
    results["len_dataset"] = len(y["dataset"]["text"]) # processed dataset
    results["example_context"] = y["dataset"]["text"][0]
    results["example_question"] = y["dataset"]["labels"][0]

    ignore = ["model", "dataset", "encoded_dataset", "data_loader", "tokenizer", "metric",
            "scheduler", "optimizer", "epoch_total_loss", "batch_loss"]
    for arg in args.__dict__:
            results[arg] = args.__dict__[arg]
    for key in y:
        if key not in ignore:
            results[key] = y[key]
    
    file_name = f"{args.dataset_split}_results_{args.model_name}_{today}_{now}.json"
    PATH = os.path.join(RESULTS_DIR, file_name)
    
    with open(PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
    
    with open(RESULTS_DIR/f"{args.dataset_split}_metric.txt", 'w', encoding='utf-8') as f:
        f.write(str(y["metric"]))

    _logger.info(f"Task finished. Results saved in: {RESULTS_DIR}")