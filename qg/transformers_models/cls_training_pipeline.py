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
from qg.transformers_models.pipeline_components.model import PreTrainedModelDownloader
from qg.transformers_models.pipeline_components.train import Trainer
from qg.transformers_models.pipeline_components.validate import ModelValidator
from qg.transformers_models.arguments.args_cls_training import cls_train_parser
from qg.config.config import get_logger, device, today, now, PACKAGE_ROOT
_logger = get_logger(logger_name=__file__)
args = cls_train_parser.parse_args()


RESULTS_DIR = PACKAGE_ROOT/"qg"/"transformers_models"/f"classifier_{args.learning_rate}"
RESULTS_DIR.mkdir(exist_ok=True)

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
            "PreTrainedModelDownloader",
            PreTrainedModelDownloader(task="SequenceClassification", model=args.model)
        ),
        (
            "Trainer",
            Trainer(
                device = device,
                learning_rate = args.learning_rate,
                adam_epsilon = args.adam_epsilon,
                n_epochs = args.n_epochs,
                test = args.test,
                model_name = args.model_name,
                results_dir = RESULTS_DIR,
                task = "SequenceClassification",
                evaluation_metric = "accuracy"
            )
        ),
    ]
)


from sklearn import set_config
set_config(display="diagram")
cls_training_pipeline

if __name__ == '__main__':
    _logger.info(f"""Running. 
    Test = {args.test}
    Epochs = {args.n_epochs}
    learning_rate = {args.learning_rate}
    batch_size = {args.batch_size}
    model = {args.model}
    model_name = {args.model_name},
    results folder = {RESULTS_DIR}
    """)    

    with open(RESULTS_DIR/f"{args.dataset_split}_pipeline_params.txt", 'w', encoding='utf-8') as f:
        f.write(str(cls_training_pipeline.get_params()))

    
    ### RUNNING PIPELINE ###
    X = {}
    y = cls_training_pipeline.transform(X)
    
    ### SAVING RESULTS ###
    results = {}
    results["device"] = device
    results["len_dataset"] = len(y["dataset"]) # processed dataset
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