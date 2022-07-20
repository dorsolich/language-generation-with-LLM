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
from qg.transformers_models.pipeline_components.preprocess import DataProcessor
from qg.transformers_models.pipeline_components.encode import (
    PreTrainedTokenizerDownloader,
    Encoder,
    DataLoaderComponent
)
from qg.transformers_models.pipeline_components.model import PreTrainedModelDownloader
from qg.transformers_models.pipeline_components.train import Trainer
from qg.transformers_models.pipeline_components.validate import ModelValidator
from qg.transformers_models.arguments.args_qg_training import encoder_parser
from qg.config.config import get_logger, device, today, now, PACKAGE_ROOT
_logger = get_logger(logger_name=__file__)
RESULTS_DIR = PACKAGE_ROOT/"qg"/"transformers_models"/f"results_{today}_{now}"
RESULTS_DIR.mkdir(exist_ok=True)
args = encoder_parser.parse_args()

seed_val = args.seed
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
    

qg_training_pipeline = Pipeline(
    [
        (
            "DatasetUploader",
            DatasetLoader(dataset=args.dataset, split="train"),
        ),
        (
            "DataProcessor",
            DataProcessor(preprocess_setting=args.preprocess_setting)
        ),
        (
            "PreTrainedTokenizerDownloader",
            PreTrainedTokenizerDownloader(model=args.model)
        ),
        (
            "Encoder",
            Encoder(
                device=device, 
                max_length_source=args.max_length_source, 
                max_length_target=args.max_length_target,
                batch_size=args.batch_size,
            )
        ),
        (
            "DataLoaderComponent",
            DataLoaderComponent(task="QuestionGeneration", batch_size=args.batch_size)
        ),
        (
            "PreTrainedModelUploader",
            PreTrainedModelDownloader(task="QuestionGeneration", model=args.model)
        ),
        (
            "Trainer",
            Trainer(
                device,
                learning_rate=args.learning_rate,
                adam_epsilon=args.adam_epsilon,
                n_epochs=args.n_epochs,
                test=args.test,
                model_name=args.model_name,
                results_dir=RESULTS_DIR,
                task="QuestionGeneration"
            )

        ),
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
    y = qg_training_pipeline.transform(X)
    
    results = {}
    results["device"] = device
    results["len_dataset"] = len(y["dataset"]) # processed dataset
    results["example_context"] = y["dataset"][0]["context"]
    results["example_question"] = y["dataset"][0]["question"]
    


    ignore = ["model", "dataset", "encoded_dataset", "data_loader", "tokenizer", "metric",
            "scheduler", "optimizer", "epoch_total_loss", "batch_loss", "targets"]

    for arg in args.__dict__:
            results[arg] = args.__dict__[arg]
    for key in y:
        if key not in ignore:
            results[key] = y[key]

    file_name = f"results_{args.model_name}_{today}_{now}.json"
    PATH = os.path.join(RESULTS_DIR, file_name)
    
    with open(PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

    _logger.info(f"Task finished. Results saved in: {RESULTS_DIR}")