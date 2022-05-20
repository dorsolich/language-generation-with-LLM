import os
import json
from sklearn.pipeline import Pipeline
from qg.transformers_models.pipeline_components.dataset import DatasetLoader
from qg.transformers_models.pipeline_components.encode import PreTrainedTokenizerDownloader
from qg.transformers_models.pipeline_components.model import TrainedModelUploader
from qg.transformers_models.pipeline_components.decode import Decoder
from qg.transformers_models.arguments.args_qg_decoder import decoder_parser
from qg.config.config import get_logger, device, today, now, PACKAGE_ROOT
_logger = get_logger(logger_name=__file__)
args = decoder_parser.parse_args()
RESULTS_T5_DIR = PACKAGE_ROOT/"qg"/"transformers_models"/args.results_folder

qg_decoder_pipeline = Pipeline(
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
            "TrainedModelUploader",
            TrainedModelUploader(
                model = args.model,
                model_name = args.model_name,
                model_dir = RESULTS_T5_DIR,
                device = device,
            )
        ),
        (
            "Decoder",
            Decoder(
                device = device,
                context_max_length = args.context_max_length,
                question_max_length = args.question_max_length,
                num_beams = args.num_beams,
                test = args.test,
            )
        )
    ]
)

if __name__ == '__main__':
    
    X = {}
    y = qg_decoder_pipeline.transform(X)

    results = {}
    results["source_texts"] = y["source_texts"]
    results["target_texts"] = y["target_texts"]
    results["model_outputs"] = y["model_outputs"]
    results["batch_size"] = args.batch_size
    results["num_beams"] = args.num_beams
    
    file_name = f"{args.dataset_split}_questions_{today}_{now}.json"
    PATH = os.path.join(RESULTS_T5_DIR, file_name)
    with open(PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    with open(RESULTS_T5_DIR/f"{args.dataset_split}_source_texts.txt", 'w', encoding='utf-8') as f:
        f.write("\n".join(y["source_texts"]))

    with open(RESULTS_T5_DIR/f"{args.dataset_split}_target_texts.txt", 'w') as f:
        f.write("\n".join(y["target_texts"]))

    with open(RESULTS_T5_DIR/f"{args.dataset_split}_model_outputs.txt", 'w') as f:
        f.write("\n".join(y["model_outputs"]))

    _logger.info(f"Questions file: {file_name}, and text files saved in path: {RESULTS_T5_DIR}")


