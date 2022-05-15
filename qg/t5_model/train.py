# import sys
# import pathlib
# PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[2]
# sys.path.append(str(PACKAGE_ROOT))


import os
import json

import torch
from torch.utils.data import DataLoader, RandomSampler
import numpy as np
import random
import argparse

from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer, 
    AdamW, 
    get_linear_schedule_with_warmup,
    )

from qg.t5_model.objects.encoder import Encoder
from qg.t5_model.objects.preprocessor import DataProcessor
from qg.config.config import get_logger, device, today, now, PACKAGE_ROOT
_logger = get_logger(logger_name=__name__)
RESULTS_T5_DIR = PACKAGE_ROOT/"qg"/"t5_model"/f"results_{today}_{now}"
RESULTS_T5_DIR.mkdir(exist_ok=True)


encoder_parser = argparse.ArgumentParser(description='Get all command line arguments.')
encoder_parser.add_argument('--model', type=str, default="t5-base", help='Tokenizer and pretrained model')
encoder_parser.add_argument('--model_name', type=str, default="t5_base", help='Tokenizer and pretrained model')
encoder_parser.add_argument('--batch_size', type=int, default=2, help='Specify the training batch size')
encoder_parser.add_argument('--preprocess', type=bool, default=True, help='Specify the global random seed')
encoder_parser.add_argument('--learning_rate', type=float, default=1e-4, help='Specify the initial learning rate')
encoder_parser.add_argument('--adam_epsilon', type=float, default=1e-6, help='Specify the AdamW loss epsilon')
encoder_parser.add_argument('--lr_decay', type=float, default=0.85, help='Specify the learning rate decay rate')
encoder_parser.add_argument('--dropout', type=float, default=0.1, help='Specify the dropout rate')
encoder_parser.add_argument('--n_epochs', type=int, default=10, help='Specify the number of epochs to train for')
encoder_parser.add_argument('--max_length_source', type=int, default=512, help='Maximum length of the source text')
encoder_parser.add_argument('--max_length_target', type=int, default=32, help='Maximum length of the target text')
encoder_parser.add_argument('--seed', type=int, default=1, help='Specify the global random seed')
encoder_parser.add_argument('--test', type=bool, default=False, help='Set to true for testing the code, it will un a shortcut')

def main(args):
        # Set the seed value all over the place to make this reproducible.
        seed_val = args.seed
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
        
        _logger.info(f"""Running. Test = {args.test}
        Epochs = {args.n_epochs}
        batch_size = {args.batch_size}
        model = {args.model}
        model_name = {args.model_name},
        results folder = {RESULTS_T5_DIR}
        """)
        
        train_data = load_dataset('squad_v2', split='train')
        
        if args.preprocess:
                print("processing dataset...")
                processor = DataProcessor(
                        sep_token=" <hl> ",
                        eos_token=" </s>"
                )
                train_data = processor.process(train_data)

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        encoder = Encoder(device=device)
        encoder.tokenize(
                        dataset=train_data, 
                        tokenizer=tokenizer, 
                        max_length_source=args.max_length_source, 
                        max_length_target=args.max_length_target, 
                        truncation=True, 
                        padding="max_length",
                        )
        
        encoder.TensorDataset()
                
        train_sampler = RandomSampler(encoder.dataset)
        train_dataloader = DataLoader(encoder.dataset, sampler=train_sampler, batch_size=args.batch_size)

        
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
        model.to(device)
        optimizer = AdamW(model.parameters(),
                        lr = args.learning_rate,
                        eps = args.adam_epsilon,
                        weight_decay = 0.01
                        )

        total_steps = len(train_dataloader) * args.n_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 0.1*total_steps,
                                                num_training_steps = total_steps)

        encoder.train_model(
                model = model, 
                dataset = train_dataloader,
                optimizer = optimizer, 
                scheduler = scheduler, 
                epochs = args.n_epochs,
                test = args.test
                )


        encoder.save_model(
                model_name = f'{args.model_name}.pt',
                dir = RESULTS_T5_DIR
                )
        
        results = {}
        results["device"] = device
        results["batch_size"] = args.batch_size
        results["learning_rate"] = args.learning_rate
        results["adam_epsilon"] = args.adam_epsilon
        results["lr_decay"] = args.lr_decay
        results["dropout"] = args.dropout
        results["n_epochs"] = args.n_epochs
        results["seed"] = args.seed
        results["model_name"] = f"{args.model_name}"
        results["model_path"] = f"{RESULTS_T5_DIR}"
        results["epoch_loss"] = encoder.epoch_loss_values
        results["batch_loss"] = encoder.batch_loss
        results["training_epoch_time"] = encoder.epoch_training_time
        results["total_training_time"] = encoder.total_training_time
        
        file_name = f"results_{args.model_name}_{today}_{now}.json"
        PATH = os.path.join(RESULTS_T5_DIR, file_name)
        
        with open(PATH, "w", econding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False)
        
if __name__ == '__main__':
    encoder_arguments = encoder_parser.parse_args()
    main(encoder_arguments)