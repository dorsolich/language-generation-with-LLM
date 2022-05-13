import argparse
import os
import json

import torch
from tqdm import tqdm 

from datasets import load_dataset
from transformers import T5Tokenizer

from config import get_logger, device, today, now, PACKAGE_ROOT
from decoder import Decoder
_logger = get_logger(logger_name=__name__)

decoder_parser = argparse.ArgumentParser(description='Get all command line arguments.')
decoder_parser.add_argument('--model', type=str, default="t5-base", help='Tokenizer and pretrained model')
decoder_parser.add_argument('--results_folder', type=str, help='Load path of trained model and save test results')
decoder_parser.add_argument('--model_name', type=str, default="t5-base", help='Fine-tuned model name')
decoder_parser.add_argument('--dataset_split', type=str, default="validation", help='Passage max length')
decoder_parser.add_argument('--context_max_length', type=int, default=512, help='Passage max length')
decoder_parser.add_argument('--question_max_length', type=int, default=32, help='Passage max length')
decoder_parser.add_argument('--batch_size', type=int, default=2, help='Specify the training batch size')
decoder_parser.add_argument('--num_beams', type=int, default=4, help='Number of beams to use')
decoder_parser.add_argument('--test', type=bool, default=False, help='Set to true for testing the code, it will un a shortcut')


def main(args):
    RESULTS_T5_DIR = PACKAGE_ROOT/args.results_folder

    test_data = load_dataset('squad_v2', split=args.dataset_split)

    tokenizer = T5Tokenizer.from_pretrained(args.model)
    
    file_name = f'{args.model_name}.pt'
    PATH = os.path.join(RESULTS_T5_DIR, file_name)
    model = torch.load(PATH, map_location=device)
    model.eval().to(device)

    decoder = Decoder(device=device)
    prev_passage = ""
    for i, ex in tqdm(enumerate(test_data)):
        if args.test == True:
            if i == 0: # uncomment to debug
                if len(ex["answers"]["text"])==0:
                    continue
                if ex["context"].replace('\n', '')==prev_passage:
                    continue
                decoder.tokenize(
                                example=ex,
                                tokenizer=tokenizer, 
                                max_length_source=args.context_max_length, 
                                truncation=True, 
                                padding="max_length",
                                return_tensors = "pt"
                                )
                decoder.decode(
                                model=model, 
                                num_beams=args.num_beams,  
                                question_max_length=args.question_max_length, 
                                repetition_penalty=2.5,
                                length_penalty=1,
                                early_stopping=True,
                                use_cache=True,
                                num_return_sequences=1,
                                do_sample=False
                                )
        else:
            if len(ex["answers"]["text"])==0:
                continue
            if ex["context"].replace('\n', '')==prev_passage:
                continue
            decoder.tokenize(
                            example=ex,
                            tokenizer=tokenizer, 
                            max_length_source=args.context_max_length, 
                            truncation=True, 
                            padding="max_length",
                            return_tensors = "pt"
                            )
            decoder.decode(
                            model=model, 
                            num_beams=args.num_beams,  
                            question_max_length=args.question_max_length, 
                            repetition_penalty=2.5,
                            length_penalty=1,
                            early_stopping=True,
                            use_cache=True,
                            num_return_sequences=1,
                            do_sample=False
                            )
                

    results = {}
    results["passages"] = decoder.contexts
    results["questions"] = decoder.questions
    results["gen_questions"] = decoder.generated_questions
    results["batch_size"] = args.batch_size
    results["num_beams"] = args.num_beams
    
    file_name = f"{args.dataset_split}_questions_{today}_{now}.json"
    PATH = os.path.join(RESULTS_T5_DIR, file_name)
    with open(PATH, "w") as f:
        json.dump(results, f)
    _logger.info(f"Questions file: {file_name}, saved in path: {RESULTS_T5_DIR}")

if __name__ == '__main__':
    decoder_arguments = decoder_parser.parse_args()
    main(decoder_arguments)