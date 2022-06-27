import argparse

decoder_parser = argparse.ArgumentParser(description='Get all command line arguments.')
decoder_parser.add_argument('--model', type=str, default="t5-small", help='Tokenizer and pretrained model')
decoder_parser.add_argument('--results_folder', type=str, help='Load path of fine-tuned model and save decoder results')
decoder_parser.add_argument('--model_name', type=str, default="t5_small", help='Fine-tuned model name')
decoder_parser.add_argument('--dataset_split', type=str, default="validation", help='Dataset split (train, validation)')
decoder_parser.add_argument('--context_max_length', type=int, default=512, help='Context max length')
decoder_parser.add_argument('--question_max_length', type=int, default=32, help='Passage max length')
decoder_parser.add_argument('--batch_size', type=int, default=2, help='Training batch size')
decoder_parser.add_argument('--num_beams', type=int, default=4, help='Number of beams to use')
decoder_parser.add_argument('--test', type=bool, default=False, help='Set to true for testing the code, it will un a shortcut')
decoder_parser.add_argument('--dataset', type=str, default="squad_v2", help='Select the dataset')