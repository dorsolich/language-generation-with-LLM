import argparse

cls_validation_parser = argparse.ArgumentParser(description='Get all command line arguments.')
cls_validation_parser.add_argument('--seed', type=int, default=1, help='Specify the global random seed to make results reproducible')
cls_validation_parser.add_argument('--test', type=bool, default=False, help='Set to true for testing the code, it will un a shortcut')
cls_validation_parser.add_argument('--results_folder', type=str, help='Load path of fine-tuned model and save results')

cls_validation_parser.add_argument('--dataset', type=str, default="LearningQ", help='Select the dataset')
cls_validation_parser.add_argument('--dataset_split', type=str, default="test", help='Dataset split (train, validation)')


cls_validation_parser.add_argument('--model', type=str, default="distilbert-base-cased", help='Tokenizer and pretrained model')
cls_validation_parser.add_argument('--model_name', type=str, default="distilbert_base_cased", help='Tokenizer and pretrained model')

cls_validation_parser.add_argument('--batch_size', type=int, default=32, help='Specify the training batch size')
cls_validation_parser.add_argument('--n_epochs', type=int, default=4, help='Number of training epochs')

cls_validation_parser.add_argument('--max_length_source', type=int, default=100, help='Maximum length of the source text')
cls_validation_parser.add_argument('--max_length_target', type=int, default=32, help='Maximum length of the target text')
