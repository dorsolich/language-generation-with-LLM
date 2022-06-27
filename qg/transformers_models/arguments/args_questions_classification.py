import argparse


questions_cls_parser = argparse.ArgumentParser(description='Get all command line arguments.')
questions_cls_parser.add_argument('--seed', type=int, default=1, help='Specify the global random seed to make results reproducible')
questions_cls_parser.add_argument('--test', type=bool, default=False, help='Set to true for testing the code, it will un a shortcut')
questions_cls_parser.add_argument('--preprocess_setting', type=str, default="AQPL", help="'AQPL', 'OQPL', 'AA' or 'basic'")

questions_cls_parser.add_argument('--classifier_folder', type=str, help="Folder where the classifier model is located")

questions_cls_parser.add_argument('--dataset', type=str, default="generated_questions", help='Select the dataset')
questions_cls_parser.add_argument('--dataset_split', type=str, default="validation", help='Dataset split (train, validation)')


questions_cls_parser.add_argument('--model', type=str, default="distilbert-base-cased", help='Tokenizer and pretrained model')
questions_cls_parser.add_argument('--model_name', type=str, default="distilbert_base_cased", help='Tokenizer and pretrained model')

questions_cls_parser.add_argument('--batch_size', type=int, default=32, help='Specify the training batch size')

questions_cls_parser.add_argument('--max_length_source', type=int, default=100, help='Maximum length of the source text')
questions_cls_parser.add_argument('--max_length_target', type=int, default=32, help='Maximum length of the target text')
