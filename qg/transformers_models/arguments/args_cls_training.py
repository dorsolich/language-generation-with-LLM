import argparse

cls_train_parser = argparse.ArgumentParser(description='Get all command line arguments.')
cls_train_parser.add_argument('--seed', type=int, default=1, help='Specify the global random seed to make results reproducible')
cls_train_parser.add_argument('--test', type=bool, default=False, help='Set to true for testing the code, it will un a shortcut')

cls_train_parser.add_argument('--dataset', type=str, default="LearningQ", help='Select the dataset')
cls_train_parser.add_argument('--dataset_split', type=str, default="train", help='Dataset split (train, validation)')


cls_train_parser.add_argument('--model', type=str, default="distilbert-base-cased", help='Tokenizer and pretrained model')
cls_train_parser.add_argument('--model_name', type=str, default="distilbert_base_cased", help='Tokenizer and pretrained model')

# https://arxiv.org/pdf/1810.04805.pdf
cls_train_parser.add_argument('--learning_rate', type=float, default=5e-5, help='Initial learning rate')
cls_train_parser.add_argument('--adam_epsilon', type=float, default=1e-08, help='AdamW loss epsilon')

cls_train_parser.add_argument('--batch_size', type=int, default=32, help='Specify the training batch size')
cls_train_parser.add_argument('--n_epochs', type=int, default=4, help='Number of training epochs')


cls_train_parser.add_argument('--lr_decay', type=float, default=0.85, help='Learning rate decay rate')
cls_train_parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')

cls_train_parser.add_argument('--max_length_source', type=int, default=100, help='Maximum length of the source text')
cls_train_parser.add_argument('--max_length_target', type=int, default=32, help='Maximum length of the target text')
