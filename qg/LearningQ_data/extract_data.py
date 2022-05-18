from qg.LearningQ_data.DataExtractor import LearningQDataExtractor
import argparse


data_parser = argparse.ArgumentParser(description='Get all command line arguments.')
data_parser.add_argument('--task', type=str, default="classification", help='Either "classification" or "qg"')

def main(args):
    extractor = LearningQDataExtractor(zipfile_name = "qg/LearningQ_data/LearningQ.zip")

    #  python -m qg.LearningQ_data.extract_data --task classification
    if args.task == "classification":
        extractor.extract_data(data_path = "data/khan/khan_labeled_data", task = "classification")
        extractor.transform_classification_data()
        extractor.save_json_file(filename = "qg/LearningQ_data/LearningQ_classification.json",
                                data = extractor.formatted_data)

    elif args.task == "qg":
        pass

if __name__ == '__main__':
    encoder_arguments = data_parser.parse_args()
    main(encoder_arguments)
