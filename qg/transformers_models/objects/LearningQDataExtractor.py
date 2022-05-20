from zipfile import ZipFile
import torch
import json

class LearningQDataExtractorObject:
    def __init__(self, zipfile_name) -> None:
        self.zipfile_name = zipfile_name

    def extract_data(self, data_path, task) -> dict:
        """Extracts LearningQ data for training the 
        classifier or for transfer learning in QG from a zip file

        Args:
            zipfile_name (str): name of the zip file the data is extracted from
            data_path (str): path of the data to be extracted
        Returns:
            dictionary: dictionary with train, val, test splits
        """
        data = {}
        with ZipFile(self.zipfile_name) as zf:
            paths = [path for path in zf.namelist() if data_path in path and 'MACOSX' not in path]
            if task == "qg":
                paths = [path for path in zf.namelist() if data_path in path and path.endswith('.txt') and 'MACOSX' not in path]
            # accessing the data within each path
            # storing data in a suitable data structure
            for path in paths:

                if task == "classification":
                    file = path.split('/')[-1]
                    if file == "": # last value is ""
                        continue

                elif task == "qg":
                    source = 't-' if 'teded' in self.path else 'k-'
                    file = path.split('/')[-1]
                    file = file[:file.index('.')]
                    file = source+file if file.endswith('test') else file

                with zf.open(path) as f:
                    data[file] = f.read().decode().split('\n')
                    
        self.data = data
        return data

    def transform_classification_data(self):
        """Format to a dictionary with the data split as keys, and
        second keys a list of "questions" and "labels"

        Returns:
            data: transformed dictionary
        """
        formatted_data = {}
        formatted_data["train"] = {}
        formatted_data["test"] = {}
        formatted_data["val"] = {}
        formatted_data["train"]["text"] = []
        formatted_data["train"]["labels"] = []
        formatted_data["test"]["text"] = []
        formatted_data["test"]["labels"] = []
        formatted_data["val"]["text"] = []
        formatted_data["val"]["labels"] = []


        for key in self.data:
            for i, example in enumerate(self.data[key]):
                # the last i is an empty string
                if example == "":
                    continue
                
                label = example[0]
                question = example[2:]
                formatted_data[key]["text"].append(question)
                formatted_data[key]["labels"].append(int(label))

        # getting rid of val
        formatted_data["train"]["text"].extend(formatted_data["val"]["text"])
        formatted_data["train"]["labels"].extend(formatted_data["val"]["labels"])
        del formatted_data["val"]
                
        self.formatted_data = formatted_data
        return self

    def save_json_file(self, filename, data):
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)


# class Dataset(torch.utils.data.Dataset):
#     def __init__(self, dataset):

#         self.labels = dataset['labels']
#         self.texts = dataset['text']


if __name__=="__main__":
    
    extractor = LearningQDataExtractorObject(zipfile_name = "qg/LearningQ_data/LearningQ.zip")
    extractor.extract_data(data_path = "data/khan/khan_labeled_data", task = "classification")
    extractor.transform_classification_data()

    with open("cls_dataset.json", "w") as f:
        json.dump(extractor.formatted_data, f)
    print("done")
