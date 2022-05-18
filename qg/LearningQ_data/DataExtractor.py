from zipfile import ZipFile
import json

class LearningQDataExtractor:
    def __init__(self, zipfile_name) -> None:
        self.zipfile_name = zipfile_name

    def extract_data(self, data_path, task) -> dict:
        """Extracts the data needed for training the 
        classifier, from a zip file

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
                self.path = path

                if task == "classification":
                    self._classification()
                elif task == "qg":
                    self._qg()

                if self.extract:
                    with zf.open(self.path) as f:
                        data[self.file] = f.read().decode().split('\n')
        self.data = data
        return data
    
    def _classification(self):
        self.extract = True
        self.file = self.path.split('/')[-1]
        if self.file == "": # last value is ""
            self.extract = False
        return self

    def _qg(self):
        self.extract = True
        source = 't-' if 'teded' in self.path else 'k-'
        file = self.path.split('/')[-1]
        file = file[:file.index('.')]
        file = source+file if file.endswith('test') else file
        self.file = file
        return self


    def transform_classification_data(self):
        """Format to a dictionary with the data split as keys, and
        second keys a list of "questions" and "labels"

        Returns:
            data: transformed dictionary
        """
        formatted_data = {}
        formatted_data["train"] = {}
        formatted_data["val"] = {}
        formatted_data["test"] = {}

        for key in self.data:
            for i, example in enumerate(self.data[key]):
                if i == 0:
                    formatted_data[key]["questions"] = []
                    formatted_data[key]["labels"] = []
                if example != "":
                    label = example[0]
                    question = example[2:]
                    formatted_data[key]["questions"].append(question)
                    formatted_data[key]["labels"].append(int(label))
        self.formatted_data = formatted_data 
        return self

    def save_json_file(self, filename, data):
        with open(filename, "w") as f:
            json.dump(data, f)
