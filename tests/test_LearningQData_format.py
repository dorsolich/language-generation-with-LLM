from qg.LearningQ_data.LearningQDataExtractor import LearningQDataExtractorObject
import json
from qg.config.config import PACKAGE_ROOT

def upload_cls_data():
    try:
        extractor = LearningQDataExtractorObject(zipfile_name = "qg/LearningQ_data/LearningQ.zip")
        extractor.extract_data(data_path = "data/khan/khan_labeled_data", task = "classification")
        extractor.transform_classification_data()
        dataset = extractor.formatted_data
    except:
        with open(PACKAGE_ROOT/"qg"/"LearningQ_data"/"cls_dataset.json") as f:
            dataset = json.load(f)
    return dataset

def test_cls_data_format():
    x = upload_cls_data()
    assert set(x.keys()) == set(["train", "test"])
    for key in x:
        for id in x[key]:
            assert id in ["text", "labels"]
            if id == "text":
                for text in x[key][id]:
                    assert text != ""
        assert len(x[key]["text"]) == len(x[key]["labels"])
        assert set(x[key]["labels"]) == set([0,1])