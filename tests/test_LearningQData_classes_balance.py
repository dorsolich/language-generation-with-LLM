from qg.config.config import PACKAGE_ROOT
import json

def upload_cls_data():
    with open(PACKAGE_ROOT/"qg"/"LearningQ_data"/"cls_balanced_dataset.json") as f:
        dataset = json.load(f)
    return dataset

def test_is_cls_dataset_balanced():
    dataset = upload_cls_data()
    assert dataset["train"]["labels"].count(0) == dataset["train"]["labels"].count(1)
    assert dataset["test"]["labels"].count(0) == dataset["test"]["labels"].count(1)

