from qg.config.config import LEARNINGQ_DATA_DIR
import numpy as np
import json

def upload_cls_data():
    with open(LEARNINGQ_DATA_DIR/"LearningQ_classification.json") as f:
        x = json.load(f)
    return x

def test_cls_data_format():
    x = upload_cls_data()
    for key in x:
        assert set(np.unique(x[key]["labels"])) == set([0,1])
        assert len(x[key]["questions"]) == len(x[key]["labels"])