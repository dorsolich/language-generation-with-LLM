from qg.transformers_models.objects.LearningQDataExtractor import LearningQDataExtractorObject

def upload_cls_data():
    extractor = LearningQDataExtractorObject(zipfile_name = "qg/LearningQ_data/LearningQ.zip")
    extractor.extract_data(data_path = "data/khan/khan_labeled_data", task = "classification")
    extractor.transform_classification_data()
    return extractor.formatted_data

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