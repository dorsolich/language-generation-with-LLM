def test_DataTrainingArguments(class_DataTrainingArguments):
    model_dic = class_DataTrainingArguments
    assert "task" in list(model_dic.keys())
    assert model_dic["model_type"] == "t5"