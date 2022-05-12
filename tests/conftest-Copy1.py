import pytest

from question_generation.data_preparation.prep_data import DataTrainingArguments


@pytest.fixture
def class_DataTrainingArguments():
    model = DataTrainingArguments("e2e_qg", "t5")
    return model.__dict__