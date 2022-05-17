
from qg.config.config import get_logger
_logger = get_logger(logger_name=__name__)
class DataProcessor:
    def __init__(self, sep_token, eos_token, setting = "") -> None:
        self.sep_token = sep_token
        self.eos_token = eos_token
        self.setting = setting

        self.prev_context = ""
        self.prev_questions = ""
        self.index = 0
        self.relevant_examples_indices = []

    def process(self, dataset):

        if self.setting == "AQPL":
            dataset = dataset.map(self._all_questions_per_line)

        elif self.setting == "OQPL":
            dataset = dataset.map(self._one_question_per_line)

        elif self.setting == "AA": # answer awareness
            dataset = dataset.map(self._answer_awareness)

        dataset = dataset.map(self._eos_token)
        _logger.info(self.relevant_examples_indices)
        return dataset

    def _eos_token(self, example):
        example["context"] = example["context"] + self.eos_token
        example["question"] = example["question"] + self.eos_token
        return example

    def _one_question_per_line(self, example):
        example["context"] = example["context"] + self.eos_token + example["question"]
        return example

    def _all_questions_per_line(self, example):
        if example["context"] == self.prev_context:
            self.prev_questions += example["question"] + self.sep_token
            example["question"] = self.prev_questions
            self.prev_context = example["context"]
            self.index += 1

        else:
            self.prev_questions = example["question"] + self.sep_token
            self.prev_context = example["context"]
            if self.index != 0:
                self.relevant_examples_indices.append(self.index-1)
            self.index = self.index + 1

        return example

    def _answer_awareness(self, example):
        init_answer_token = " [ANSS] "
        end_answer_token = " [ANSE] "
        
        i_init = example["answers"]["answer_start"][0]
        i_end = i_init + len(init_answer_token) + len(example["answers"]["text"])

        example["context"] = example["context"][:i_init] + init_answer_token + example[i_init:]
        example["context"] = example["context"][:i_end] + end_answer_token + example[i_end:]
        return example

    def filter_examples(self, processed_train_data):
        length = len(processed_train_data)
        if self.setting == "AQPL":
                processed_train_data = processed_train_data.select(self.relevant_examples_indices)
                _logger.info(f"""Filtered in {len(self.relevant_examples_indices)}, 
                filtered out {length - len(self.relevant_examples_indices)}""")
        return processed_train_data