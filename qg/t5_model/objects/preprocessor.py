
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
        self.init_indices = []
        self.len_answers = 0

    def process(self, dataset):

        # Experiment 4: one question per line
        if self.setting == "OQPL": 
            self.check_dataset = dataset
            dataset = dataset.map(self._one_question_per_line)
        
        # Experiment 3: all questions per line
        elif self.setting == "AQPL": # all questions per line
            dataset = dataset.map(self._all_questions_per_line)

        # Experiment 2: answer awareness
        elif self.setting == "AA":
            dataset = dataset.map(self._answer_awareness)

        # Experiment 1: basic processing
        dataset = dataset.map(self._eos_token)

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
            self.index = self.index + 1

        else:
            self.prev_questions = example["question"] + self.sep_token
            self.prev_context = example["context"]
            if self.index != 0:
                self.relevant_examples_indices.append(self.index-1)
            self.index = self.index + 1

        return example

    def _answer_awareness(self, example):
        init_answer_token = "[ANSS] "
        end_answer_token = " [ANSE]"
        
        # Examples that don't contain answers
        if example["answers"]["answer_start"] == []:
            return example
        else:
            # Examples might contain more that one answer
            for i in range(len(example["answers"]["answer_start"])):
                
                if i == 0:
                    i_init = example["answers"]["answer_start"][i]
                else:
                    i_init = example["answers"]["answer_start"][i] + (len(init_answer_token)*i)

                str_context = str(example["context"])
                example["context"] = str_context[:i_init] + init_answer_token + str_context[i_init:]
                self.init_indices.append(i_init)

            for i in range(len(example["answers"]["answer_start"])):

                if i == 0:
                    i_end = self.init_indices[i] + len(init_answer_token) + len(example["answers"]["text"][0])
                    self.len_answers += len(example["answers"]["text"][0])
                else:
                    i_end = self.init_indices[i] + len(init_answer_token)*i+1 + self.len_answers

                str_context = str(example["context"])
                example["context"] = str_context[:i_end] + end_answer_token + str_context[i_end:]

            return example

    def filter_examples(self, processed_train_data):

        # only applicable in setting AQPL
        if self.setting == "AQPL":

            # due to cache memory, self.relevant_examples_indices can be an empty list
            # this if statement checks it and populates it if needed
            if self.relevant_examples_indices == []:
                self.check_dataset = self.check_dataset.map(self._one_question_per_line)
            
            processed_train_data = processed_train_data.select(self.relevant_examples_indices)
            _logger.info(f"""Filtered in {len(self.relevant_examples_indices)}, 
            filtered out {len(processed_train_data) - len(self.relevant_examples_indices)}""")
            
        return processed_train_data
