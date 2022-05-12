class DataProcessor:
    def __init__(self, sep_token, eos_token) -> None:
        self.sep_token = sep_token
        self.eos_token = eos_token

    def process(self, dataset):
        dataset = dataset.map(self._add_eos_examples)
        return dataset

    def _add_eos_examples(self, example):
        answers = " "
        for i, answer in enumerate(example["answers"]["text"]):
            if i == 0:
                answers = self.sep_token + answer + self.sep_token
            else:
                answers += answer + self.sep_token

        example["context"] = answers + example["context"] + self.eos_token
        example["question"] = example["question"] + self.eos_token
        return example
