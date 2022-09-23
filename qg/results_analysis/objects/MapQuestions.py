import spacy
import os
import  numpy as np
from qg.config.config import get_logger, PACKAGE_ROOT
_logger = get_logger(logger_name=__file__)
try:
    nlp = spacy.load("en_core_web_lg")
except:
    os.system("python -m spacy download en_core_web_lg")
    nlp = spacy.load("en_core_web_lg")

class MapReferencesWithGenerateQuestions:
    def __init__(self, setting=None):
        self.setting = setting

    def map_references_with_predictions(self, 
                                        dataset, 
                                        source_texts:list, 
                                        ref_questions:list, 
                                        gen_questions:list) -> tuple:
        """ The target is to gather all the reference questions that happens in the raw dataset for each unique source text
            and to gather the model generated question for each unique source text.
            Then, place them in two dictionaries, that are mapped by the dictionary key
        Args:
            dataset (Dataset): Huggingface dataset
            source_texts (list): list of source texts output by qg_pipeline
            ref_questions (list): list of reference questions output by qg_pipeline
            gen_questions (list): list of generated questions output by qg_pipeline

        Returns:
            tuple of length 2: (generated_questions: dict, target_questions: dict)
            generated_questions = {
                0: "When was Benyonce born?",
                1: "What is the capital city of Colombia?"
                2: ...
                ...
                18836: ...
            }
            if setting == AQPL:
                generated_questions = {
                    0: ["When was Benyonce born?", "In what year was Beyonce born?"],
                    1: ["What is the capital city of Colombia?", "What is the population size of Colombia?"]
                    2: ...
                    ...
                    18836: ...
                }

            target_questions = {
                0: ["Where was Benyonce born?", "What is the name of Beyonce's first solo album?"]
                1: ["What is the capital city of Colombia?", "Where is the Amazonas river located?"]
                2: ...
                ...
                18836: ...
            }
        """
        target_questions = {}
        generated_questions = {}

        # two variables to inform
        questions_not_found = []
        answers_of_questions_not_found = []

        for i, example in enumerate(dataset):

            # accessing the raw context of the example
            try:
                # searching the raw context in the model output source_texts.txt
                index = source_texts.index(example["context"])

            except: 
                # sometimes, the raw context is not found because it has an extra space: "\n"
                # removing the extra space: "\n"
                example_context = example["context"].strip('\n')

                if example_context in source_texts:
                    index = source_texts.index(example_context)

                # the raw context might not be found even though "\n" is removed
                else:
                    questions_not_found.append(example_context)
                    answers_of_questions_not_found.append(example["answers"]["text"])
            
            # saving the target questions in the dictionary
            if index not in target_questions:
                target_questions[index] = []
            target_questions[index].append(example["question"])

            # saving the generated questions in the dictionary
            if example["question"] in ref_questions:
                idx = ref_questions.index(example["question"])
            if index not in generated_questions:
                generated_questions[index] = gen_questions[idx]

        empty_answers = [answer for answer in answers_of_questions_not_found if answer==[]]

        _logger.info(f"""
        There are {len(set(questions_not_found))} unique texts not processed and {len(target_questions.keys())} unique texts processed.
        From a total of {len(questions_not_found)} texts not processed, {len(empty_answers)} haven't been processed because they don't contain an answer.
        """)
        assert np.sum(generated_questions.keys()) == np.sum(target_questions.keys())

        # AQPL setting needs and extra preprocessing as many questions per line are generated.
        if self.setting == "AQPL":
            self.clean_AQPL(generated_questions, target_questions)

        else:
            self.generated_questions = generated_questions
            self.target_questions = target_questions

        return (self.generated_questions, self.target_questions)

    def clean_AQPL(self, generated_questions:dict, target_questions:dict):
        clean_gen_questions = {}
        
        total_questions = 0
        total_clean_questions = 0

        for key in generated_questions:
            
            gen_question = generated_questions[key]
            questions = []
            if " hl>" in gen_question:
                splitted_q = gen_question.split(" hl>")
                questions = [("").join(q) for q in splitted_q]
                clean_questions = [q for q in questions if len(q)>0 and q[-1] == "?"]
            else:
                clean_questions = [gen_question]
            clean_gen_questions[key] = clean_questions
            
            total_questions += len(questions)
            total_clean_questions += len(clean_questions)

        assert len(clean_gen_questions) == len(generated_questions)
        _logger.info(f"""
        A total of {total_questions} raw questions (not cleaned) have been generated by AQPL settings
        A total of {total_clean_questions} clean questions have been filtered in
        Example:
        Raw model output:
        {generated_questions[0]}
        Clean:
        {clean_gen_questions[0]}
        """)

        self.generated_questions = clean_gen_questions
        self.target_questions = target_questions

        return self