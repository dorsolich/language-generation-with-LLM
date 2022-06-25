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
    def __init__(self) -> None:
        pass

    def map_references_with_predictions(self, 
                                        dataset, 
                                        source_texts:list, 
                                        ref_questions:list, 
                                        gen_questions:list) -> tuple:
        """ The target is to gather all the reference questions that happens in the raw dataset for each unique source text
            and to gather the model generated question for each unique source text.
            Then, place them in two dictionaries, that are mapped by the dictionary key
        Args:
            dataset (_type_): Huggingface dataset
            source_texts (_type_): list of source texts output by qg_pipeline
            ref_questions (_type_): list of reference questions output by qg_pipeline
            gen_questions (_type_): list of generated questions output by qg_pipeline

        Returns:
            tuple of length 2: (generated_questions: dict, target_questions: dict)
            generated_questions = {
                0: "When was Benyonce born?",
                1: "What is the capital city of Colombia?"
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
        questions_not_found = []
        answers_of_questions_not_found = []
        for i, example in enumerate(dataset):
            try:
                index = source_texts.index(example["context"])
            except: 
                example_context = example["context"].strip('\n')
                if example_context in source_texts:
                    index = source_texts.index(example_context)
                else:
                    questions_not_found.append(example_context)
                    answers_of_questions_not_found.append(example["answers"]["text"])
            
            if index not in target_questions:
                target_questions[index] = []
            target_questions[index].append(example["question"])

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

        self.generated_questions = generated_questions
        self.target_questions = target_questions
        return (generated_questions, target_questions)


    def select_references_by_similarity(self, generated_questions: dict, target_questions: dict) -> tuple:
        """given one generated question and a list of target questions,
        it calculates the similarity between the generated questions and each target question,
        and selects the generated question with the highest similarity.

        Args:
            generated_questions (dict): {
                0: "When was Benyonce born?",
                1: "What is the capital city of Colombia?"
                2: ...
                ...
                18836: ...
            }
            target_questions (dict): {
                0: ["Where was Benyonce born?", "What is the name of Beyonce's first solo album?"]
                1: ["What is the capital city of Colombia?", "Where is the Amazonas river located?"]
                2: ...
                ...
                18836: ...
            }

        Returns:
            _type_: _description_
        """
        _logger.info(f"Calculating best similarity between generated questions and references...")

        predictions = []
        references = []

        
        examples_similarity = []

        for i in target_questions:
            prediction = generated_questions[i]
            predictions.append(prediction)

            sim_prediction = nlp(prediction)
            best_similarity = 0

            for target_q in target_questions[i]:
                sim_target = nlp(target_q)
                similarity = sim_prediction.similarity(sim_target)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_reference = target_q

            references.append(best_reference)
            examples_similarity.append(best_similarity)

        data = {}
        data["predictions"] = predictions
        data["references"] = references

        scores = {}
        scores["examples_similarity"] = examples_similarity
        scores["average_similarity"] = np.mean(examples_similarity)
            
        return (data, scores)
