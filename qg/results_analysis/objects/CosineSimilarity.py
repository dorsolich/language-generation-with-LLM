import spacy
import os
import  numpy as np
from tqdm import tqdm
from qg.config.config import get_logger, PACKAGE_ROOT
_logger = get_logger(logger_name=__file__)
try:
    nlp = spacy.load("en_core_web_lg")
except:
    os.system("python -m spacy download en_core_web_lg")
    nlp = spacy.load("en_core_web_lg")

class CosineSimilarityObject:
    def __init__(self, setting=None):
        self.setting = setting

    def filter_by_similarity(self, generated_questions: dict, target_questions: dict) -> tuple:
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

        if self.setting == "AQPL":

            for i in tqdm(target_questions):
                gen_predictions = generated_questions[i]

                # more that one question might have been generated in AQPL setting
                for prediction in gen_predictions:
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

        else:
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
