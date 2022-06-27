import spacy
import os
try:
    nlp = spacy.load("en_core_web_lg")
except:
    os.system("python -m spacy download en_core_web_lg")
    nlp = spacy.load("en_core_web_lg")

# cumulative BLEU scores
from nltk.translate.bleu_score import sentence_bleu

class ComputeBleuObject:

    def __init__(self) -> None:
        pass

    def compute_scores(self, references, predictions, setting="tokens"):
        # initializing variables...
        self.bleu_1 = []
        self.bleu_2 = []
        self.bleu_3 = []
        self.bleu_4 = []

        for ref, cand in zip(references, predictions):
            # creating spaCy nlp pipeline to get tokens of lemmas...
            ref_doc = nlp(ref)
            cand_doc = nlp(cand)

            if setting == "tokens":
                reference = [[token.text for token in ref_doc]]
                candidate = [token.text for token in cand_doc]

            elif setting =="lemmas":
                reference = [[token.lemma_ for token in ref_doc]]
                candidate = [token.lemma_ for token in cand_doc]
                
            else:
                raise Exception("Please select as setting either 'raw_tokens' or 'lemmas'")

            # https://www.nltk.org/_modules/nltk/translate/bleu_score.html
            example_bleu_1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
            example_bleu_2 = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))
            example_bleu_3 = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0))
            example_bleu_4 = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))

            self.bleu_1.append(example_bleu_1)
            self.bleu_2.append(example_bleu_2)
            self.bleu_3.append(example_bleu_3)
            self.bleu_4.append(example_bleu_4)

        return self