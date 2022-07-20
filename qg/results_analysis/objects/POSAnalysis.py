# import sys
# import pathlib
# PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
# sys.path.append(str(PACKAGE_ROOT))

import pandas as pd
import pathlib
import spacy
import os
import json
from qg.config.config import get_logger, PACKAGE_ROOT
_logger = get_logger(logger_name=__file__)
from tqdm import tqdm
try:
    nlp = spacy.load("en_core_web_lg")
except:
    os.system("python -m spacy download en_core_web_lg")
    nlp = spacy.load("en_core_web_lg")
list_stopwords = nlp.Defaults.stop_words

class POS_analysis_object:
    def __init__(self) -> None:
        
        self.concepts_lemma = []
        self.concepts_string = []
        self.tracking_index = []

    def nlp_pipeline(self, texts):
        return list(nlp.pipe(texts))

    def extract_pos_concepts(self, nlp_sentence, filter_pos:list = ["NOUN", "ADJ", "PROPN"], split_in_documents = False):
        """
        """
        self.filter_pos = filter_pos

        if split_in_documents:
            self.concepts_lemma = []
            self.concepts_string = []
            self.tracking_index = []

        for token in nlp_sentence:
            
            if (token.is_alpha and not token.is_stop and
                token.pos_ in self.filter_pos):

                ## the following if statement manage to match words that belong to the same concept. 
                ## e.g: ["software", "development"] -> "software development"

                # only for the first concept
                if self.tracking_index == []:
                    self.concepts_lemma.append([token.lemma_])
                    self.concepts_string.append([token.text])
                    self.tracking_index.append([token.i])

                # checking if the word (NOUN or ADJ) matches with the previous word (NOUN or ADJ)
                elif self.tracking_index[-1][-1] == token.i-1:
                    # managing concept lemma
                    self.concepts_lemma[-1].append(token.lemma_)
                    updated_concept_lemma = (" ").join(self.concepts_lemma[-1])
                    self.concepts_lemma[-1] = [updated_concept_lemma]

                    # managing concept string
                    self.concepts_string[-1].append(token.text)
                    updated_concept_string = (" ").join(self.concepts_string[-1])
                    self.concepts_string[-1] = [updated_concept_string]

                    # managing concept position in the context (index)
                    self.tracking_index[-1].append(token.i)

                else: # initializing new concept
                    self.concepts_lemma.append([token.lemma_])
                    self.concepts_string.append([token.text])
                    self.tracking_index.append([token.i])

        self.all_concepts_string = [concept[0] for concept in self.concepts_string]
        self.all_concepts_lemma = [concept[0] for concept in self.concepts_lemma]

        return self


if __name__=="__main__":
    print(f"Root directory: {PACKAGE_ROOT}")
    DATA_DIR_AA = PACKAGE_ROOT/'qg'/'transformers_models'/'t5small_batch32_AA'
    DATA_DIR_AQPL = PACKAGE_ROOT/'qg'/'transformers_models'/'t5small_batch32_AQPL'
    DATA_DIR_BASIC = PACKAGE_ROOT/'qg'/'transformers_models'/'t5small_batch32_basic'
    DATA_DIR_OQPL = PACKAGE_ROOT/'qg'/'transformers_models'/'t5small_batch32_OQPL'

    results_folders = ["AA", "AQPL", "basic", "OQPL"]
    splits = ["train", "validation"]
    results = {}

    for split in splits:
        for folder in results_folders:

            # Uploading generated questions...
            with open(PACKAGE_ROOT/f"qg/transformers_models/t5small_batch32_{folder}/mapped_{split}_questions.json", encoding="utf-8") as f:
                questions = json.load(f)
                predictions = questions["predictions"]
                references = questions["references"]

                sets = [predictions, references]
                sets_names = ["predictions", "references"]

                for s_i,(set, name) in enumerate(zip(sets, sets_names)):

                    _logger.info(f"Generating NLP pipeline with '{name} {folder}' questions...")
                    pos_analysis = POS_analysis_object()
                    questions_pipeline = pos_analysis.nlp_pipeline(set)

                    _logger.info(f"Pipeline generated. Extracting concepts...")
                    strings = []
                    lemmas = []

                    for question in tqdm(questions_pipeline):
                        pos_analysis.extract_pos_concepts(question, split_in_documents=True)
                        strings.append(pos_analysis.all_concepts_string)
                        lemmas.append(pos_analysis.all_concepts_lemma)

                    concepts = {}
                    concepts["strings"] = strings
                    concepts["lemmas"]  = lemmas
                    results[f"{folder}_{split}_{name}"] = concepts
                
                    _logger.info(f"Concepts from {folder}_{split}_{name} extracted.")

    RESULTS_DIR = PACKAGE_ROOT/"qg"/"results_analysis"
    file_name = "concepts_ref_gen_questions.json"
    PATH = os.path.join(RESULTS_DIR, file_name)
    

    with open (PATH,"w+", encoding='utf-8') as f:
        json.dump(results,f, indent=4)
    _logger.info(f'Task finished. Results saved in path: {PATH}')