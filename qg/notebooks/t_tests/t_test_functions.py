from pickletools import read_uint1
import numpy as np
import scipy.stats as stats
from qg.results_analysis.objects.POSAnalysis import POS_analysis_object
from tqdm import tqdm
from collections import Counter

def count_tokens(text):
        return len(text.split())

def diff_number_words_per_question(sets: list, sets_name: list):
    """Are there statistically significant differences in the number of words in the questions of each group"""

    word_counts = {}
    for set, name in zip(sets, sets_name):
        n_words = []
        for question in set["text"].values:
            n_words.append(count_tokens(question))

        word_counts[name] = n_words

    fvalue, pvalue = stats.f_oneway(word_counts["useful"], word_counts["not_useful"])

    if pvalue < 0.05:
        print("Yes, there are statistically significant differences in the length of the questions between the two groups")
    else:
        print("No, there are not statistically significant differences in the length of the questions between the two groups")
    print(f"   P-value = {pvalue}")
    print(f"   Average length of useful questions {np.mean(word_counts['useful'])} tokens")
    print(f"   Average length of not_useful questions {np.mean(word_counts['not_useful'])} tokens")

    return None


def diff_number_of_concepts_per_question(sets: list, sets_name: list):
    """Are there statistically significant differences in the number of concepts in the questions of each group?
    """

    concepts_count = {}
    strings_groups = {}
    for set, name in zip(sets, sets_name):
        pos_analysis = POS_analysis_object()
        questions_pipeline = pos_analysis.nlp_pipeline(set["text"].values)

        strings = []
        lemmas = []
        for question in tqdm(questions_pipeline):
            pos_analysis.extract_pos_concepts(question, split_in_documents=True)
            strings.append(pos_analysis.all_concepts_string)
            lemmas.append(pos_analysis.all_concepts_lemma)
        

        concepts_count[name] = [len(x) for x in strings]
        strings_groups[name] = strings

    fvalue, pvalue = stats.ttest_ind(concepts_count["useful"], concepts_count["not_useful"])

    if pvalue < 0.05:
        print("Yes, there are statistically significant differences in the number of concepts in each question between the two groups")

    else:
        print("No, there are not statistically significant differences in the number of concepts in each question between the two groups")

    print(f"   P-value = {pvalue}")
    print(f"   Average number of concepts in useful questions: {np.mean(concepts_count['useful'])}")
    print(f"   Average number of concepts in not useful questions: {np.mean(concepts_count['not_useful'])}")

    return strings_groups


def most_frequent_concepts(dict_groups:dict):
    all_strings_useful = [concept for question in dict_groups["useful"] for concept in question ]
    all_strings_not_useful = [concept for question in dict_groups["not_useful"] for concept in question ]

    cnt = Counter()
    cnt_useful = Counter(all_strings_useful)
    cnt_not_useful = Counter(all_strings_not_useful)

    cnt_useful = sorted(cnt_useful.items(), key=lambda item: item[1])
    cnt_not_useful = sorted(cnt_not_useful.items(), key=lambda item: item[1])

    cnt_useful.reverse()
    cnt_not_useful.reverse()
    return (cnt_useful, cnt_not_useful, all_strings_useful, all_strings_not_useful)


def diff_prop_concepts_per_words(sets: list, sets_name: list):
    """Are there statistically significant differences in the proportion of concepts per words in each question between both groups?"""
    concepts_proportions = {}
    for set, name in zip(sets, sets_name):
        pos_analysis = POS_analysis_object()
        questions_pipeline = pos_analysis.nlp_pipeline(set["text"].values)

        proportions = []
        for question in tqdm(questions_pipeline):
            pos_analysis.extract_pos_concepts(question, split_in_documents=True)
            proportions.append(len(pos_analysis.all_concepts_string)/len(question))

        concepts_proportions[name] = proportions

    fvalue, pvalue = stats.ttest_ind(concepts_proportions["useful"], concepts_proportions["not_useful"])

    if pvalue < 0.05:
        print("Yes, there are statistically significant differences in the proportion of concepts per word in each question between the two groups")

    else:
        print("No, there are not statistically significant differences HAVE NOT been found in the proportion of concepts per word in each question between the two groups")

    print(f"   P-value = {pvalue}")
    print(f"   Average proportion of concepts per words in useful questions: {np.mean(concepts_proportions['useful'])}")
    print(f"   Average proportion of concepts per words in not_useful questions: {np.mean(concepts_proportions['not_useful'])}")
    return None

def diff_number_of_verbs(sets: list, sets_name: list):
    """Are there statistically significant differences in the number of verbs in each question between both groups?"""
    verbs_count = {}
    verbs_groups = {}
    for set, name in zip(sets, sets_name):
        pos_analysis = POS_analysis_object()
        questions_pipeline = pos_analysis.nlp_pipeline(set["text"].values)

        verbs = []
        for question in tqdm(questions_pipeline):
            pos_analysis.extract_pos_concepts(question, split_in_documents=True, filter_pos=["VERB"])
            verbs.append(pos_analysis.all_concepts_string)

        verbs_count[name] = [len(x) for x in verbs]
        verbs_groups[name] = verbs

    fvalue, pvalue = stats.ttest_ind(verbs_count["useful"], verbs_count["not_useful"])

    if pvalue < 0.05:
        print("Yes, there are statistically significant differences in the number of VERBS in each question between the two groups")

    else:
        print("No, there are not statistically significant differences in the number of VERBS in each question between the two groups")

    print(f"   P-value = {pvalue}")
    print(f"   Average number of verbs in useful questions: {np.mean(verbs_count['useful'])} verbs")
    print(f"   Average number of verbs in not useful questions: {np.mean(verbs_count['not_useful'])} verbs")
    return verbs_groups

def diff_prop_of_verbs(sets: list, sets_name: list):
    """Are there statistically significant differences in the proportion of verbs per words in each question between both groups?"""
    verbs_proportions = {}
    for set, name in zip(sets, sets_name):
        pos_analysis = POS_analysis_object()
        questions_pipeline = pos_analysis.nlp_pipeline(set["text"].values)

        proportions = []
        for question in tqdm(questions_pipeline):
            pos_analysis.extract_pos_concepts(question, split_in_documents=True, filter_pos=["VERB"])
            proportions.append(len(pos_analysis.all_concepts_string)/len(question))

        verbs_proportions[name] = proportions

    fvalue, pvalue = stats.ttest_ind(verbs_proportions["useful"], verbs_proportions["not_useful"])

    if pvalue < 0.05:
        print("Yes, there are statistically significant differences in the proportion of VERBS per words in each question between the two groups")

    else:
        print("No, there are not Statistically significant differences in the proportion of VERBS per words in each question between the two groups")

    print(f"   P-value = {pvalue}")
    print(f"   Average proportion of verbs per words in useful questions: {np.mean(verbs_proportions['useful'])} verbs")
    print(f"   Average proportion of verbs per words in not useful questions: {np.mean(verbs_proportions['not_useful'])} verbs")
    return None