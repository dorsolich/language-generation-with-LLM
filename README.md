# What is this project about

This project was developed to generate a model able to generate questions sintactically and gramatically correct, evaluating the model ability of generating _deep_ questions. Shall the model be able to generate _deep questions_ will prove that an **_smarter GenAI_** has been created, being able to extract logical inferences and to make logical reasoning.

In order to achieve (1) the generation of questions and (2) the evaluation of the model logical reasoning capability, two models have been trained and evaluated to extract conclussions:
1. A **Large Language Model (LLM)** was fine-tuned in the task of generating sintactically and gramatically correct questions from input text (Gen-IA). The LLM is the **Transformer T5**.

2. Then, these artifically generated questions were evaluated as usefull for human learning or not usefull for human learning. To classify the questions, a classifier model **DistilBERT** was fine-tunned in this particular task.

---
Additionally, four data-processing settings have been tested to understand which is the most effective method for processing the input text.

# Input Data


## Classification task with DistilBERT

**LearningQ** (Guanliang Chen and Houben, 2018) is a dataset for educational question generation.
The questions are gathered directly from two worldwide known online learning platforms: TED-Ed and Khan-Academy.
This dataset presents tremendous advantages to the educational question-generation problem:
1. It contains 5636 manually-labelled questions as either "useful for learning" or "not useful
for learning".
2. 30% of the questions are deep questions.
3. It includes document-question pairs over a variety of topics for model pre-training.
LearningQ would allow the model to learn a large variety of syntax, and to capture exhaustively
the differences between each class of questions. For this reason, LearningQ has been selected
as the dataset for training the Distil-BERT classifier.

## Language Generation task with Transforer T5
**Stanford Question Answering Dataset (SQuAD)** (Rajpurkar et al., 2016) contains 107,785
context-question-answer triples on 536 articles. It is the most widely used dataset in recent
years to train question generation models, which raises four very important points.
1. Allows applying the different data-processing settings.
2. It is primarily designed for reading comprehension purposes, and, as a secondary task,
the authors proposed its suitability for open-domain question answering. Although it is
not designed for question-generation, it is widely used for training question-generation
models (e.g., Du, Shao and Cardie, 2017; Wang et al., 2018; Chen, Wu and Zaki, 2019;
Chan and Fan, 2019; Hu and Liao, 2020; Lopez et al., 2021; Patil, 2020) because there
was a lack of specific educational question-generation datasets.
3. It allows accurate comparisons between models’ performance. Moreover, models’ performance is generally evaluated with similar metrics, such as BLEU or ROUGE.
4. A comparison between the questions in SQuAD and LearningQ showed that LearningQ
had questions whose length were almost twice than questions from SQuAD, had longer
documents and, although the questions were longer, it had fewer entities in the questions
(Kopp et al., 2017). Large sequences are difficult for a Transformer model to handle,
and although the questions can be truncated to a maximum length, large sequences can
be an inconvenience.
For these reasons, SQuAD dataset has been selected for training the T5 model over the four
data-processing settings.
### Context- Question processing settings:

The context (X) - question (y) pairs were preprocessed in four different ways and the model was train under each setting with the target of understanding which setting is the optimal and how much the data processing affects model performance.

- **Baseline**: is the most basic method that applies the least number of transformations to the input tada. It adds an `end-of-sentence` `[EOS]` token to indicate the end of each parragraph x<sub>i</sub> and each question y<sub>i</sub>. If the data contains a parragraph x<sub>i</sub> with more than one question y<sub>i</sub>, the parragraphs were duplicated and each duplication was liked to one possible training question.
- **All questions per line** (AQPL): In this setting, a training example consisted on x<sub>i</sub> and y<sub>i</sub>, where x<sub>i</sub> is the `context + [EOS]`, and y<sub>i</sub> has concatenated all the questions that can be generated from x<sub>i</sub>. The questions in y<sub>i</sub> were separated by a `separator token [SEP]`.
- **One question per line** (OQPL): In this case, the question was concatenated to the context input, including the `[EOS]` token between the context and the question.
- **Atention Awarenes** (AA): In this case, the answer is highlited in the parragraph x<sub>i</sub>, as context spam. In the context position right before the first answer word, a token `[ANSS]` was inserted. In the context position where the answer ends, a token `[ANSE]` was inserted.

# Training and Evaluation Pipelines

A pipeline contains a number of components that are functions that make some transformations to the data. The ouput of one component is the input to the next logical component. In general, each pipeline component apply one object to make the data tansformation. For example, the _trainer_ component is going to apply the _TrainerObject()_.

## Classification task with DistilBERT
1. **Training Pipeline** (_transformers_models/cls_training_pipeline.py_): 
    - loads the data
    - generates encodings
    - loads the pre-trained classifier
    - fine-tunes the classifier in the specific task of classifiying questions as _deep questions_.
2. **Validation Pipeline** (_transformers_models/cls_validation_pipeline.py_): 
    - loads the data
    - generates encodings
    - loads the fine-tuned classifier 
    - runs the encodings through the Classifier to classify the questions and returns its accuracy and other performance metrics.


## Language Generation with Transforer T5
1. **Training Pipeline** (_transformers_models/qg_training_pipeline.py_): 
    - loads and preprocess the data
    - generates encodings (transforms the human language into input word embeddings)
    - loads a pre-trained model
    - fine-tune the model on the specific task of question generation. Then, the fine-tuned model is saved.
2. **Decoder Pipeline** (_transformers_models/qg_decoder_pipeline.py_): 
    - loads the data
    - generates encodings
    - loads the fine-tuned Gen-AI model 
    - runs the encodings throuhg the Transformer model to generates decoded questions (transforms the output embeddings into human language).
3. **Validation Pipeline** (_results_analysis/questions_validation_pipeline_):
    - loads the generated questions and the _source_ questions
    - evaluate several aspects of the language generated such as its correcness syntactically and gramatically by applying several metrics. 
4. **Classification Pipeline** (_results_analysis/questions_validation_pipeline_):
    - loads the fine-tuned DistilBERT classifier
    - classifies each question as _deep questions_/_not deep question_-



# How to run the project
Steps:
1. First you train and validate the classifier (classification pipelines 1 and 2 to be run).
2. Secondly you train a language generation model and decode the model outputs (Gen-AI pipelines 1 and 2 to be run).
3. Thirdly, you validate the qg model (Gen-AI pipeline 3)
4. Lastly, you classify questions (Gen-AI pipeline 4)

You can either run a shortcut of the process or run the hole process to obtain the same results.

Open Bash terminal in the root directory of the project:
```
~/educational_questions_generation
```

Folders with the results will be automatically created under qg/transformers_models

# ENVIRONMENT SET-UP
Create and activate a new virtual environment:
```
conda create --name questions_env
conda activate questions_env
```
Install requirements:
```
pip install -r requirements.txt
```
To make sure the environment is correctly set-up, do in the terminal
```
cd tests
pytest
```
The command pytest will run three tests: one tests that you correctly have the cleaned and balanced LearningQ dataset (should cls_balaced_dataset.json be located under qg/LearningQ_data). Other two tests checks the version of transformers and pytorch

If you pass the three tests, move up again to the root directory doing:
```
cd ..
```

# RUNNING A SHORTCUT (Recommended for testing the code, although the model performance will not be good)
To run a shortcut of the training QG process for each experiment, make sure that the argument `--test` is set to `TRUE` and run the following in order:

## Classifier
1.	Estimated training time: 1 min.


To train the classifier, run the following command. The `learning_rate` is a hyperparameter that you can modify:
```
python -m qg.transformers_models.cls_training_pipeline --test TRUE --n_epochs 1 --learning_rate 2e-5
```
To validate the classifier model, run the following command, and make sure that the `result_folder` is named correctly depending on the `learning rate`:
```
python -m qg.transformers_models.cls_validation_pipeline --test TRUE --results_folder classifier_2e-05
```

##	QUESTION GENERATION: estimated total time: 4 mins

###### EXPERIMENT BASIC:
Training:
```
python -m qg.transformers_models.qg_training_pipeline --test TRUE --n_epochs 1 --batch_size 2 --preprocess_setting basic
```
Decoding training questions:

```
python -m qg.transformers_models.qg_decoder_pipeline --test TRUE --dataset_split train --results_folder experiment_basic
```
Decoding validation questions:
```
python -m qg.transformers_models.qg_decoder_pipeline --test TRUE --dataset_split validation --results_folder experiment_basic
```
###### EXPERIMENT OQPL:
Training:
```
python -m qg.transformers_models.qg_training_pipeline --test TRUE --n_epochs 1 --batch_size 2 --preprocess_setting OQPL
```
Decoding training questions:
```
python -m qg.transformers_models.qg_decoder_pipeline --test TRUE --dataset_split train --results_folder experiment_OQPL
```
Decoding validation questions:
```
python -m qg.transformers_models.qg_decoder_pipeline --test TRUE --dataset_split validation --results_folder experiment_OQPL
```
###### EXPERIMENT AA:
Training:
```
python -m qg.transformers_models.qg_training_pipeline --test TRUE --n_epochs 1 --batch_size 2 --preprocess_setting AA
```
Decoding training questions:
```
python -m qg.transformers_models.qg_decoder_pipeline --test TRUE --dataset_split train --results_folder experiment_AA
```
Decoding Validation Questions:
```
python -m qg.transformers_models.qg_decoder_pipeline --test TRUE --dataset_split validation --results_folder experiment_AA
```
###### EXPERIMENT AQPL

AQPL setting storages a variable used during pre-processing in cache, and this will generate an error when this setting is run more than once. Please run AQPL setting only once.

Training:
```
python -m qg.transformers_models.qg_training_pipeline --test TRUE --n_epochs 1 --batch_size 2 --preprocess_setting AQPL
```
Decoding training questions:
```
python -m qg.transformers_models.qg_decoder_pipeline --test TRUE --dataset_split train --results_folder experiment_AQPL
```
Decoding validation questions:
```
python -m qg.transformers_models.qg_decoder_pipeline --test TRUE --dataset_split validation --results_folder experiment_AQPL
```
##	VALIDATION OF QG MODELS (this takes quite a lot of minutes as runs through all the questions generated...)
```
python -m qg.results_analysis.questions_validation_pipeline
```
##	CLASSIFICATION OF QUESTIONS
```
python -m qg.results_analysis.questions_classification_pipeline --classifier_folder classifier_2e-05 --preprocess_setting basic --dataset_split train
```
```
python -m qg.results_analysis.questions_classification_pipeline --classifier_folder classifier_2e-05 --preprocess_setting basic --dataset_split validation
```
```
python -m qg.results_analysis.questions_classification_pipeline --classifier_folder classifier_2e-05 --preprocess_setting AA --dataset_split train
```
```
python -m qg.results_analysis.questions_classification_pipeline --classifier_folder classifier_2e-05 --preprocess_setting AA --dataset_split validation
```
```
python -m qg.results_analysis.questions_classification_pipeline --classifier_folder classifier_2e-05 --preprocess_setting AQPL --dataset_split train
```
```
python -m qg.results_analysis.questions_classification_pipeline --classifier_folder classifier_2e-05 --preprocess_setting AQPL --dataset_split validation
```
```
python -m qg.results_analysis.questions_classification_pipeline --classifier_folder classifier_2e-05 --preprocess_setting OQPL --dataset_split train
```
```
python -m qg.results_analysis.questions_classification_pipeline --classifier_folder classifier_2e-05 --preprocess_setting OQPL --dataset_split validation
```
### RUNNING ALL THE HOLE PROJECT WITHOUT SHORTCUT (It might take very long...)

Run the same as above but change the argument `--test TRUE` to `--test FALSE`
