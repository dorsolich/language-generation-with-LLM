# How to run the project.
1. First you train and validate the classifier (Two pipelines will be run).
2. Secondly you train a qg model and decode the training a validation questions for each experiment (Training pipeline will be run four times, decoder pipeline will be run two times per experiment, one time for decoding training questions and one time for decoding validation questions).
3. Thirdly, you validate the qg model (one pipeline)
4. Lastly, you classify questions (one pipeline)
5. You can either run a shortcut of the process or run the hole process to obtain the same results.

### RUNNING A SORTCUT (Recommended, although the model performance will not be good)
To run a shortcut of the training QG process for each experiment, run the following in order:

# Classifier
1.	CLASSIFIER: estimated total time: 1 min
To train the classifier, run the following command, where the learning_rate is a hyperparameter that you can modify:

- python -m qg.transformers_models.cls_training_pipeline --test TRUE --n_epochs 1 --learning_rate 2e-5

To validate the classifer model, run the following command, and make sure that the result_folder is named correctly depending of the learning rate:

- python -m qg.transformers_models.cls_validation_pipeline --test TRUE --results_folder classifier_2e-05


# Questions generation
##	QUESTION GENERATION: estimated total time: 4 mins

###### EXPERIMENT BASIC:
Training:

- python -m qg.transformers_models.qg_training_pipeline --test TRUE --n_epochs 1 --batch_size 2 --preprocess_setting basic

Decoding training questions:
- python -m qg.transformers_models.qg_decoder_pipeline --test TRUE --dataset_split train --results_folder experiment_basic

Decoding validation questions:
- python -m qg.transformers_models.qg_decoder_pipeline --test TRUE --dataset_split validation --results_folder experiment_basic

###### EXPERIMENT OQPL:
Training:

- python -m qg.transformers_models.qg_training_pipeline --test TRUE --n_epochs 1 --batch_size 2 --preprocess_setting OQPL

Decoding training questions:

- python -m qg.transformers_models.qg_decoder_pipeline --test TRUE --dataset_split train --results_folder experiment_OQPL

Decoding validation questions:

- python -m qg.transformers_models.qg_decoder_pipeline --test TRUE --dataset_split validation --results_folder experiment_OQPL
###### EXPERIMENT AA:
Training:

- python -m qg.transformers_models.qg_training_pipeline --test TRUE --n_epochs 1 --batch_size 2 --preprocess_setting AA

Decoding training questions:

- python -m qg.transformers_models.qg_decoder_pipeline --test TRUE --dataset_split train --results_folder experiment_AA

Decoding Validation Questions:

- python -m qg.transformers_models.qg_decoder_pipeline --test TRUE --dataset_split validation --results_folder experiment_AA
###### EXPERIMENT AQPL
AQPL setting storages a variable used during pre-processing in cache, and this will generate an error when this setting is run more than once. Please run AQPL setting only once.

Training:

- python -m qg.transformers_models.qg_training_pipeline --test TRUE --n_epochs 1 --batch_size 2 --preprocess_setting AQPL

Decoding training questions:

- python -m qg.transformers_models.qg_decoder_pipeline --test TRUE --dataset_split train --results_folder experiment_AQPL

Decoding validation questions:

- python -m qg.transformers_models.qg_decoder_pipeline --test TRUE --dataset_split validation --results_folder experiment_AQPL

##	VALIDATION OF QG MODELS

- python -m qg.results_analysis.questions_validation_pipeline

##	CLASSIFICATION OF QUESTIONS

- python -m qg.results_analysis.questions_classification_pipeline --classifier_folder classifier_2e-05 --preprocess_setting basic --dataset_split train


- python -m qg.results_analysis.questions_classification_pipeline --classifier_folder classifier_2e-05 --preprocess_setting basic --dataset_split validation


- python -m qg.results_analysis.questions_classification_pipeline --classifier_folder classifier_2e-05 --preprocess_setting AA --dataset_split train


- python -m qg.results_analysis.questions_classification_pipeline --classifier_folder classifier_2e-05 --preprocess_setting AA --dataset_split validation


- python -m qg.results_analysis.questions_classification_pipeline --classifier_folder classifier_2e-05 --preprocess_setting AQPL --dataset_split train


- python -m qg.results_analysis.questions_classification_pipeline --classifier_folder classifier_2e-05 --preprocess_setting AQPL --dataset_split validation


- python -m qg.results_analysis.questions_classification_pipeline --classifier_folder classifier_2e-05 --preprocess_setting OQPL --dataset_split train


- python -m qg.results_analysis.questions_classification_pipeline --classifier_folder classifier_2e-05 --preprocess_setting OQPL --dataset_split validation

### RUNNING ALL THE HOLE PROJECT WITHOUT SHORTCUT (It might take two days)

Run the same as above but change the argument --test TRUE to --test FALSE
