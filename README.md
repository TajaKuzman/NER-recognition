# Named-Entity Recognition for Slovenian, Croatian and Serbian

An evaluation of various encoder Transformer-based large language models on the named entity recognition task. The models are compared on 6 datasets, manually-annotated with named entitites:
 - standard Slovene (separated into 3 datasets)
 - non-standard Slovene
 - standard Croatian
 - non-standard Croatian
 - standard Serbian
 - non-standard Serbian

**Dataset sizes**

(Sizes in no. of words - instances)

| elexiswsd (SL_s) | /senticoref (SL_s) | hr500k (HR_s) | reldi-normtagner-hr (HR_ns) | reldi-normtagner-sr (SR_ns) | set.sr.plus (SR_s) | ssj500k (SL_s) | Janes-Tag (SL_ns) |
|------------------|--------------------|---------------|-----------------------------|-----------------------------|--------------------|----------------|-------------------|
| 31,233           | 391,962            | 499,635       | 89,855                      | 97,673                      | 92,271             | 235,864        | 75,926            |

## Results

**Base-sized models**

(Each model was trained and tested three times, these results are the means of three runs.)




**Large-sized models**

(Each model was trained and tested three times, these results are the means of three runs.)


## Dataset preparation

To download the datasets from the CLARIN.SI repository and prepare JSON files which will be used as train, dev and test files for classification with the simpletransformers library, run the following command in the command line:

```
bash prepare_datasets.sh "s_Slovene" "ns_Slovene" "s_Croatian" "ns_Croatian" "s_Serbian" "ns_Serbian" > dataset_preparation.log
```

You can use all available datasets or define just a couple of them as the arguments (e.g., if you want to download only standard and non-standard Serbian: "s_Serbian" "ns_Serbian" )

Extracted JSON files are dictionaries which consist of the following keys:
 - "labels" (list of NE labels used in the dataset)
 - "train", "dev", "test" (dataset splits)

"train", "dev", "test" and "dataset" are also dictionaries, with the following keys:
 - "sentence_id" (original sentence id)
 - "words" (word forms)
 - "labels" (NE labels)

Croatian and Serbian datasets already have information on splits, so we split them according to the original splits. The Slovenian datasets are originally not split into train, dev and test subset - we split them in 80:10:10 train-dev-test ratio according to doc ids (we extract doc ids and randomly split them into train-dev-test). The Slovenian Elexis WSD dataset does not have doc ids (and sentences are not connected), so we split them according to sentence ids.

To use them for classification with the simpletransformers library:
```
import json
import pandas as pd

# Define the path to the dataset
dataset_path = "datasets/set.sr.plus.conllup_extracted.json"

# Load the json file
with open(dataset_path, "r") as file:
    json_dict = json.load(file)

# Open the train, eval and test dictionaries as DataFrames
train_df = pd.DataFrame(json_dict["train"])
test_df = pd.DataFrame(json_dict["test"])
dev_df = pd.DataFrame(json_dict["dev"])

# Change the sentence_ids to integers (!! important - otherwise, the models do not work)
test_df['sentence_id'] = pd.factorize(test_df['sentence_id'])[0]
train_df['sentence_id'] = pd.factorize(train_df['sentence_id'])[0]
dev_df['sentence_id'] = pd.factorize(dev_df['sentence_id'])[0]

# Define the labels
LABELS = json_dict["labels"]
print(LABELS)

print(train_df.shape, test_df.shape, dev_df.shape)
print(train_df.head())

```

## Hyperparameter search

We did hyperparameter search for the following models:
- XLM-R-base - hyperparameters to be used for XLM-R-b-BCMS models as well
- XLM-R-large - - hyperparameters to be used for XLM-R-l-BCMS and XLM-R-l-SL-BCMS models as well
- BERTić
- CSEBERT

For each model, we do 3 searches:
- one on hr500k: to be used for hr500k
- one on reldi-hr: to be used for reldi-hr, reldi-sr
- one on set.sr: to be used for set.sr

(For later experiments on SL data, we will do separate hyperparameter searches for these models and SloBERTa on ssj500k and Janes-Tag.)

I searched for the optimum no. of epochs, while we set the other hyperparameters to these values:

```
model_args ={"overwrite_output_dir": True,
             "labels_list": LABELS,
             "learning_rate": 4e-05,
             "train_batch_size": 32,
             "no_cache": True,
             "no_save": True,
             "max_seq_length": 256,
             "save_steps": -1,
            # Use these parameters if you want to evaluate during training
            "evaluate_during_training": True,
            ## Calculate how many steps will each epoch have
            # num steps in epoch = training samples / batch size
            # Then evaluate after every 3rd epoch
            "evaluate_during_training_steps": len(train_df.words)/32*3,
            "evaluate_during_training_verbose": True,
            "use_cached_eval_features": True,
            'reprocess_input_data': True,
}
```

Code:
- search on all datasets with all base models: ``` CUDA_VISIBLE_DEVICES=4 nohup bash hyperparameter-search-base.sh > hyperparameter-search-pipeline.log &```
- search on all datasets with all large models: ```CUDA_VISIBLE_DEVICES=3 nohup bash hyperparameter-search-large.sh > hyperparameter-search-pipeline-large.log &```

I searched for the optimum no. of epochs by training the model for 15 epochs and then evaluating during training. Then I inspected how the evaluation loss falls during training (when F1 plateaus and evaluation loss starts rising).

### Hyperparameters used 

We use the following hyperparameters for all models and change only the no. of epochs (`num_train_epochs`):

```
    model_args = {"overwrite_output_dir": True,
                "num_train_epochs": 5,
                "labels_list": LABELS,
                "learning_rate": 4e-05,
                "train_batch_size": 32,
                # Comment out no_cache and no_save if you want to save the model
                "no_cache": True,
                "no_save": True,
                "max_seq_length": 256,
                "save_steps": -1,
                "silent": True,
                }
```

Number of epochs:

| model | hr500k (HR_s) | reldi-normtagner-hr (HR_ns) | set.sr (SR_s) |
|---|---|---|---|
| xlm-r-b; xlm-r-b-bcms | 5 | 8 | 6 |
| xlm-r-l; xlm-r-l-bcms; xlm-r-l-si-bcms | 7 | 11 | 13 |
| bertic | 9 | 6 | 8 |
| csebert | 4 | 7 | 9 |


## Model evaluation

We run each models 3 times. I run the code for public and custom models separately and also run base and large models separately. I run each code 2 times, and change in the code whether to use base or large models:
- public models (change models to run in model_list on line 198): ```CUDA_VISIBLE_DEVICES=7 nohup python ner-classification.py > ner_classification_base.log &```
- custom models (change which models to run in lines 227-229): ```CUDA_VISIBLE_DEVICES=6 nohup python ner-classification-with-custom-models.py > ner_classification_custom_base.log &```

The outputs are saved as:
- *ner-results.txt* and *ner-results-custom.txt*: files with all results from all runs (including per label results)
- *ner-results-summary-table.csv* and *ner-results-summary-table-our-models.csv*: tables with summarized results (just Macro F1s for each dataset for each model)
- *datasets/hr500k.conllup_extracted.json-test_df-with-predictions.csv* (dataset path + -test_df-with-predictions.csv): test splits with y_pred values from all runs
- confusion matrices are saved in the *figures/cm-datasets* folder