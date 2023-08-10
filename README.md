# Named-Entity Recognition for Slovenian, Croatian and Serbian

An evaluation of various encoder Transformer-based large language models on the named entity recognition task. The models are compared on 6 datasets, manually-annotated with named entitites:
 - standard Slovene (separated into 3 datasets)
 - non-standard Slovene
 - standard Croatian
 - non-standard Croatian
 - standard Serbian
 - non-standard Serbian

## Dataset preparation

To download the datasets from the CLARIN.SI repository and prepare JSON files which will be used as train, dev and test files for classification with the simpletransformers library, run the following command in the command line:

```
bash prepare_datasets.sh "s_Slovene" "ns_Slovene" "s_Croatian" "ns_Croatian" "s_Serbian" "ns_Serbian" > dataset_preparation.log
```

You can use all available datasets or define just a couple of them as the arguments (e.g., if you want to download only standard and non-standard Serbian: "s_Serbian" "ns_Serbian" )

Extracted JSON files are dictionaries which consist of the following keys:
 - "labels" (list of NE labels used in the dataset)
 - "train", "dev", "test" (dataset splits - in case of Croatian and Serbian datasets)
 - "dataset" (in case of Slovenian datasets which are not separated into splits)

"train", "dev", "test" and "dataset" are also dictionaries, with the following keys:
 - "sentence_id" (original sentence id)
 - "words" (word forms)
 - "labels" (NE labels)

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

```
