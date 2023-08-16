import pandas as pd
import json
from simpletransformers.ner import NERModel, NERArgs
from tqdm.autonotebook import tqdm as notebook_tqdm
import wandb
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import logging
import sklearn
import numpy as np
import argparse
import torch
from numba import cuda
import gc


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Login to wandb
wandb.login()

# Import the dataset

# Define the path to the dataset
dataset_path = "datasets/hr500k.conllup_extracted.json"

# Load the json file
with open(dataset_path, "r") as file:
    json_dict = json.load(file)

# Open the train, eval and test dictionaries as DataFrames
train_df = pd.DataFrame(json_dict["train"])
test_df = pd.DataFrame(json_dict["test"])
dev_df = pd.DataFrame(json_dict["dev"])

# Define the labels
LABELS = json_dict["labels"]

print("Label list")
print(LABELS)

print("\n\ntest_df:")
print(test_df.head())


epochs = 20

model_args = NERArgs()

# define hyperparameters
model_args ={"overwrite_output_dir": True,
             #"num_train_epochs": epochs,
             "labels_list": LABELS,
             "learning_rate": 1e-5,
             "train_batch_size": 32,
             # Comment out no_cache and no_save if you want to save the model
             "no_cache": True,
             "no_save": True,
            # Only the trained model will be saved (if you want to save it)
            # - to prevent filling all of the space
            # "save_model_every_epoch":False,
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
            "wandb_project": "NER",
            "silent": True,
            "wandb_kwargs": {"name": "default"},
             }

def train(model_name, train_df, dev_df):
    # Initialize a new wandb run
    wandb.init()

    model_type_dict = {
        "sloberta": ["camembert", "EMBEDDIA/sloberta"],
        "csebert": ["bert", "EMBEDDIA/crosloengual-bert"],
        "xlm-r-base": ["xlmroberta", "xlm-roberta-base"],
        "xlm-r-large": ["xlmroberta", "xlm-roberta-large"],
        "bertic": ["electra", "classla/bcms-bertic"]
    }

    # Create a model
    model = NERModel(
        model_type_dict[model_name][0],
        model_type_dict[model_name][1],
        use_cuda=True,
        args = model_args)
    
    # Train the model
    model.train_model(train_df, eval_df=dev_df)

    # Evaluate the model
    model.eval_model(dev_df)

    # Sync wandb
    wandb.join()

    gc.collect()
    torch.cuda.empty_cache()

# Choose the model - insert the model name as it is in the model_type_dict
model_name = "xlm-r-large"

# Run the training
print("Training and running evaluation during training, epochs: {}".format(epochs))

train(model_name, train_df, dev_df)