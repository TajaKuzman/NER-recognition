import evaluate
from datetime import datetime
import pandas as pd
import numpy as np
import json
from simpletransformers.ner import NERModel, NERArgs
from sklearn.metrics import classification_report, f1_score
from tqdm.autonotebook import tqdm as notebook_tqdm
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import logging
import sklearn
from numba import cuda
import argparse
import gc
import torch
import time

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Create lists of all needed models for the task
base_dict = {"/cache/nikolal/xlmrb_bcms_exp/checkpoint-12000": "xlmrb_bcms-12", "/cache/nikolal/xlmrb_bcms_exp/checkpoint-24000": "xlmrb_bcms-24", "/cache/nikolal/xlmrb_bcms_exp/checkpoint-36000": "xlmrb_bcms-36", "/cache/nikolal/xlmrb_bcms_exp/checkpoint-48000": "xlmrb_bcms-48", "/cache/nikolal/xlmrb_bcms_exp/checkpoint-60000": "xlmrb_bcms-60", "/cache/nikolal/xlmrb_bcms_exp/checkpoint-72000": "xlmrb_bcms-72", "/cache/nikolal/xlmrb_bcms_exp/checkpoint-84000": "xlmrb_bcms-84", "/cache/nikolal/xlmrb_bcms_exp/checkpoint-96000": "xlmrb_bcms-96"}
large_dict = {"/cache/nikolal/xlmrl_bcms_exp/checkpoint-6000": "xlmrl_bcms-6", "/cache/nikolal/xlmrl_bcms_exp/checkpoint-12000":"xlmrl_bcms-12", "/cache/nikolal/xlmrl_bcms_exp/checkpoint-18000": "xlmrl_bcms-18", "/cache/nikolal/xlmrl_bcms_exp/checkpoint-24000": "xlmrl_bcms-24", "/cache/nikolal/xlmrl_bcms_exp/checkpoint-30000": "xlmrl_bcms-30", "/cache/nikolal/xlmrl_bcms_exp/checkpoint-36000": "xlmrl_bcms-36", "/cache/nikolal/xlmrl_bcms_exp/checkpoint-42000": "xlmrl_bcms-42", "/cache/nikolal/xlmrl_bcms_exp/checkpoint-48000": "xlmrl_bcms-48", "/cache/nikolal/xlmrl_sl-bcms_exp/checkpoint-6000": "xlmrl_sl-bcms-6", "/cache/nikolal/xlmrl_sl-bcms_exp/checkpoint-12000": "xlmrl_sl-bcms-12", "/cache/nikolal/xlmrl_sl-bcms_exp/checkpoint-18000": "xlmrl_sl-bcms-18", "/cache/nikolal/xlmrl_sl-bcms_exp/checkpoint-24000": "xlmrl_sl-bcms-24", "/cache/nikolal/xlmrl_sl-bcms_exp/checkpoint-30000": "xlmrl_sl-bcms-30", "/cache/nikolal/xlmrl_sl-bcms_exp/checkpoint-42000": "xlmrl_sl-bcms-42", "/cache/nikolal/xlmrl_sl-bcms_exp/checkpoint-48000": "xlmrl_sl-bcms-48"}

base_list = list(base_dict.keys())
large_list = list(large_dict.keys())

# Import the dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="path to the dataset in JSON format")
    args = parser.parse_args()

# Define the path to the dataset
dataset_path = args.dataset

# Load the json file
with open(dataset_path, "r") as file:
    json_dict = json.load(file)

# Open the train, eval and test dictionaries as DataFrames
train_df = pd.DataFrame(json_dict["train"])
dev_df = pd.DataFrame(json_dict["dev"])

# Change the sentence_ids to numbers
train_df['sentence_id'] = pd.factorize(train_df['sentence_id'])[0]
dev_df['sentence_id'] = pd.factorize(dev_df['sentence_id'])[0]

# Define the labels
LABELS = json_dict["labels"]

def train_and_save_checkpoint(model_path, model_size, train_df, LABELS):
	# When fine-tuning our custom models that we pre-trained, and using them from checkpoints, the process is a bit different than with publicly available models: first, we need to fine-tune a model from the original checkpoint, so that we save the model and overwrite its original settings which force pretraining from a specific step (and disable fine-tuning by that). Then we take that new model and fine-tune it, as we did with the models before. 

	# Create lists of all needed models for the task
	path_list = {"/cache/nikolal/xlmrb_bcms_exp/checkpoint-12000": "xlmrb_bcms-12", "/cache/nikolal/xlmrb_bcms_exp/checkpoint-24000": "xlmrb_bcms-24", "/cache/nikolal/xlmrb_bcms_exp/checkpoint-36000": "xlmrb_bcms-36", "/cache/nikolal/xlmrb_bcms_exp/checkpoint-48000": "xlmrb_bcms-48", "/cache/nikolal/xlmrb_bcms_exp/checkpoint-60000": "xlmrb_bcms-60", "/cache/nikolal/xlmrb_bcms_exp/checkpoint-72000": "xlmrb_bcms-72", "/cache/nikolal/xlmrb_bcms_exp/checkpoint-84000": "xlmrb_bcms-84", "/cache/nikolal/xlmrb_bcms_exp/checkpoint-96000": "xlmrb_bcms-96", "/cache/nikolal/xlmrl_bcms_exp/checkpoint-6000": "xlmrl_bcms-6", "/cache/nikolal/xlmrl_bcms_exp/checkpoint-12000":"xlmrl_bcms-12", "/cache/nikolal/xlmrl_bcms_exp/checkpoint-18000": "xlmrl_bcms-18", "/cache/nikolal/xlmrl_bcms_exp/checkpoint-24000": "xlmrl_bcms-24", "/cache/nikolal/xlmrl_bcms_exp/checkpoint-30000": "xlmrl_bcms-30", "/cache/nikolal/xlmrl_bcms_exp/checkpoint-36000": "xlmrl_bcms-36", "/cache/nikolal/xlmrl_bcms_exp/checkpoint-42000": "xlmrl_bcms-42", "/cache/nikolal/xlmrl_bcms_exp/checkpoint-48000": "xlmrl_bcms-48", "/cache/nikolal/xlmrl_sl-bcms_exp/checkpoint-6000": "xlmrl_sl-bcms-6", "/cache/nikolal/xlmrl_sl-bcms_exp/checkpoint-12000": "xlmrl_sl-bcms-12", "/cache/nikolal/xlmrl_sl-bcms_exp/checkpoint-18000": "xlmrl_sl-bcms-18", "/cache/nikolal/xlmrl_sl-bcms_exp/checkpoint-24000": "xlmrl_sl-bcms-24", "/cache/nikolal/xlmrl_sl-bcms_exp/checkpoint-30000": "xlmrl_sl-bcms-30", "/cache/nikolal/xlmrl_sl-bcms_exp/checkpoint-42000": "xlmrl_sl-bcms-42", "/cache/nikolal/xlmrl_sl-bcms_exp/checkpoint-48000": "xlmrl_sl-bcms-48"}

	# Define the model arguments - use the same one as for XLM-R-large if model is based on it,
	# if the model is of same size as XLM-R-base, use its optimal hyperparameters (I searched for them before)
	xlm_r_large_args = {"overwrite_output_dir": True,
			"num_train_epochs": 5,
			"labels_list": LABELS,
			"learning_rate": 1e-5,
			"train_batch_size": 32,
			# Comment out no_cache and no_save if you want to save the model
			"no_cache": True,
			"no_save": True,
			"max_seq_length": 256,
			"save_steps": -1,
			"silent": True,
			}

	xlm_r_base_args = {"overwrite_output_dir": True,
			"num_train_epochs": 9,
			"labels_list": LABELS,
			"learning_rate": 1e-5,
			"train_batch_size": 32,
			# Comment out no_cache and no_save if you want to save the model
			"no_cache": True,
			"no_save": True,
			"max_seq_length": 256,
			"save_steps": -1,
			"silent": True,
			}
	
	if model_size == "base":
		# Update the hyperparameters accordingly to the model
		model_args = xlm_r_base_args
	elif model_size == "large":
		model_args = xlm_r_large_args

	# Add additional arguments, specific for our own models
	# Specify the folder where we want to save the models
	new_model_path = path_list[model_path]
	model_args["output_dir"] = "models/{}/".format(new_model_path)
	model_args["no_save"] = False
	model_args["num_train_epoch"] = 1

	# Define the model
	current_model = NERModel(
	"xlmroberta",
	model_path,
	labels = LABELS,
	use_cuda=True,
	args = model_args)

	print("Training started. Current model: {}".format(model_path))

	# Fine-tune the model
	current_model.train_model(train_df)

	print("Training completed.")

	print("Model saved as models/{}/".format(new_model_path))

	# Clean cache
	gc.collect()
	torch.cuda.empty_cache()

#for i in base_list:
#	train_and_save_checkpoint(i, "base", train_df, LABELS)

for i in large_list:
	train_and_save_checkpoint(i, "large", train_df, LABELS)