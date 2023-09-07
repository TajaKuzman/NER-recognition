import evaluate
from datetime import datetime
import pandas as pd
import numpy as np
import json
from simpletransformers.ner import NERModel, NERArgs
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
test_df = pd.DataFrame(json_dict["test"])
dev_df = pd.DataFrame(json_dict["dev"])

# Change the sentence_ids to numbers
test_df['sentence_id'] = pd.factorize(test_df['sentence_id'])[0]
train_df['sentence_id'] = pd.factorize(train_df['sentence_id'])[0]
dev_df['sentence_id'] = pd.factorize(dev_df['sentence_id'])[0]

# Define the labels
LABELS = json_dict["labels"]
print(LABELS)

print(train_df.shape, test_df.shape, dev_df.shape)
print(train_df.head())

# Define the main model arguments
model_args = {"overwrite_output_dir": True,
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

# Create a dict and list for additional model(s)
specific_model_dict = {"/cache/nikolal/xlmrl_sl-bcms_exp/checkpoint-36000": "xlmrl_sl-bcms-36"}

specific_models_list = list(specific_model_dict.keys())

def train_and_save_checkpoint(model_path, train_df, LABELS, model_args):
    # When fine-tuning our custom models that we pre-trained, and using them from checkpoints, the process is a bit different than with publicly available models: first, we need to fine-tune a model from the original checkpoint, so that we save the model and overwrite its original settings which force pretraining from a specific step (and disable fine-tuning by that). Then we take that new model and fine-tune it, as we did with the models before. 

    # Add additional arguments, specific for our own models
    # Specify the folder where we want to save the models
    new_model_path = "models/"
    model_args["output_dir"] = new_model_path
    model_args["no_save"] = False
    model_args["num_train_epoch"] = 1

    # Define the model
    current_model = NERModel(
    "xlmroberta",
    model_path,
    labels = LABELS,
    use_cuda=True,
    args = model_args)

    print("Training of pre-trained model started. Current model: {}".format(model_path))

    # Fine-tune the model
    current_model.train_model(train_df)

    print("Training of pre-trained model completed.")

    print("Model saved in models/")

    # Clean cache
    gc.collect()
    torch.cuda.empty_cache()

# After creating pre-trained model that we can use, train it properly
def train_and_test(model, train_df, test_df, dataset_path, LABELS, model_args):

    # Define the model

    # Define the model arguments - use the same one as for XLM-R-large if model is based on it,
    # if the model is of same size as XLM-R-base, use its optimal hyperparameters (I searched for them before).
    # Args also depend on the dataset.

    # Define the type of dataset we are using
    # - when we extend the code for SL, change this
    dataset_type = "standard_hr"

    if "reldi" in dataset_path:
        dataset_type = "non_standard"
    elif "set.sr" in dataset_path:
        dataset_type = "standard_sr"

    # Change no. of epochs based on the model and the dataset
    if dataset_type == "standard_hr":
        # If the model is based on XLM-R-base, use the same arg as XLM-R-base
        if "xlmrb" in model:
            model_args["num_train_epochs"] = 9
        # If the model is based on XLM-R-large, use the same arg as XLM-R-large
        elif "xlmrl" in model:
            model_args["num_train_epochs"] = 5
    elif dataset_type == "non_standard":
        if "xlmrb" in model:
            model_args["num_train_epochs"] = 11
        elif "xlmrl" in model:
            model_args["num_train_epochs"] = 7
    elif dataset_type == "standard_sr":
        model_args["num_train_epochs"] = 11

    # Define the model
    current_model = NERModel(
    "xlmroberta",
    "models/",
    labels = LABELS,
    use_cuda=True,
    args = model_args)

    print("Training started. Current model: {}".format(model))
    start_time = time.time()

    # Fine-tune the model
    current_model.train_model(train_df)

    print("Training completed.")

    training_time = round((time.time() - start_time)/60,2)

    print("It took {} minutes for {} instances.".format(training_time, train_df.shape[0]))

    # Clean cache
    gc.collect()
    torch.cuda.empty_cache()

    start_evaluation_time = time.time()

    # Evaluate the model
    results = current_model.eval_model(test_df)

    print("Evaluation completed.")

    evaluation_time = round((time.time() - start_evaluation_time)/60,2)

    print("It took {} minutes for {} instances.".format(evaluation_time, test_df.shape[0]))

    # Get predictions
    preds = results[1]

    # Create a list with predictions
    preds_list = []

    for sentence in preds:
        for word in sentence:
            current_word = []
            for element in word:
                # Find prediction with the highest value
                highest_index = element.index(max(element))
                # Transform the index to label
                current_pred = current_model.config.id2label[highest_index]
                # Append to the list
                current_word.append(current_pred)
            # Segmentation can result in multiple predictions for one word - use the first prediction only
            preds_list.append(current_word[0])
    
    # Get y_true
    y_true = list(test_df.labels)

    run_name = "{}-{}".format(dataset_path, model)

    # Evaluate predictions
    metrics = evaluate.testing(y_true, preds_list, list(test_df.labels.unique()), run_name, show_matrix=True)

    # Add y_pred and y_true to the metrics dict
    metrics["y_true"] = y_true
    metrics["y_pred"] = preds_list

    # Let's also add entire results
    metrics["results_output"] = results    
    
    # The function returns a dict with accuracy, micro f1, macro f1, y_true and y_pred
    return metrics

# For each model, repeat training and testing 5 times - let's do 2 times for starters
model_list = specific_models_list

for model_path in model_list:
    # First, save a fine-tuned version that we can use for proper fine-tuning
    train_and_save_checkpoint(model_path, train_df, LABELS, model_args)
    model = specific_model_dict[model_path]
    # Then do multiple runs with this model
    # Let's do 4 runs
    for run in [0,1,2,3]:
        current_results_dict = train_and_test(model, train_df, test_df, dataset_path, LABELS, model_args)

        # Add to the dict model name, dataset name and run
        current_results_dict["model"] = model
        current_results_dict["run"] = "{}-{}".format(model, run)
        current_results_dict["dataset"] = dataset_path

        # Add to the file with results all important information
        with open("ner-results-our-models.txt", "a") as file:
            file.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), current_results_dict["model"], current_results_dict["run"], current_results_dict["dataset"], current_results_dict["micro F1"], current_results_dict["macro F1"], current_results_dict["label-report"]))

        # Add to the original test_df y_preds
        test_df["y_pred_{}_{}".format(model, run)] = current_results_dict["y_pred"]

        # Save also y_pred and y_true
        with open("logs/{}-{}-{}-true-and-pred-backlog.txt".format(dataset_path,model,run), "w") as backlog:
            backlog.write("y-true\ty-pred\toutputs\n")
            backlog.write("{}\t{}\t{}\n".format(current_results_dict["y_true"], current_results_dict["y_pred"], current_results_dict["results_output"]))
    
        print("Run {} finished.".format(run))
    
    # Then delete the model from the /models folder
    folder_path = "models"

    # List all files in the folder
    file_list = os.listdir(folder_path)

    # Loop through the files and delete each one
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)


# At the end, save the test_df with all predictions
test_df.to_csv("{}-test_df-with-predictions-custom-models.csv".format(dataset_path))

# At the end, create a csv table with a summary of results

results = pd.read_csv("ner-results-our-models.txt", sep="\t")

results["Macro F1"] = results["Macro F1"].round(2)

# Pivot the DataFrame to rearrange columns into rows
pivot_df = results.pivot(index='Run', columns='Dataset', values='Macro F1')

# Reset the index to have 'Model' as a column
pivot_df.reset_index(inplace=True)

# Pivot the DataFrame to rearrange columns into rows
pivot_df.to_csv("ner-results-summary-table-our-models.csv")