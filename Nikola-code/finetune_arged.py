
import sys
import os
import argparse
import json
import random as python_random
import numpy as np
import torch
from simpletransformers.ner import NERModel
import pandas as pd
from utils import parse_ner as parse
from simpletransformers.ner import NERModel
import evaluate

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
#parser.add_argument("-i", "--train_file", required=True,
#                    type=str,
#                     help="Input file to learn from"
#                     )
#parser.add_argument("-t", "--test_file", type=str, required=True,
#                    help="Files to test on."
#                    )
#parser.add_argument("-d", "--dev_file", type=str, required=True,
#                    help="Files to test on."
#                    )
#parser.add_argument("-lt", "--lm_type", type=str, default="electra", required=True,
#                    help="Simpletransformers LM type identifier")
parser.add_argument("-ln", "--lm_name", type=str, default="xlm-r-base", required=True,
                    help="Simpletransformers LM name or path")
parser.add_argument("-s", "--seed", type=int, default=2222,
                    help="Random seed that we use")
parser.add_argument("-o", "--output_file", default="eval.log", type=str,
                    help="What file to append the results to.")
#parser.add_argument("-n", "--gpu_n", type=int, default=-1,
#                    help="Which gpu device to use")
parser.add_argument("--runs", type=int, default=1,
                    help="How many runs to perform, and average the results")
parser.add_argument("--output_dir", type=str, default='output/',
                    help="What output dir to use (for parallel runs, we need different output dirs)")

args = parser.parse_args()

   
train_path = "/home/nikolal/tools/LM_eval/finetune_data/hr500k-train.ner"
#train_path = args.train_file
dev_path = "/home/nikolal/tools/LM_eval/finetune_data/hr500k-dev.ner"
#dev_path = args.dev_file
test_path = "/home/nikolal/tools/LM_eval/finetune_data/hr500k-test.ner"
#test_path = args.test_file

task = "ner"

train = parse(train_path, target=task)
dev = parse(dev_path, target=task)
test = parse(test_path, target=task)

# Extract all the labels that appear in the data:
labels = train.labels.unique().tolist() + test.labels.unique().tolist() + dev.labels.unique().tolist()
labels = list(set(labels))

xlm_r_large_args = {"overwrite_output_dir": True,
            "num_train_epochs": 7,
            "labels_list": labels,
            "learning_rate": 1e-5,
            "train_batch_size": 32,
            # Comment out no_cache and no_save if you want to save the model
            "no_cache": True,
            "no_save": True,
            "max_seq_length": 256,
            "save_steps": -1,
            #"silent": True,
            "use_multiprocessing": False,
            "use_multiprocessing_for_evaluation": False,
            "evaluate_during_training": False,
            }

xlm_r_base_args = {"overwrite_output_dir": True,
            "num_train_epochs": 8,
            "labels_list": labels,
            "learning_rate": 1e-5,
            "train_batch_size": 32,
            # Comment out no_cache and no_save if you want to save the model
            "no_cache": True,
            "no_save": True,
            "max_seq_length": 256,
            "save_steps": -1,
            #"silent": True,
            "use_multiprocessing": False,
            "use_multiprocessing_for_evaluation": False,
            "evaluate_during_training": False,
            }

model_type_dict = {
    "sloberta": ["camembert", "EMBEDDIA/sloberta", xlm_r_base_args],
    "csebert": ["bert", "EMBEDDIA/crosloengual-bert", xlm_r_base_args],
    "xlm-r-base": ["xlmroberta", "xlm-roberta-base", xlm_r_base_args],
    "xlm-r-large": ["xlmroberta", "xlm-roberta-large", xlm_r_large_args],
    "bertic": ["electra", "classla/bcms-bertic", xlm_r_base_args]
}

# Create a NERModel
defined_model = args.lm_name
model_name = model_type_dict[defined_model][1]
model_type = model_type_dict[defined_model][0]

model_args = model_type_dict[defined_model][2]

# Set no. of epochs to 2 just to see if everything works
model_args["num_train_epochs"] = 2

macrof1_N=[]
microf1_N=[]
for i in range(args.runs):
    test = parse(test_path, target=task)

    model = NERModel(model_type, model_name, labels=labels, use_cuda=True)

    model.train_model(train_data=train, 
                eval_data=dev,
                args = model_args)

    results, model_outputs, predictions = model.eval_model(test)

    from sklearn.metrics import f1_score, classification_report

   # Calculate macro F1 with Nikola's/Rik's method as well
    # The model returns a list of lists, with the total count sometimes not adding up (model
    # discards some instances, it seems. Happens to ~2 instances per test split, but then the evaluation crashes.)
    # We unfold this list of lists, add it to original test data, and discard all of the sentences where
    # there is a mismatch.
    kept_sentences = 0
    discarded_sentences = 0
    test["y_pred"] = ""
    for i in test.sentence_id.unique():
        subset = test[test.sentence_id == i]
        if subset.shape[0] == len(predictions[i]):
            test.loc[test.sentence_id == i, "y_pred"] = predictions[i]
            kept_sentences += 1
        else:
            discarded_sentences += 1
            continue
    test_N = test[test.y_pred != ""]
    y_true_N = test_N.labels.tolist()
    y_pred_N = test_N.y_pred.tolist()

    macrof1_N.append(f1_score(y_true_N, y_pred_N, labels=labels, average='macro'))
    microf1_N.append(f1_score(y_true_N, y_pred_N, labels=labels, average='micro'))
    clfreport = classification_report(y_true_N, y_pred_N, labels=labels)
    print("Micro F1 and Macro F1 according to Nikola's method:\n")
    print(str(microf1_N)+'\t'+str(macrof1_N))
    print(clfreport)

    print("Number of kept and discarded sentences; percentage of discarded sentences:")
    print(kept_sentences)
    print(discarded_sentences)
    print(discarded_sentences/(kept_sentences+discarded_sentences)*100)

    # My method
    # Get predictions
    preds_T = model_outputs

    # Create a list with predictions - my method
    preds_list_T = []

    for sentence in preds_T:
        for word in sentence:
            current_word = []
            for element in word:
                # Find prediction with the highest value
                highest_index = element.index(max(element))
                # Transform the index to label
                current_pred = model.config.id2label[highest_index]
                # Append to the list
                current_word.append(current_pred)
            # Segmentation can result in multiple predictions for one word - use the first prediction only
            preds_list_T.append(current_word[0])
    
    y_true_T = test.labels.tolist()

    # Evaluate predictions
    metrics = evaluate.testing(y_true_T, preds_list_T, labels, show_matrix=True)

    from datetime import datetime
    log = {"train": train_path,
       "dev": dev_path,
       "test": test_path,
       "model_type": model_type,
       "model_name": model_name,
       "microF1": microf1_N,
       "macroF1": macrof1_N,
       "timestamp": datetime.now().__str__(),
       "train_args": model_args,
       "seed": args.seed,
       "labels": labels,
       "y_true": y_true_N,
       "y_pred": y_pred_N,
       "classification_report": "\n" + clfreport + "\n",
       }

    with open(args.output_file, "a") as f:
        import json
        f.write(json.dumps(log) + "\n")
    #print(clfreport)
    #print(str(np.mean(microf1))+'\t'+str(np.mean(macrof1)))
    #print(microf1,macrof1)
