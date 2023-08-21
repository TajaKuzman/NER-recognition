import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", help="Possible arguments: s_Slovene (standard Slovene), ns_Slovene (non-standard Slovene), s_Croatian, ns_Croatian, s_Serbian, ns_Serbian")
    args = parser.parse_args()

# Define the scenario
scenario = args.scenario


def extract_ner_dataset(scenario):
    """
    Extract a NER dataset that can be used for NER evaluation with simple transformers.
    Args:
        - scenario: s_Slovene (standard Slovene), ns_Slovene (non-standard Slovene),
                    s_Croatian, ns_Croatian, s_Serbian, ns_Serbian
    """
    from conllu import parse
    import pandas as pd
    import numpy as np
    import json
    import random

    datasets = {
    "s_Slovene": {
        "name": "Slovenian Training corpus SUK 1.0",
        "path":"https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1747/SUK.CoNLL-U.zip",
        "downloaded_file": "SUK.CoNLL-U.zip",
        "dataset":["SUK.CoNLL-U/elexiswsd.ud.conllu", "SUK.CoNLL-U/senticoref.ud.conllu", "SUK.CoNLL-U/ssj500k-syn.ud.conllu"]},
    "ns_Slovene": {
        "name": "Slovenian CMC training corpus Janes-Tag 3.0 ",
        "path":"https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1732/Janes-Tag.3.0.CoNLL-U.zip",
        "downloaded_file": "Janes-Tag.3.0.CoNLL-U.zip",
        "dataset": ["Janes-Tag.3.0.CoNLL-U/janes-tag.ud.conllu"]},
    "s_Croatian": {
        "name": "Croatian linguistic training corpus hr500k 2.0",
        "path":"https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1792/hr500k.conllup",
        "dataset": ["hr500k.conllup"]},
    "ns_Croatian": {
        "name": "Croatian Twitter training corpus ReLDI-NormTagNER-hr 3.0",
        "path":"https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1793/reldi-normtagner-hr.conllup",
        "dataset": ["reldi-normtagner-hr.conllup"]},
    "s_Serbian": {
        "name": "Serbian linguistic training corpus SETimes.SR 2.0",
        "path":"https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1843/set.sr.plus.conllup",
        "dataset": ["set.sr.plus.conllup"]},
    "ns_Serbian": {
        "name": "Serbian Twitter training corpus ReLDI-NormTagNER-sr 3.0",
        "path":"https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1794/reldi-normtagner-sr.conllup",
        "dataset": ["reldi-normtagner-sr.conllup"]}
    }

    # Loop through all the datasets if there are multiple datasets for one scenario
    for i in range(len(datasets[scenario]["dataset"])):
        dataset = datasets[scenario]["dataset"][i]
        doc = "datasets/{}".format(dataset)

        # Open the dataset
        data = open("{}".format(doc), "r").read()

        # Parse conllu file
        sentences = parse(data)

        word_list = []
        sent_id_list = []
        NER_list = []
        split_list = []
        doc_list = []

        # Slovene corpora are not split into train, dev, test splits and have NER information under different keys than Croatian and Serbian
        if "Slovene" in scenario:
            # Collect all important information from the dataset
            for sentence in sentences:
                current_sent_id = sentence.metadata["sent_id"]

            # Extract doc_ids and create a list of doc_ids
                if sentence.metadata.get("newdoc id", None) != None:
                    current_doc_id = sentence.metadata["newdoc id"]
                # If sentence does not have a new doc id, use the one from the previous sentence that has it

                for token in sentence:
                    current_word = token["form"]
                    current_ner = token["misc"]["NER"]

                    word_list.append(current_word)
                    sent_id_list.append(current_sent_id)
                    NER_list.append(current_ner)
                    doc_list.append(current_doc_id)

            # Create a dictionary for all words and all needed information
            data_dict = {"sentence_id": sent_id_list, "words": word_list, "labels": NER_list, "doc_ids": doc_list}

            # Create a pandas df out of the dictionary
            df = pd.DataFrame(data_dict)

            LABELS = list(df.labels.unique())
            # If * is used, change * to O, because this causes errors
            if "*" in LABELS:
                LABELS[LABELS.index("*")] = "O"

                df["labels"] = np.where(df["labels"] == "*", "O", df["labels"])

            # Create splits - random 80:10:10 splits based on doc ids/sentence ids

            # Set a random seed for reproducibility
            random_seed = 42
            random.seed(random_seed)

            if "elexiswsd" in dataset:
                # Split the dataset based on sentence ids in a 80:10:10 ratio - elexis wsd does not have doc ids
                sen_ids = list(df["sentence_id"].unique())

                # Shuffle the sen_ids randomly
                random.shuffle(sen_ids)

                # Calculate the number of sen_ids for each split
                total_sents = len(sen_ids)
                train_size = int(0.8 * total_sents)
                test_size = int(0.1 * total_sents)

                # Split the shuffled doc_ids into train, test, and dev sets
                train_ids = sen_ids[:train_size]
                test_ids = sen_ids[train_size:train_size + test_size]
                dev_ids = sen_ids[train_size + test_size:]

                # Apply this to the dataset
                df["split"] = ""
                df["split"] = np.where(df['sentence_id'].isin(train_ids), "train", df["split"])
                df["split"] = np.where(df['sentence_id'].isin(test_ids), "test", df["split"])
                df["split"] = np.where(df['sentence_id'].isin(dev_ids), "dev", df["split"])

            else:
                # Split the dataset based on doc ids in a 80:10:10 ratio
                doc_ids = list(df["doc_ids"].unique())

                # Shuffle the doc_ids randomly
                random.shuffle(doc_ids)

                # Calculate the number of doc_ids for each split
                total_docs = len(doc_ids)
                train_size = int(0.8 * total_docs)
                test_size = int(0.1 * total_docs)

                # Split the shuffled doc_ids into train, test, and dev sets
                train_ids = doc_ids[:train_size]
                test_ids = doc_ids[train_size:train_size + test_size]
                dev_ids = doc_ids[train_size + test_size:]

                # Apply this to the dataset
                df["split"] = ""
                df["split"] = np.where(df['doc_ids'].isin(train_ids), "train", df["split"])
                df["split"] = np.where(df['doc_ids'].isin(test_ids), "test", df["split"])
                df["split"] = np.where(df['doc_ids'].isin(dev_ids), "dev", df["split"])

            # Show the df
            print(df.head())
            print("\n")
            print(df.describe(include="all"))
            print("\n")
            print(df.split.value_counts(normalize=True))
            print("\n")
            print(df.labels.value_counts(normalize=True))
            print("\n")

            # Save the information in a format that will be used by simpletransformers
            json_dict = {
                "labels": LABELS,
                "train": df[df["split"] == "train"].drop(columns=["split", "doc_ids"]).to_dict(),
                "dev": df[df["split"] == "dev"].drop(columns=["split", "doc_ids"]).to_dict(),
                "test": df[df["split"] == "test"].drop(columns=["split", "doc_ids"]).to_dict()
            }

    # Code for Serbian and Croatian corpora
        else:
            # Collect all important information from the dataset
            for sentence in sentences:
                current_sent_id = sentence.metadata["sent_id"]
                if sentence.metadata.get("contained_in_datasets", None) != None:
                    current_dataset = sentence.metadata["contained_in_datasets"]
                if "train" in current_dataset:
                    current_split = "train"
                elif "dev" in current_dataset:
                    current_split = "dev"
                elif "test" in current_dataset:
                    current_split = "test"
                for token in sentence:
                    current_word = token["form"]
                    current_ner = token["reldi:ne"]

                    word_list.append(current_word)
                    sent_id_list.append(current_sent_id)
                    NER_list.append(current_ner)
                    split_list.append(current_split)

            # Create a dictionary for all words and all needed information
            data_dict = {"sentence_id": sent_id_list, "words": word_list, "labels": NER_list, "split": split_list}

            # Create a pandas df out of the dictionary
            df = pd.DataFrame(data_dict)

            LABELS = list(df.labels.unique())

            # If * is used, change * to O, because this causes errors
            if "*" in LABELS:
                LABELS[LABELS.index("*")] = "O"

                df["labels"] = np.where(df["labels"] == "*", "O", df["labels"])
            
            # Show the df
            print(df.head())
            print("\n")
            print(df.describe(include="all"))
            print("\n")
            print(df.split.value_counts(normalize=True))
            print("\n")
            print(df.labels.value_counts(normalize=True))
            print("\n")

            # Save the information in a format that will be used by simpletransformers

            json_dict = {
                "labels": LABELS,
                "train": df[df["split"] == "train"].drop(columns="split").to_dict(),
                "dev": df[df["split"] == "dev"].drop(columns="split").to_dict(),
                "test": df[df["split"] == "test"].drop(columns="split").to_dict()
            }

        # Save json as file
        with open("datasets/{}_extracted.json".format(dataset), "w") as end_file:
            json.dump(json_dict, end_file, indent=2)

        print("\n\nExtracted dataset saved as datasets/{}_extracted.json".format(dataset))

extract_ner_dataset(scenario)