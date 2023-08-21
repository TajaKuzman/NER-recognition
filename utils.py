
import pandas as pd

def parse_conllu(filename: str, target:str = "xpos") -> pd.DataFrame:
    """Reads conllu file, returns dataframe with columns 
    * `sentence_id`,
    * `form`,
    * `label`

    Args:
        filename (str): path to file
        target (str, optional): either `upos`, `xpos`, `lemma`,
            `deprel`, `head`, `deps`, `misc`, or `feats`, . Defaults to "xpos".

    Returns:
        pd.DataFrame: _description_
    """    
    from conllu import parse
    import pandas as pd
    with open(filename, "r") as f:
        sentences = parse(f.read())
    sentence_id_list = []
    text_list = []
    label_list = []
    for current_sentence_id, sentence in enumerate(sentences):
        for token in sentence:
            sentence_id_list.append(current_sentence_id)
            text_list.append(token["form"])
            label_list.append(token[target])
    df = pd.DataFrame({"sentence_id": sentence_id_list, "words": text_list, "labels": label_list})
    return df

def parse_ner(filename: str, **kwargs) -> pd.DataFrame:
    """Reads ner file (tab separated file with sentence_id, word, ner), returns dataframe with columns 
    * `sentence_id`,
    * `form`,
    * `label`

    Args:
        filename (str): path to file

    Returns:
        pd.DataFrame: _description_
    """

    # df = pd.read_csv(filename,
    #         sep="\t", 
    #         names=["sentence_id", "words", "labels"],
    #         engine="python",
    #         on_bad_lines="skip",
    #         )

    # d = {label: i for i, label in enumerate(df.sentence_id.unique())}
    # df["sentence_id"] = df.sentence_id.apply(lambda s: d[s])
    # return df
    sentence_ids, words, labels = [], [], []
    with open(filename) as f:
        content = f.readlines()
    for line in content:
        line = line.replace("\n", "")
        items = line.split("\t")
        sentence_ids.append(items[0])
        words.append(items[1])
        labels.append(items[2])
    df = pd.DataFrame(data={
        "sentence_id":sentence_ids,
        "words": words,
        "labels": labels
    })
    d = {label: i for i, label in enumerate(df.sentence_id.unique())}
    df["sentence_id"] = df.sentence_id.apply(lambda s: d[s])
    return df

def parse_class(filename: str, **kwargs) -> pd.DataFrame:
    df=pd.read_csv(filename,names=("text","labels"),sep="\t")
    return df