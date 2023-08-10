#!/bin/bash
bash download_ner_datasets.sh "$@"

# Iterate through arguments
for arg in "$@"; do
    echo "Processing dataset: $arg"
    python3 create_NER_task_files.py "$arg"
done