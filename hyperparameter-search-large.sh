#!/bin/sh

export CUDA_VISIBLE_DEVICES=3

python hyperparameter_search.py -d datasets/hr500k.conllup_extracted.json -ln xlm-r-large > logs/search_hr_large_new_lr.txt

python hyperparameter_search.py -d datasets/reldi-normtagner-hr.conllup_extracted.json -ln xlm-r-large > logs/search_hr_reldi_large_new_lr.txt

python hyperparameter_search.py -d datasets/set.sr.plus.conllup_extracted.json -ln xlm-r-large > logs/search_sr_large_new_lr.txt