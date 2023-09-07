#!/bin/sh

export CUDA_VISIBLE_DEVICES=4

python hyperparameter_search.py -d datasets/hr500k.conllup_extracted.json -ln xlm-r-base > logs/search_hr_base_new_lr.txt

python hyperparameter_search.py -d datasets/hr500k.conllup_extracted.json -ln csebert > logs/search_hr_csebert_new_lr.txt

python hyperparameter_search.py -d datasets/hr500k.conllup_extracted.json -ln bertic > logs/search_hr_bertic_new_lr.txt

python hyperparameter_search.py -d datasets/reldi-normtagner-hr.conllup_extracted.json -ln xlm-r-base > logs/search_hr_reldi_base_new_lr.txt

python hyperparameter_search.py -d datasets/reldi-normtagner-hr.conllup_extracted.json -ln csebert > logs/search_hr_reldi_csebert_new_lr.txt

python hyperparameter_search.py -d datasets/reldi-normtagner-hr.conllup_extracted.json -ln bertic > logs/search_hr_reldi_bertic_new_lr.txt

python hyperparameter_search.py -d datasets/set.sr.plus.conllup_extracted.json -ln xlm-r-base > logs/search_sr_base_new_lr.txt

python hyperparameter_search.py -d datasets/set.sr.plus.conllup_extracted.json -ln csebert > logs/search_sr_csebert_new_lr.txt

python hyperparameter_search.py -d datasets/set.sr.plus.conllup_extracted.json -ln bertic > logs/search_sr_bertic_new_lr.txt