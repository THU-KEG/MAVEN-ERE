# Temporal Relation Extraction
This directory contains codes for jointly training event coreference, temporal, causal and subevent relation extraction experiments on MAVEN-ERE.

## Dataset
- MAVEN-ERE
  - Released in this repo.

## Overview
- `data.py` handle data processing and loading part for MAVEN-ERE
- `main.py` is the main access to run on MAVEN-ERE

## Usage
- Running on MAVEN-ERE
    ```shell
    python -u main.py --eval_steps 200 --epochs 100 --lr 3e-4 --bert_lr 2e-5 --accumulation_steps 4 --batch_size 8
    ```