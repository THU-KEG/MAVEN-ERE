# Subevent Relation Extraction
This directory contains code for causal relation extraction.
## Dataset
- MAVEN
- Hievents

## Setting
- For MAVEN, for n events in one document, consider each pair of events (altogether n*(n-1)), classify them into different types of causal relations.
- For other datasets, for n events in one document, consider each pair of events with restriction e1 appears before e2 (altogether n*(n-1)/2), classify them into different types of causal relations.
- reference: https://github.com/CogComp/Subevent_EventSeg
- for evaluation, use standard classification metrics.

## Overview
- data.py, data_other.py handle data processing and loading part for MAVEN and other datasets
- main.py is the main access to run on MAVEN
- main_other.py is the main access to run on other datasets

## Usage
- run on MAVEN
    ```shell
    # full training set
    python -u main.py --epochs 20 --eval_steps 100 --log_steps 50
    ```
- run on Hievents
    ```shell
    python -u main_other.py \
    --dataname hievents \
    --epochs 5000 --eval_steps 500 --log_steps 500 --batch_size 16
    ```

## Evaluation
