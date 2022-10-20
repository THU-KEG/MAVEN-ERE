# Causal Relation Extraction
This directory contains code for causal relation extraction.
## Dataset
- MAVEN
- CausalTimeBank
- EventStoryLine

## Setting
- For MAVEN, for n events in one document, consider each pair of events (altogether n*(n-1)), classify them into different types of causal relations.
- For other datasets, for n events in one document, consider each pair of events with restriction e1 appears before e2 (altogether n*(n-1)/2), classify them into different types of causal relations.
- according to previous works, CusalTimebank is evaluated with 10-fold cross-validation, EventStoryline is evaluated with 5-fold cross-validation
- for evaluation, use standard classification metrics.

## Overview
- data.py, data_other.py handle data processing and loading part for MAVEN and other datasets
- main.py is the main access to run on MAVEN
- main_other.py is the main access to run on other datasets

## Usage
- run on MAVEN
    ```shell
    python -u main.py --eval_steps 500 --epochs 50 --batch_size 4
    ```
- run on CausallTimebank and EventStoryline
    ```shell
    python -u main_other.py --dataname CausalTimeBank --epochs 200 --eval_steps 100 --log_steps 50 --K 10
    python -u main_other.py --dataname EventStoryLine --epochs 50 --eval_steps 100 --log_steps 50 --K 5
    ```
    - `K` denotes K-fold validation

## Evaluation
