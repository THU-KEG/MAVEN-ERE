# Causal Relation Extraction
This directory contains codes for causal relation extraction experiments.

## Dataset
- MAVEN-ERE
  - Released in this repo.
- CausalTimeBank
  - Retrieved from [this link](https://github.com/paramitamirza/Causal-TimeBank). We also share a preprocessed copy in `/data`.
- EventStoryLine
  - Retrieved from [this link](https://github.com/tommasoc80/EventStoryLine). We also share a preprocessed copy in `/data`.

## Brief Method Description
- On MAVEN-ERE, for the `n` events in each document, we consider each pair of events (altogether `n*(n-1)` pairs) and classify them.
- On the other datasets, for the `n` events in each document, we consider each pair of events with restriction e1 appears before e2 (altogether `n*(n-1)/2` pairs) and classify them.
- Following previous work, CausalTimeBank is evaluated with `10`-fold cross-validation and EventStoryline is evaluated with `5`-fold cross-validation

## Overview
- `data.py` and `data_other.py` handle data processing and loading part for MAVEN-ERE and the other datasets
- `main.py` is the main access to run on MAVEN-ERE
- `main_other.py` is the main access to run on the other datasets

## Usage
- Running on MAVEN-ERE
    ```shell
    python -u main.py --eval_steps 500 --epochs 50 --batch_size 4
    ```
- Running on CausalTimebank and EventStoryline
    ```shell
    python -u main_other.py --dataname CausalTimeBank --epochs 200 --eval_steps 100 --log_steps 50 --K 10
    python -u main_other.py --dataname EventStoryLine --epochs 50 --eval_steps 100 --log_steps 50 --K 5
    ```
    - `K` denotes K-fold validation