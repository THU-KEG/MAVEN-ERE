# Subevent Relation Extraction
This directory contains codes for subevent relation extraction experiments.

## Dataset
- MAVEN-ERE
  - Released in this repo.
- HiEve
  - Retrieved from [this link](http://takelab.fer.hr/hievents.rar). We also share a preprocessed copy in `/data`.

## Brief Method Description

- On MAVEN-ERE, for the `n` events in each document, we consider each pair of events (altogether `n*(n-1)` pairs) and classify them.
- On the other datasets, for the `n` events in each document, we consider each pair of events with restriction e1 appears before e2 (altogether `n*(n-1)/2` pairs) and classify them.
- Our implementations refer to [this repo](https://github.com/CogComp/Subevent_EventSeg).

## Overview
- `data.py` and `data_other.py` handle data processing and loading part for MAVEN-ERE and other datasets
- `main.py` is the main access to run on MAVEN-ERE
- `main_other.py` is the main access to run on the other datasets

## Usage
- Running on MAVEN-ERE
    ```shell
    python -u main.py --epochs 20 --eval_steps 100 --log_steps 50
    ```
- Running on HiEve
    ```shell
    python -u main_other.py \
    --dataname hievents \
    --epochs 5000 --eval_steps 500 --log_steps 500 --batch_size 16
    ```