# Temporal Relation Extraction
This directory contains codes for temporal relation extraction experiments.
## Dataset
- MAVEN-ERE
  - Released in this repo.
- MATRES
  - Retrieved from [this link](https://github.com/qiangning/MATRES). We also share a preprocessed copy in `/data`.
- TB-Dense
  - Retrieved from [this link](https://www.usna.edu/Users/cs/nchamber/caevo/). We also share a preprocessed copy in `/data`.
- TCR
  - Retrieved from [this link](https://github.com/qiangning/TemporalCausalReasoning). We also share a preprocessed copy in `/data`.

## Brief Method Description
- On MAVEN-ERE, for the `n` events in each document, we consider each pair of events (altogether `n*(n-1)` pairs) and classify them.
- On the other datasets, we consider only event pairs within `2`-sentence windows since they only annotate them.
- For the TCR dataset, we directly test on the whole TCR dataset with models trained on MATRES

## Overview
- `data.py` and `data_other.py` handle data processing and loading part for MAVEN-ERE and other datasets
- `main.py` is the main access to run on MAVEN-ERE
- `main_other.py` is the main access to run on the other datasets

## Usage
- Running on MAVEN-ERE
    ```shell
    python -u main.py --epochs 50 --log_steps 20 --eval_steps 50
    ```
- Running on MATRES and TB-Dense
    ```shell
    python -u main_other.py --dataname MATRES --epochs 50 --eval_steps 100 --log_steps 50 --ignore_nonetype
    python -u main_other.py --dataname TB-Dense --epochs 500 --eval_steps 20 --log_steps 20 --ignore_nonetype
    ```
- Running on TCR
    ```shell
    python -u main_other.py --dataname TCR --eval_only --load_ckpt ../output/0/MATRES_ignore_none_True/best --ignore_nonetype
    ```