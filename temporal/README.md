# Temporal Relation Extraction
This directory contains code for temporal relation extraction.
## Dataset
- MAVEN
- MATRES
- TB-Dense
- TCR (only for test)

## Setting
- There are two settings
    - **setting 1** for n events in one document, consider each pair of events (altogether n*(n-1)), classify them into different types of temporal relations.
    - **setting 2** consider only event paris within a window of 2 sentences, classify them into different types of temporal relations.
- For MATRES and TB-Dense datasets, only **setting 2** is available because both datasets only annotate events within a window of 2 sentences.
- for TCR dataset, directly test on the whole TCR dataset with model trained on MATRES
- for evaluation, use standard classification metrics.

## Overview
- data.py, data_other.py handle data processing and loading part for MAVEN and other datasets
- main.py is the main access to run on MAVEN
- main_other.py is the main access to run on other datasets

## Usage
- run on MAVEN under setting 1
    ```shell
    python -u main.py --epochs 50 --log_steps 20 --eval_steps 50
    ```
- run on MAVEN under setting 2
    ```shell
    python -u main.py --epochs 50 --log_steps 20 --eval_steps 50 --ignore_nonetype
    ```
- run on MATRES, TB-Dense under setting 2
    ```shell
    python -u main_other.py --dataname MATRES --epochs 50 --eval_steps 100 --log_steps 50 --ignore_nonetype
    python -u main_other.py --dataname TB-Dense --epochs 500 --eval_steps 20 --log_steps 20 --ignore_nonetype
    ```
- test on TCR under setting 2
    ```shell
    python -u main_other.py --dataname TCR --eval_only --load_ckpt ../output/0/MATRES_ignore_none_True/best --ignore_nonetype
    ```

## Evaluation
