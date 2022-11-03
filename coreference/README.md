# Coreference Relation Extraction
This directory contains codes for the event coreference relation extraction (event coreference resolution) experiments.

## Dataset
- MAVEN-ERE
  - Released in this repo.
- TAC KBP
  - Retrieved from [LDC2020T13](https://catalog.ldc.upenn.edu/LDC2020T13) and [LDC2020T18](https://catalog.ldc.upenn.edu/LDC2020T18). See the paper for more split details. Due to the license limit, we cannot share preprocessed data here. We use [OmniEvent](https://github.com/THU-KEG/OmniEvent) to preprocess the data. 
- ACE 2005
  - Retrieved from [LDC2006T06](https://catalog.ldc.upenn.edu/LDC2006T06). Due to the license limit, we cannot share preprocessed data here. We use [OmniEvent](https://github.com/THU-KEG/OmniEvent) to preprocess the data. 

## Brief Method Description
- For the `n` events in each document, we consider each pair of events with restriction e1 appears before e2 (altogether `n*(n-1)/2` pairs) and classify them into `coreference` or `none` labels.
- For evaluation, we use `BLANC`, `CEAF-e`, `MUC`, `B-cubed` metrics.

## Overview
- `data.py`, `data_ace.py`, and `data_kb.py` handle data processing and loading parts for the 3 datasets.
- `main.py` is the main access to run the code.

## Usage
- Running on MAVEN-ERE
    ```shell
    python -u main.py --epochs 50 --log_steps 20 --eval_steps 50
    ```
- Running on TAC KBP
    ```shell
    python -u main.py --dataset kbp --epochs 50 --log_steps 20 --eval_steps 50
    ```
- Running on ACE 2005
    ```shell
    python -u main.py --dataset ace --epochs 50 --log_steps 20 --eval_steps 50
    ```

## Evaluation
- We report 4 metrics for evaluation: `BLANC`, `CEAF-e`, `MUC`, `B-cubed`
- See [`metrics.py`](metrics.py) for implementation details.