# Coreference Resolution
This directory contains code for coreference relation.
## Dataset
- MAVEN
- KBP
- ACE

## Setting
- for n events in one document, consider each pair of events with restriction e1 appears before e2 (altogether n*(n-1)/2), classify them into "coreference" or "none" relation.
- for evaluation, use B-cubed, CEAFe, BLANC, and MUC metrics.

## Overview
- data.py, data_ace.py, and data_kb.py handle data processing and loading part for the 3 datasets
- main.py is the main access to run the code

## Usage
- run on MAVEN
    ```shell
    python -u main.py --epochs 50 --log_steps 20 --eval_steps 50
    ```
- run on KBP
    ```shell
    python -u main.py --dataset kbp --epochs 50 --log_steps 20 --eval_steps 50
    ```
- run on ACE
    ```shell
    python -u main.py --dataset ace --epochs 50 --log_steps 20 --eval_steps 50
    ```

## Evaluation
- We report 3 metrics for evaluation: BLANC, CEAF-e, MUC, B-cubed
- see [`metrics.py`](metrics.py) for implementation details