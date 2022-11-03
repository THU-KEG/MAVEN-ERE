# MAVEN-ERE
Source code and dataset for EMNLP 2022 paper "MAVEN-ERE: A Unified Large-scale Dataset for Event Coreference, Temporal, Causal, and Subevent Relation Extraction".

## Contents

- [Overview](#overview)
- [Getting Started](#requirements)
  - [Requirements](#requirements)
  - [MAVEN-ERE Dataset](#maven-ere-dataset)
    - [Get the Data](#get-the-data)
    - [Data Format](#data-format)
  - [Project Structure](#project-structure)
  - [Key Implementations](#key-implementations)
  - [How to Run](#how-to-run)
- [Citation](#citation)
- [Contact](#contact)

## Overview
- MAVEN-ERE is the first unified large-scale dataset for events relation extraction. It contains 4480 documents, with 103,193 coreference chains, 1,216,217 temporal relations, 15,841 subevent relations, and 57,992 causal relations. 
  - Temporal relations include: `BEFORE`, `OVERLAP`, `CONTAINS`, `SIMULTANEOUS`, `BEGINS-ON`, and `ENDS-ON`
  - Causal relations include: `CAUSE` and `PRECONDITION`
- The annotation rule of above relations mainly follows [RED guideline](https://github.com/timjogorman/RicherEventDescription/blob/master/guidelines.md).
- For more details, please see [the paper]().
![illustration](https://user-images.githubusercontent.com/55496186/196908089-e240d099-cdfd-4286-9046-e7e9d764562e.png)


## Requirements
```shell
pip install -r requirements.txt
```
## MAVEN-ERE Dataset
### Get the data
```bash
cd data
bash download_maven.sh
bash download.sh
cd ../
```

### Data Format
Each line in `jsonl` file contains one document. A sample of documents in `train.jsonl` and `dev.jsonl` is shown below
- `events` contains a list of coreference chains, where `mentions` contains a list of co-referenced mentions
- all relations are relations between `events`

```json
{
  "doc": {
    "id": "f28bce270df5a122c09365002d247e76",
    "title": "United States occupation of Nicaragua",
    "tokens": [
      ["The", "United", "States", "occupation", "of", "Nicaragua", "from", 
      "1912", "to", "1933", "was", "part", "of", "the", "Banana", "Wars", ",", 
      "when", "the", "US", "military", "intervened", "in", "various", 
      "Latin", "American", "countries", "from", "1898", "to", "1934", "."],
      
    ],
  },
  "events": [
      {
        "id": "EVENT_5b9109bbbddb8f4e254b6829b5e0397e",
        "mentions": [
          {
            "id": "85772635fcfb55540ee7e507ff874001",
            "trigger_word": "ended",
            "sent_id": 5,
            "offset": [14, 15],
            "event_type": "Process_end",
          },
          
        ]
      }
  ]
  "temporal_event_relation": {
    "before": [
      ["EVENT_id_1", "EVENT_id_2"],
      
    ],
    "overlap": [
      ["EVENT_id_1", "EVENT_id_2"],
      
    ],
    "contains": [
      ["EVENT_id_1", "EVENT_id_2"],
      
    ],
    "simultaneous": [
      ["EVENT_id_1", "EVENT_id_2"],
      
    ],
    "ends-on": [
      ["EVENT_id_1", "EVENT_id_2"],
      
    ],
    "begins-on": [
      ["EVENT_id_1", "EVENT_id_2"],
      
    ],
  },
  "causal_relation": {
    "CAUSE": [
      ["EVENT_id_1", "EVENT_id_2"],
      
    ],
    "PRECONDITION": [ 
      ["EVENT_id_1", "EVENT_id_2"],
      
    ],
  },
  "subevent_relation": [
    ["EVENT_id_1", "EVENT_id_2"],
    
  ]
}
```

- For `test.jsonl`, we do not release the true events. Instead, we release a set of `candidates` for each document (which contains more annotations that true events). 
- In predict mode, the model is expected to output relation prediction for each pair of candidates, but official evaluation is only conducted on true events.

Below is an example of test document.
```json
{
  "doc": {
    "id": "0f276a11d36371a901269fb1d0be6355",
    "title": "Conquest of Stockholm",
    "tokens": [
      ["The", "Conquest", "of", "Stockholm", "(", ")", "was", "a", "battle", 
      "in", "the", "Swedish", "War", "of", "Liberation", "that", "took", 
      "place", "in", "Stockholm", ",", "Sweden", "on", "17", "June", "1523", 
      "."],
      
    ],
  },
  "candidates": [
    {
      "trigger_word": "battle", "sent_id": 0, "offset": [8, 9], "id": "ad1e13bf97cd52fcf43086da34074953"
    },
    
  ]
}
```

## Project Structure
```
.
├── data
│   ├── download_maven.sh
│   ├── download.sh
├── causal
│   ├── output
│   ├── src
│       ├── data.py # dataloader utils
│       ├── dump_result.py # predict result output utils
│   ├── main.py # running script
├── coreference
│   ├── ...
├── temporal
│   ├── ...
├── subevent
│   ├── ...
├── utils
│   ├── model.py
│   ├── utils.py
└── README.md
```

## Key Implementations

### Batching strategy
- refer to `data.py` in each task folder
- We consider each document as a single sample, so `batch size=4` means in one batch there are 4 documents sent into the model.
- For each document, the representation of each possible pair of events are extracted and combined for prediction. E.g. if there are `n` events in one document, altogether `n*(n-1)` pairs of events are extracted.
- details can be seen in `src/data.py` in each task folder


### Model structure
- refer to [utils/model.py](utils/model.py)
- We implement a most basic RoBERTa-base model to extract event representation (the average last hidden states of each token)
- For each pair of events, event representations are concatenated as pair representation and a stack of linear layers is used for classification.

## How to run
- [Run coreference](coreference/README.md)
- [Run temporal](temporal/README.md)
- [Run causal](causal/README.md)
- [Run subevent](subevent/README.md)
- [Run joint](joint/README.md)


## Citation
TBD

## Contact
- wangxz20@mails.tsinghua.edu.cn
- yl-chen21@mails.tsinghua.edu.cn
