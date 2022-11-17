# MAVEN-ERE
Source code and dataset for EMNLP 2022 paper ["MAVEN-ERE: A Unified Large-scale Dataset for Event Coreference, Temporal, Causal, and Subevent Relation Extraction"](https://arxiv.org/abs/2211.07342).

## Overview
- MAVEN-ERE is the first unified large-scale dataset for events relation extraction. It contains 4480 documents, with 103,193 coreference chains, 1,216,217 temporal relations, 15,841 subevent relations, and 57,992 causal relations. 
  - Temporal relations include: `BEFORE`, `OVERLAP`, `CONTAINS`, `SIMULTANEOUS`, `BEGINS-ON`, and `ENDS-ON`
  - Causal relations include: `CAUSE` and `PRECONDITION`
- The annotation rule of above relations mainly follows [RED guideline](https://github.com/timjogorman/RicherEventDescription/blob/master/guidelines.md).
- For more details, please see [the paper](https://arxiv.org/abs/2211.07342).
![illustration](https://user-images.githubusercontent.com/55496186/196908089-e240d099-cdfd-4286-9046-e7e9d764562e.png)

## Requirements
```shell
pip install -r requirements.txt
```

## MAVEN-ERE Dataset

### Get the data
The dataset (ver. 1.0) can be obtained from [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/a7d1db6c44ea458bb6f0/?dl=1) or [Google Drive](https://drive.google.com/file/d/1fxomYO6zPl5DDrDr_HeWFK14s8BpW1z-/view?usp=sharing).

You can also download `MAVEN-ERE` and other processed datasets used in our experiments with the scripts in `/data`:
```bash
cd data
bash download_maven.sh
bash download.sh
cd ../
```

### Data Format
Each `.jsonl` file is a subset of `MAVEN-ERE` and each line in the files is a json string for a document. For the `train.jsonl` and `valid.jsonl`, the json format sample is as below:

```JSON5
{
  "id": "f28bce270df5a122c09365002d247e76", // an unique string for each document
  "title": "United States occupation of Nicaragua", // the tiltle of the document
  "tokens": [ // a list for tokenized document content. each item is a tokenized sentence
    ["The", "United", "States", "occupation", "of", "Nicaragua", "from", 
    "1912", "to", "1933", "was", "part", "of", "the", "Banana", "Wars", ",", 
    "when", "the", "US", "military", "intervened", "in", "various", 
    "Latin", "American", "countries", "from", "1898", "to", "1934", "."],
  ],
  "sentences": [ // untokenized sentences of the document. each item is a sentence (string)
      "The United States occupation of Nicaragua from 1912 to 1933 was part of the Banana Wars, when the US military intervened in various Latin American countries from 1898 to 1934.",
  ],
  "events": [ // a list for annotated events, each item is a dict for an event (coreference chain)
        {
            "id": "EVENT_0341c5cced5545ffe7c543b7a155bfa8", // an unique string for the event (coreference chain)
            "type": "Choosing", // the event type
            "type_id": 25, // the numerical id for the event type, consistent with MAVEN
            "mention": [ // a list for the coreferential event mentions of the chain, each item is a dict. they have coreference relations to each other
                {
                    "id": "a75ba55cadad23555a0ffc9454088687", // an unique string for the event mention
                    "trigger_word": "assumed", // a string of the trigger word or phrase
                    "sent_id": 3, // the index of the corresponding sentence, starts with 0
                    "offset": [1, 2] // the offset of the trigger words in the tokens list
                }
            ]
        }
  ],
  "TIMEX": [ // a list for annotated temporal expressions (TIMEX), each item is a dict for a TIMEX
    {
      "id": "TIME_833e41f3304210094101eca59905055e", // an unique string for the TIMEX
      "mention": "1912", // a string of the mention of the TIMEX
      "type": "DATE", // the type of the TIMEX
      "sent_id": 0, // the index of the corresponding sentence, starts with 0
      "offset": [7, 8] // the offset of the trigger words in the tokens list
    },
  ],
  "temporal_relations": { // a list for annotated temporal relations between events (and TIMEXs)
    "BEFORE": [ // a list for temporal relations of BEFORE type
      ["EVENT_id_1", "TIME_id_2"] // a temporal relation instance, means EVENT_id_1 BEFORE TIME_id_2
    ],
    "OVERLAP": [ // all the following types are similar
      ["EVENT_id_1", "EVENT_id_2"]
    ],
    "CONTAINS": [
      ["EVENT_id_1", "EVENT_id_2"]
    ],
    "SIMULTANEOUS": [
      ["EVENT_id_1", "EVENT_id_2"]
    ],
    "ENDS-ON": [
      ["EVENT_id_1", "EVENT_id_2"]
    ],
    "BEGINS-ON": [
      ["EVENT_id_1", "EVENT_id_2"]
    ],
  },
  "causal_relations": { // a list for annotated causal relations between events
    "CAUSE": [  // a list for causal relations of CAUSE type
      ["EVENT_id_1", "EVENT_id_2"] // a causal relation instance, means EVENT_id_1 CAUSE EVENT_id_2
    ],
    "PRECONDITION": [ // the PRECONDITION type is similar
      ["EVENT_id_1", "EVENT_id_2"]
    ],
  },
  "subevent_relations": [ // a list for annotated subevent relations between events
    ["EVENT_id_1", "EVENT_id_2"] // a subevent relation instance, means EVENT_id_2 is a subevent of EVENT_id_1
  ]
}
```

- For the `test.jsonl`, the format is similar but we hide the ground truth annotations to organize a fair evaluation challenge. To get evaluation results on the test set, you need to submit the prediction results to our [CodaLab competition](). 
- To avoid leak the test set of the original MAVEN event detection challenge, the candidate event mentions we offered here have more than the golden annotations. But we only evaluate your predictions for the golden event mentions.

```JSON5
{
  "id": "f28bce270df5a122c09365002d247e76", // an unique string for each document
  "title": "United States occupation of Nicaragua", // the tiltle of the document
  "tokens": [ // a list for tokenized document content. each item is a tokenized sentence
    ["The", "United", "States", "occupation", "of", "Nicaragua", "from", 
    "1912", "to", "1933", "was", "part", "of", "the", "Banana", "Wars", ",", 
    "when", "the", "US", "military", "intervened", "in", "various", 
    "Latin", "American", "countries", "from", "1898", "to", "1934", "."],
  ],
  "sentences": [ // untokenized sentences of the document. each item is a sentence (string)
      "The United States occupation of Nicaragua from 1912 to 1933 was part of the Banana Wars, when the US military intervened in various Latin American countries from 1898 to 1934.",
  ],
  "event_mentions": [ // a list for event mentions (and distractors), you need to predict the relations between the given mentions
    {
      "id": "a75ba55cadad23555a0ffc9454088687", // an unique string for the event mention
      "trigger_word": "assumed", // a string of the trigger word or phrase
      "sent_id": 3, // the index of the corresponding sentence, starts with 0
      "offset": [1, 2], // the offset of the trigger words in the tokens list
      "type": "Choosing", // the event type
      "type_id": 25, // the numerical id for the event type, consistent with MAVEN
    }
  ],
  "TIMEX": [ // a list for annotated temporal expressions (TIMEX), each item is a dict for a TIMEX
    {
      "id": "TIME_833e41f3304210094101eca59905055e", // an unique string for the TIMEX
      "mention": "1912", // a string of the mention of the TIMEX
      "type": "DATE", // the type of the TIMEX
      "sent_id": 0, // the index of the corresponding sentence, starts with 0
      "offset": [7, 8] // the offset of the trigger words in the tokens list
    }
  ]
}
```

## How to run experiments
- [Coreference Relation Experiments](coreference/README.md)
- [Temporal Relation Experiments](temporal/README.md)
- [Causal Relation Experiments](causal/README.md)
- [Subevent Relation Experiments](subevent/README.md)
- [Joint Training Experiments](joint/README.md)

## Get Test Results from CodaLab
To get the test results, you can submit your predictions to our permanent [CodaLab competition](https://codalab.lisn.upsaclay.fr/competitions/8691#learn_the_details).

You need to name your result file as `test_prediction.jsonl` and compress it into zip format file named `submission.zip` for submission. Each line in the submission file should be a `json` string encoding the prediction results for one document. The json format is as below:

```JSON5
{
  "id": "f28bce270df5a122c09365002d247e76", // an unique string for each document, mandatory
  "coreference": [ // a list for predicted coreference clusters, each item is a cluster of event mentions having coreference relations with each other
      ["a75ba55cadad23555a0ffc9454088687", "555a0ffc9454a75ba08868755cadad23"] // a list for a predicted cluster, each item is the id of an event mention
  ],
  "temporal_relations": { // a list for predicted temporal relations between event mentions (not events) and TIMEXs
    "BEFORE": [ // a list for predicted temporal relations of BEFORE type
      ["a75ba55cadad23555a0ffc9454088687", "555a0ffc9454a75ba08868755cadad23"] // a temporal relation instance, its items shall be id of event mentions or TIMEXs
    ],
    "OVERLAP": [ // all the following types are similar
      ["a75ba55cadad23555a0ffc9454088687", "TIME_id_1"]
    ],
    "CONTAINS": [
      ["a75ba55cadad23555a0ffc9454088687", "555a0ffc9454a75ba08868755cadad23"]
    ],
    "SIMULTANEOUS": [
      ["a75ba55cadad23555a0ffc9454088687", "555a0ffc9454a75ba08868755cadad23"]
    ],
    "ENDS-ON": [
      ["a75ba55cadad23555a0ffc9454088687", "555a0ffc9454a75ba08868755cadad23"]
    ],
    "BEGINS-ON": [
      ["555a0ffc9454a75ba08868755cadad23", "TIME_id_2"]
    ],
  },
  "causal_relations": { // a list for predicted causal relations between event mentions (not events)
    "CAUSE": [  // a list for causal relations of CAUSE type
      ["a75ba55cadad23555a0ffc9454088687", "555a0ffc9454a75ba08868755cadad23"] // a causal relation instance, its items shall be id of event mentions
    ],
    "PRECONDITION": [ // the PRECONDITION type is similar
      ["a75ba55cadad23555a0ffc9454088687", "555a0ffc9454a75ba08868755cadad23"]
    ],
  },
  "subevent_relations": [ // a list for predicted subevent relations between event mention (not events)
    ["a75ba55cadad23555a0ffc9454088687", "555a0ffc9454a75ba08868755cadad23"] // a subevent relation instance, its items shall be id of event mentions
  ]
}
```

For the detailed implementations of our evaluations, please refer to the [evaluation script](evaluate.py).

## Citation
```bibtex
@inproceedings{wang-chen-etal2022MAVENERE,
  title = {MAVEN-ERE: A Unified Large-scale Dataset for Event Coreference, Temporal, Causal, and Subevent Relation Extraction},
  author = {Xiaozhi Wang and Yulin Chen and Ning Ding and Hao Peng and Zimu Wang and Yankai Lin and Xu Han and Lei Hou and Juanzi Li and Zhiyuan Liu and Peng Li and Jie Zhou},
  booktitle = {Proceedings of EMNLP},
  year = {2022},
}
```
## Contact
- wangxz20@mails.tsinghua.edu.cn
- yl-chen21@mails.tsinghua.edu.cn
