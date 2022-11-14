import os
import json

from pathlib import Path 

class CorefDocument:
    def __init__(self, data):
        self.id = data["id"]
        self.words = data["tokens"]
        self.events = data["event_mentions"]
        self.events = sorted(self.events, key=lambda x: (x["sent_id"], x["offset"][0]))

class Document:
    def __init__(self, data, temporal=False):
        self.id = data["id"]
        self.words = data["tokens"]
        self.events = data["event_mentions"]
        if temporal:
            self.events += data['TIMEX']
        self.sort_events()
        self.get_pairs()
    
    def sort_events(self):
        self.events = sorted(self.events, key=lambda x: (x["sent_id"], x["offset"][0]))

    def get_pairs(self):
        self.all_pairs = []
        for e1 in self.events:
            for e2 in self.events:
                if e1["id"] == e2["id"]:
                    continue
                self.all_pairs.append((e1["id"], e2["id"]))

SUBEVENT_REL2ID = {
    "NONE": 0,
    "subevent": 1
}

SUBEVENT_ID2REL = {v:k for k, v in SUBEVENT_REL2ID.items()}

def subevent_dump(input_path, preds, all_results):
    examples = []
    with open(os.path.join(input_path))as f:
        lines = f.readlines()
    for line in lines:
        data = json.loads(line.strip())
        doc = Document(data)
        if doc.events:
            examples.append(doc)
    for example, pred_per_doc in zip(examples, preds):
        assert example.id == pred_per_doc["doc_id"]
        pred_rels = pred_per_doc["preds"]
        assert len(example.all_pairs) == len(pred_rels)
        if example.id not in all_results:
            all_results[example.id] = {"id": example.id, "coreference":[], "causal_relations": {"CAUSE":[],"PRECONDITION":[]}, "temporal_relations": {"BEFORE":[],"OVERLAP":[],"CONTAINS":[],"SIMULTANEOUS":[],"ENDS-ON":[],"BEGINS-ON":[]}, "subevent_relations": []}
        for i, pair in enumerate(example.all_pairs):
            if int(pred_rels[i])!=0:
                all_results[example.id]["subevent_relations"].append([pair[0], pair[1]])

CAUSAL_REL2ID = {
    "NONE": 0,
    "PRECONDITION": 1,
    "CAUSE": 2
}

CAUSAL_ID2REL = {v:k for k, v in CAUSAL_REL2ID.items()}

def causal_dump(input_path, preds, all_results):
    examples = []
    with open(os.path.join(input_path))as f:
        lines = f.readlines()
    for line in lines:
        data = json.loads(line.strip())
        doc = Document(data)
        if doc.events:
            examples.append(doc)
    for example, pred_per_doc in zip(examples, preds):
        assert example.id == pred_per_doc["doc_id"]
        pred_rels = pred_per_doc["preds"]
        assert len(example.all_pairs) == len(pred_rels)
        if example.id not in all_results:
            all_results[example.id] = {"id": example.id, "coreference":[], "causal_relations": {"CAUSE":[],"PRECONDITION":[]}, "temporal_relations": {"BEFORE":[],"OVERLAP":[],"CONTAINS":[],"SIMULTANEOUS":[],"ENDS-ON":[],"BEGINS-ON":[]}, "subevent_relations": []}
        for i, pair in enumerate(example.all_pairs):
            if int(pred_rels[i])==2:
                all_results[example.id]['causal_relations']['CAUSE'].append([pair[0], pair[1]])
            elif int(pred_rels[i])==1:
                all_results[example.id]['causal_relations']['PRECONDITION'].append([pair[0], pair[1]])

def coref_dump(input_path, preds, all_results):
    examples = []
    with open(os.path.join(input_path))as f:
        lines = f.readlines()
    for line in lines:
        data = json.loads(line.strip())
        doc = Document(data)
        if doc.events:
            examples.append(doc)
    for example, pred_per_doc in zip(examples, preds):
        assert example.id == pred_per_doc["doc_id"]
        clusters = pred_per_doc["clusters"]
        if example.id not in all_results:
            all_results[example.id] = {"id": example.id, "coreference":[], "causal_relations": {"CAUSE":[],"PRECONDITION":[]}, "temporal_relations": {"BEFORE":[],"OVERLAP":[],"CONTAINS":[],"SIMULTANEOUS":[],"ENDS-ON":[],"BEGINS-ON":[]}, "subevent_relations": []}
        events = example.events
        for cluster in clusters:
            all_results[example.id]["coreference"].append([events[c]["id"] for c in cluster])

TEMP_REL2ID = {
    "BEFORE": 0,
    "OVERLAP": 1,
    "CONTAINS": 2,
    "SIMULTANEOUS": 3,
    "ENDS-ON": 4,
    "BEGINS-ON": 5,
    "NONE": 6,
}

TEMP_ID2REL = {v:k for k, v in TEMP_REL2ID.items()}

def temporal_dump(input_path, preds, all_results):
    examples = []
    with open(os.path.join(input_path))as f:
        lines = f.readlines()
    for line in lines:
        data = json.loads(line.strip())
        doc = Document(data, temporal=True)
        if doc.events:
            examples.append(doc)
    for example, pred_per_doc in zip(examples, preds):
        assert example.id == pred_per_doc["doc_id"]
        pred_rels = pred_per_doc["preds"]
        assert len(example.all_pairs) == len(pred_rels)
        if example.id not in all_results:
            all_results[example.id] = {"id": example.id, "coreference":[], "causal_relations": {"CAUSE":[],"PRECONDITION":[]}, "temporal_relations": {"BEFORE":[],"OVERLAP":[],"CONTAINS":[],"SIMULTANEOUS":[],"ENDS-ON":[],"BEGINS-ON":[]}, "subevent_relations": []}
        for i, pair in enumerate(example.all_pairs):
            if int(pred_rels[i])==6:
                continue
            all_results[example.id]["temporal_relations"][TEMP_ID2REL[int(pred_rels[i])]].append([pair[0], pair[1]])