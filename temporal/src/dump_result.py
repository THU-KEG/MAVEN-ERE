import json
import os
from pathlib import Path 

REL2ID = {
    "BEFORE": 0,
    "OVERLAP": 1,
    "CONTAINS": 2,
    "SIMULTANEOUS": 3,
    "ENDS-ON": 4,
    "BEGINS-ON": 5,
    "NONE": 6,
}

ID2REL = {v:k for k, v in REL2ID.items()}

class Document:
    def __init__(self, data, ignore_nonetype=False):
        self.id = data["id"]
        self.words = data["tokens"]
        self.events = []
        self.eid2mentions = {}
        self.events = data["event_mentions"] + data["TIMEX"]

        self.sort_events()
        self.get_pairs(ignore_nonetype)
    
    def sort_events(self):
        self.events = sorted(self.events, key=lambda x: (x["sent_id"], x["offset"][0]))

    def get_pairs(self, ignore_nonetype):
        self.all_pairs = []
        for e1 in self.events:
            for e2 in self.events:
                if e1["id"] == e2["id"]:
                    continue
                if ignore_nonetype:
                    if abs(e1["sent_id"] - e2["sent_id"]) > 1:
                        continue
                self.all_pairs.append((e1["id"], e2["id"]))


def dump_result(input_path, preds, save_dir, ignore_nonetype=True):
    # load examples 
    examples = []
    with open(os.path.join(input_path))as f:
        lines = f.readlines()
    for line in lines:
        data = json.loads(line.strip())
        doc = Document(data, ignore_nonetype=ignore_nonetype)
        if doc.events:
            examples.append(doc)
    # each item is the clusters of a document
    final_results = []
    for example, pred_per_doc in zip(examples, preds):
        assert example.id == pred_per_doc["doc_id"]
        pred_rels = pred_per_doc["preds"]
        item = {
            "id": example.id,
            "temporal_relations": {"BEFORE":[],"OVERLAP":[],"CONTAINS":[],"SIMULTANEOUS":[],"ENDS-ON":[],"BEGINS-ON":[]}
        }
        assert len(example.all_pairs) == len(pred_rels)
        for i, pair in enumerate(example.all_pairs):
            if ID2REL[int(pred_rels[i])]!="NONE":
                item["temporal_relations"][ID2REL[int(pred_rels[i])]].append([pair[0],pair[1]])
        final_results.append(item)
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    with open(os.path.join(save_dir, "test_prediction.jsonl"), "w")as f:
        f.writelines("\n".join([json.dumps(item) for item in final_results]))


if __name__ == "__main__":
    preds = json.load(open("../output/results_Test.json"))
    dump_result("../data/test.json", preds, "../output/dump")