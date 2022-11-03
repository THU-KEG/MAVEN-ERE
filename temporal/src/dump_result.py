import json
import os
from pathlib import Path 

REL2ID = {
    "before": 0,
    "overlap": 1,
    "contains": 2,
    "simultaneous": 3,
    "ends-on": 4,
    "begins-on": 5,
    "NONE": 6,
}

ID2REL = {v:k for k, v in REL2ID.items()}

def valid_split(point, spans):
    # retain context of at least 3 tokens
    for sp in spans:
        if point > sp[0] - 3 and point <= sp[1] + 3:
            return False
    return True

def split_spans(point, spans):
    part1 = []
    part2 = []
    i = 0
    for sp in spans:
        if sp[1] < point:
            part1.append(sp)
            i += 1
        else:
            break
    part2 = spans[i:]
    return part1, part2


def type_tokens(type_str):
    return [f"<{type_str}>", f"<{type_str}/>"]

class Document:
    def __init__(self, data, ignore_nonetype=False):
        self.id = data["doc"]["id"]
        self.words = data["doc"]["tokens"]
        self.events = []
        self.eid2mentions = {}
        self.events = data["candidates"] + data["TIMEX3"]

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
            "pairs": []
        }
        assert len(example.all_pairs) == len(pred_rels)
        for i, pair in enumerate(example.all_pairs):
            item["pairs"].append({
                "e1": pair[0],
                "e2": pair[1],
                "pred_relation": ID2REL[int(pred_rels[i])],
            })        
        final_results.append(item)
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    with open(os.path.join(save_dir, "temporal_prediction.json"), "w")as f:
        f.writelines("\n".join([json.dumps(item) for item in final_results]))


if __name__ == "__main__":
    preds = json.load(open("../output/results_Test.json"))
    dump_result("../data/test.json", preds, "../output/dump")