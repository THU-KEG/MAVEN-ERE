import json
import os
from pathlib import Path

REL2ID = {
    "SUBEVENT": 0,
    "NONE": 1,
}

ID2REL = {v:k for k, v in REL2ID.items()}

class Document:
    def __init__(self, data):
        self.id = data["id"]
        self.words = data["tokens"]
        self.events = data["event_mentions"]
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


def dump_result(input_path, preds, save_dir):
    # load examples 
    examples = []
    with open(os.path.join(input_path))as f:
        lines = f.readlines()
    for line in lines:
        data = json.loads(line.strip())
        doc = Document(data)
        if doc.events:
            examples.append(doc)
    final_results = []
    for example, pred_per_doc in zip(examples, preds):
        assert example.id == pred_per_doc["doc_id"]
        pred_rels = pred_per_doc["preds"]
        item = {
            "id": example.id,
            "subevent_relations": []
        }
        assert len(example.all_pairs) == len(pred_rels)
        for i, pair in enumerate(example.all_pairs):
            if ID2REL[int(pred_rels[i])]=="SUBEVENT":
                item["subevent_relations"].append([pair[0],pair[1]])
        final_results.append(item)
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    with open(os.path.join(save_dir, "test_prediction.jsonl"), "w")as f:
        f.writelines("\n".join([json.dumps(item) for item in final_results]))

if __name__ == "__main__":
    preds = json.load(open("../output/results_Test.json"))
    dump_result("../data/test.json", preds, "../output/dump")
