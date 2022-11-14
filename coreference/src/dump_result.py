import os
import json

from pathlib import Path 

class Document:
    def __init__(self, data):
        self.id = data["id"]
        self.words = data["tokens"]
        self.events = data["event_mentions"]
        self.events = sorted(self.events, key=lambda x: (x["sent_id"], x["offset"][0]))

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
    # each item is the clusters of a document
    final_results = []
    for example, pred_per_doc in zip(examples, preds):
        assert example.id == pred_per_doc["doc_id"]
        clusters = pred_per_doc["clusters"]
        item = {
            "id": example.id,
            "coreference": [],
        }
        events = example.events
        for cluster in clusters:
            item["coreference"].append([
                events[c]["id"] for c in cluster
            ])
        final_results.append(item)
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    with open(os.path.join(save_dir, "test_prediction.jsonl"), "w")as f:
        f.writelines("\n".join([json.dumps(item) for item in final_results]))


if __name__ == "__main__":
    preds = json.load(open("../output/results_Test.json"))
    dump_result("../data/test.json", preds, "../output/dump")
