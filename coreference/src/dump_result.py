import os
import json

from pathlib import Path 



class Document:
    def __init__(self, data):
        self.id = data["doc"]["id"]
        self.words = data["doc"]["tokens"]
        self.events = data["candidates"]
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
            "pred_clusters": [],
        }
        events = example.events
        for cluster in clusters:
            item["pred_clusters"].append([
                events[c]["id"] for c in cluster
            ])
        final_results.append(item)
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    with open(os.path.join(save_dir, "coreference_prediction.json"), "w")as f:
        f.writelines("\n".join([json.dumps(item) for item in final_results]))


if __name__ == "__main__":
    preds = json.load(open("../output/results_Test.json"))
    dump_result("../data/test.json", preds, "../output/dump")
