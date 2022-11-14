import json
from torch.utils.data import DataLoader, Dataset
import torch
import os
from tqdm import tqdm
import random 
import torch.nn.functional as F

REL2ID = {
    "SUBEVENT": 0,
    "NONE": 1,
}

ID2REL = {v:k for k, v in REL2ID.items()}

def valid_split(point, spans):
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
    def __init__(self, data):
        self.id = data["id"]
        self.words = data["tokens"]
        self.events = []
        self.eid2mentions = {}
        if "events" in data:
            for e in data["events"]:
                self.events += e["mention"]
                self.eid2mentions[e["id"]] = [m["id"] for m in e["mention"]]
            self.relations = data["subevent_relations"]
        else:
            self.events = data["event_mentions"]
            self.relations = []
        self.sort_events()
        self.get_labels()
    
    def sort_events(self):
        self.events = sorted(self.events, key=lambda x: (x["sent_id"], x["offset"][0]))
        self.sorted_event_spans = [(event["sent_id"], event["offset"]) for event in self.events]

    def get_labels(self):
        pair2rel = {}
        for pair in self.relations:
            for e1 in self.eid2mentions[pair[0]]:
                    for e2 in self.eid2mentions[pair[1]]:
                        pair2rel[(e1, e2)] = REL2ID["SUBEVENT"]

        self.labels = []
        for e1 in self.events:
            for e2 in self.events:
                if e1["id"] == e2["id"]:
                    continue
                self.labels.append(pair2rel.get((e1["id"], e2["id"]), REL2ID["NONE"]))
        assert len(self.labels) == len(self.events) ** 2 - len(self.events)

class myDataset(Dataset):
    def __init__(self, tokenizer, data_dir, split, max_length=512, sample_rate=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.load_examples(data_dir, split)
        if sample_rate is not None:
            self.examples = list(random.sample(self.examples, int(sample_rate * len(self.examples))))
        self.tokenize()
        self.to_tensor()
    
    def load_examples(self, data_dir, split):
        self.examples = []
        with open(os.path.join(data_dir, f"{split}.jsonl"))as f:
            lines = f.readlines()
        for line in lines:
            data = json.loads(line.strip())
            doc = Document(data)
            if doc.sorted_event_spans:
                self.examples.append(doc)
    
    def tokenize(self):
        # {input_ids, event_spans, event_group}
        self.tokenized_samples = []
        for example in tqdm(self.examples, desc="tokenizing"):
            event_spans = [] # [[(start, end)], [],...]
            input_ids = [] # [[], [], ...]
            labels = example.labels
            spans = example.sorted_event_spans
            words = example.words
            event_id = 0
            sub_input_ids = [self.tokenizer.cls_token_id]
            sub_event_spans = []
            for sent_id, word in enumerate(words):
                i = 0
                tmp_event_spans = []
                tmp_input_ids = []
                # add special tokens for event
                while event_id < len(spans) and spans[event_id][0] == sent_id:
                    sp = spans[event_id]
                    if i < sp[1][0]:
                        context_ids = self.tokenizer(word[i:sp[1][0]], is_split_into_words=True, add_special_tokens=False)["input_ids"]
                        tmp_input_ids += context_ids
                    event_ids = self.tokenizer(word[sp[1][0]:sp[1][1]], is_split_into_words=True, add_special_tokens=False)["input_ids"]
                    start = len(tmp_input_ids)
                    end = len(tmp_input_ids) + len(event_ids)
                    tmp_event_spans.append((start, end))
                    tmp_input_ids += event_ids
                    i = sp[1][1]
                    event_id += 1
                if word[i:]:
                    tmp_input_ids += self.tokenizer(word[i:], is_split_into_words=True, add_special_tokens=False)["input_ids"]
                
                # add SEP between sentences
                tmp_input_ids.append(self.tokenizer.sep_token_id)

                if len(sub_input_ids) + len(tmp_input_ids) <= self.max_length:
                    sub_event_spans += [(sp[0]+len(sub_input_ids), sp[1]+len(sub_input_ids)) for sp in tmp_event_spans]
                    sub_input_ids += tmp_input_ids
                else:
                    assert len(sub_input_ids) <= self.max_length
                    input_ids.append(sub_input_ids)
                    event_spans.append(sub_event_spans)
                    while len(tmp_input_ids) >= self.max_length:
                        split_point = self.max_length - 1
                        while not valid_split(split_point, tmp_event_spans):
                            split_point -= 1
                        tmp_event_spans_part1, tmp_event_spans = split_spans(split_point, tmp_event_spans)
                        tmp_input_ids_part1, tmp_input_ids = tmp_input_ids[:split_point], tmp_input_ids[split_point:]
                        input_ids.append([self.tokenizer.cls_token_id] + tmp_input_ids_part1)
                        event_spans.append([(sp[0]+1, sp[1]+1) for sp in tmp_event_spans_part1])
                        tmp_event_spans = [(sp[0]-len(tmp_input_ids_part1), sp[1]-len(tmp_input_ids_part1)) for sp in tmp_event_spans]

                    sub_event_spans = [(sp[0]+1, sp[1]+1) for sp in tmp_event_spans]
                    sub_input_ids = [self.tokenizer.cls_token_id] + tmp_input_ids
            if sub_input_ids:
                input_ids.append(sub_input_ids)
                event_spans.append(sub_event_spans)
            
            assert event_id == len(spans)
                
            tokenized = {"input_ids": input_ids, "attention_mask": None, "event_spans": event_spans, "labels": labels, "doc_id": example.id}
            self.tokenized_samples.append(tokenized)
    
    def to_tensor(self):
        for item in self.tokenized_samples:
            attention_mask = []
            for ids in item["input_ids"]:
                mask = [1] * len(ids)
                while len(ids) < self.max_length:
                    ids.append(self.tokenizer.pad_token_id)
                    mask.append(0)
                attention_mask.append(mask)
            item["input_ids"] = torch.LongTensor(item["input_ids"])
            item["attention_mask"] = torch.LongTensor(attention_mask)
            item["labels"] = torch.LongTensor(item["labels"])
    
    def __getitem__(self, index):
        return self.tokenized_samples[index]

    def __len__(self):
        return len(self.tokenized_samples)



def collator(data):
    collate_data = {"input_ids": [], "attention_mask": [], "event_spans": [], "labels": [], "splits": [0], "doc_id": []}
    for d in data:
        for k in d:
            collate_data[k].append(d[k])
    lengths = [ids.size(0) for ids in collate_data["input_ids"]]
    for l in lengths:
        collate_data["splits"].append(collate_data["splits"][-1]+l)
    collate_data["input_ids"] = torch.cat(collate_data["input_ids"])
    collate_data["attention_mask"] = torch.cat(collate_data["attention_mask"])
    max_label_length = max([len(label) for label in collate_data["labels"]])
    collate_data["labels"] = torch.stack([F.pad(label, pad=(0, max_label_length-len(label)), value=-100) for label in collate_data["labels"]])
    collate_data["max_label_length"] = max_label_length
    return collate_data

def get_dataloader(tokenizer, split, data_dir="../data", max_length=128, batch_size=8, shuffle=True, sample_rate=None):
    dataset = myDataset(tokenizer, data_dir, split, max_length=max_length, sample_rate=sample_rate)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collator)

if __name__ == "__main__":
    from transformers import RobertaTokenizer
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    dataloader = get_dataloader(tokenizer, "test", shuffle=False, max_length=256)
    for data in dataloader:
        print(data["input_ids"].size())
        print(data["attention_mask"].size())
        print(data["labels"])        