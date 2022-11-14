import json
from torch.utils.data import DataLoader, Dataset
import torch
import os
from tqdm import tqdm
import torch.nn.functional as F

REL2ID_DICT = { # always put None type at the last
    "matres": {
        "BEFORE": 0,
        "AFTER": 1,
        "EQUAL": 2,
        "VAGUE": 3,
        "NONE": 4,
    },
    "tb-dense": {
        "BEFORE": 0,
        "AFTER": 1,
        "INCLUDES": 2,
        "IS_INCLUDED": 3,
        "SIMULTANEOUS": 4,
        "VAGUE": 5,
    },
    "tcr": {
        "BEFORE": 0,
        "AFTER": 1,
        "SIMULTANEOUS": 2,
        "VAGUE": 3,
        "NONE": 4,
    }
}

INVERSE_REL_DICT = {
    "matres": {
        "BEFORE": "AFTER",
        "AFTER": "BEFORE"
    },
    "tb-dense": {
        "BEFORE": "AFTER",
        "AFTER": "BEFORE",
        "INCLUDES": "IS_INCLUDED",
        "IS_INCLUDED": "INCLUDES"
    }
}

INVERSE_REL_DICT["tcr"] = INVERSE_REL_DICT["matres"]


BIDIRECTIONAL_REL_DICT = {
    "matres": ["EQUAL"],
    "tb-dense": ["SIMULTANEOUS"],
    "tcr": ["SIMULTANEOUS"]
}

NONE_REL_DICT = {
    "matres": "NONE",
    "tb-dense": "VAGUE"
}

NONE_REL_DICT["tcr"] = NONE_REL_DICT["matres"]


EVAL_EXCLUDE = {
    "matres": ["NONE", "VAGUE"],
    "tb-dense": ["VAGUE"]
}

EVAL_EXCLUDE["tcr"] = EVAL_EXCLUDE["matres"]

ERROR = 0

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

class Document:
    def __init__(self, data, dataname, ignore_nonetype=False, istrain=True):
        self.id = data["fid"]
        self.text = data["text"]
        self.events = data["events"] 
        if dataname.lower() == "tb-dense":
            self.events += data["timexes"]
        self.relations = data["relations"]
        self.dataname = dataname.lower()
        
        self.sort_events()
        self.get_labels(ignore_nonetype, istrain)
    
    def sort_events(self):
        self.events = sorted(self.events, key=lambda x: (x["sent_id"], x["offset"][0]))
        self.sorted_event_spans = [(event["sent_id"], event["offset"]) for event in self.events]

    def get_labels(self, ignore_nonetype, istrain):
        def get_eid(e):
            if "eid" in e:
                return e["eid"]
            else:
                return e["tid"]
        
        global ERROR
        rel2id = REL2ID_DICT[self.dataname]
        inverse_rel = INVERSE_REL_DICT[self.dataname]
        bidirectional_rel = BIDIRECTIONAL_REL_DICT[self.dataname]
        none_type = NONE_REL_DICT[self.dataname]
        pair2rel = {}
        for rel in self.relations:
            for pair in self.relations[rel]:
                pair2rel[tuple(pair)] = rel2id[rel]
                if not ignore_nonetype: # in eval, don't expand inferred relations
                    if rel in bidirectional_rel:
                        pair2rel[(pair[1], pair[0])] = rel2id[rel]
                    elif rel in inverse_rel:
                        pair2rel[(pair[1], pair[0])] = rel2id[inverse_rel[rel]]
            
        self.labels = []
        for i, e1 in enumerate(self.events):
            for j, e2 in enumerate(self.events):
                if i == j:
                    continue
                if self.dataname in ["matres", "tcr"]:
                    if "eiid" not in e1 or "eiid" not in e2: # some events don't have eiid
                        self.labels.append(-100)
                    else:
                        if e1["eiid"] == e2["eiid"]: # some events in tb-dense share same eiid
                            ERROR += 1
                        if ignore_nonetype:
                            self.labels.append(pair2rel.get((e1["eiid"], e2["eiid"]), -100))
                        else:
                            self.labels.append(pair2rel.get((e1["eiid"], e2["eiid"]), rel2id[none_type]))
                elif self.dataname in ["tb-dense"]:
                    e1_id = get_eid(e1)
                    e2_id = get_eid(e2)
                    if ignore_nonetype:
                        self.labels.append(pair2rel.get((e1_id, e2_id), -100))
                    else:
                        self.labels.append(pair2rel.get((e1_id, e2_id), rel2id[none_type]))
        assert len(self.labels) == len(self.events) ** 2 - len(self.events), print(len(self.labels), len(self.events))



class myDataset(Dataset):
    def __init__(self, tokenizer, data_dir, split, dataname, max_length=512, ignore_nonetype=False, istrain=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ignore_nonetype = ignore_nonetype
        self.istrain = istrain
        self.load_examples(data_dir, split, dataname)
        self.examples = self.examples
        self.tokenize()
        self.to_tensor()
    
    def load_examples(self, data_dir, split, dataname):
        self.examples = []
        with open(os.path.join(data_dir, f"{split}.json"))as f:
            lines = f.readlines()
        for line in lines:
            data = json.loads(line.strip())
            doc = Document(data, dataname, ignore_nonetype=self.ignore_nonetype, istrain=self.istrain)
            if doc.sorted_event_spans:
                self.examples.append(doc)
    
    def tokenize(self):
        self.tokenized_samples = []
        for example in tqdm(self.examples, desc="tokenizing"):
            event_spans = [] # [[(start, end)], [],...]
            input_ids = [] # [[], [], ...]

            labels = example.labels
            spans = example.sorted_event_spans
            text = example.text
            event_id = 0
            sub_input_ids = [self.tokenizer.cls_token_id]
            sub_event_spans = []
            for sent_id, sent in enumerate(text):
                i = 0
                tmp_event_spans = []
                tmp_input_ids = []
                # add special tokens for event
                while event_id < len(spans) and spans[event_id][0] == sent_id:
                    sp = spans[event_id]
                    if i < sp[1][0]: # TODO add white space to handle roberta tokenizer
                        context_ids = self.tokenizer(sent[i:sp[1][0]], add_special_tokens=False)["input_ids"]
                        tmp_input_ids += context_ids
                    event_ids = self.tokenizer(sent[sp[1][0]:sp[1][1]], add_special_tokens=False)["input_ids"]
                    start = len(tmp_input_ids) # not adding special tokens for now
                    end = len(tmp_input_ids) + len(event_ids)
                    tmp_event_spans.append((start, end))
                    tmp_input_ids += event_ids

                    i = sp[1][1]
                    event_id += 1
                if sent[i:]:
                    tmp_input_ids += self.tokenizer(sent[i:], add_special_tokens=False)["input_ids"]

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
                
            tokenized = {"input_ids": input_ids, "attention_mask": None, "event_spans": event_spans, "labels": labels}
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
    collate_data = {"input_ids": [], "attention_mask": [], "event_spans": [], "labels": [], "splits": [0]}
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
    return collate_data

def get_dataloader(tokenizer, split, dataname, data_dir="../data", max_length=128, batch_size=8, shuffle=True, ignore_nonetype=False, istrain=True):
    dataset = myDataset(tokenizer, data_dir, split, dataname, max_length=max_length, ignore_nonetype=ignore_nonetype, istrain=istrain)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collator)

if __name__ == "__main__":
    from transformers import RobertaTokenizer
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    dataloader = get_dataloader(tokenizer, "train", "TB-Dense", data_dir = "../../data/processed/TB-Dense", shuffle=False, max_length=512)
    print(ERROR)
    for data in dataloader:
        print(data["input_ids"].size())
        text = tokenizer.convert_ids_to_tokens(data["input_ids"][0])
        print(text)
        print(data["event_spans"][0])
        for span in data["event_spans"][0][0]:
            print(span)
            print(tokenizer.convert_tokens_to_string(text[span[0]:span[1]]))

        print(data["attention_mask"].size())
        print(data["labels"])
        break
        