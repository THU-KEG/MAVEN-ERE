# %%
import json
from torch.utils.data import DataLoader, Dataset
import torch
import os
from tqdm import tqdm
import torch.nn.functional as F
import random

SUBEVENTREL2ID = {
    "NONE": 0,
    "subevent": 1
}

COREFREL2ID = {
    "NONE": 0,
    "coref": 1
}

CAUSALREL2ID = {
    "NONE": 0,
    "PRECONDITION": 1,
    "CAUSE": 2
}

TEMPREL2ID = {
    "before": 0,
    "overlap": 1,
    "contains": 2,
    "simultaneous": 3,
    "ends-on": 4,
    "begins-on": 5,
    "NONE": 6,
}

BIDIRECTIONAL_REL = ["overlap", "simultaneous", "begins-on"]

ID2TEMPREL = {v:k for k, v in TEMPREL2ID.items()}
ID2CAUSALREL = {v:k for k, v in CAUSALREL2ID.items()}
ID2COREFREL = {v:k for k, v in COREFREL2ID.items()}
ID2SUBEVENTREL = {v:k for k, v in SUBEVENTREL2ID.items()}



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
        self.mentions = []
        self.events = [] # first mention + timex
        for e in data["events"]:
            e["mentions"][0]["eid"] = e["id"]
            self.events += e["mentions"]
            # self.mentions.append(e)
        # self.mentions = data["events"]["mentions"] + data["TIMEX3"]
        self.events += data["TIMEX3"]
        # for t in data["TIMEX3"]:
        #     self.events.append(t)

        # for expanded labels 
        self.eid2mentions = {}
        for e in data["events"]:
            self.eid2mentions[e["id"]] = e["mentions"]
        for t in data["TIMEX3"]:
            self.eid2mentions[t["id"]] = [t]

        self.temporal_relations = data["temporal_event_relation"]
        self.causal_relations = data["causal_relation"]
        self.subevent_relations = {"subevent": data["subevent_relation"]}
        self.coref_relations = self.load_coref_relations(data)

        # print(self.relations.keys())

        self.sort_events()
        # self.get_labels(ignore_nonetype)
        self.coref_labels = self.get_coref_labels(data)
        self.temporal_labels = self.get_relation_labels(self.temporal_relations, TEMPREL2ID, ignore_timex=False)
        self.causal_labels = self.get_relation_labels(self.causal_relations, CAUSALREL2ID, ignore_timex=True)
        self.subevent_labels = self.get_relation_labels(self.subevent_relations, SUBEVENTREL2ID, ignore_timex=True)

        # self.temporal_labels_eval = self.get_relation_labels_for_eval(self.temporal_relations, TEMPREL2ID, ignore_timex=False)
        # self.causal_labels_eval = self.get_relation_labels_for_eval(self.causal_relations, CAUSALREL2ID, ignore_timex=True)
        # self.subevent_labels_eval = self.get_relation_labels_for_eval(self.subevent_relations, SUBEVENTREL2ID, ignore_timex=True)
    
    def load_coref_relations(self, data):
        relations = {}
        for event in data["events"]:
            for mention1 in event["mentions"]:
                for mention2 in event["mentions"]:
                    relations[(mention1["id"], mention2["id"])] = COREFREL2ID["coref"]
        return relations
    
    def sort_events(self):
        self.events = sorted(self.events, key=lambda x: (x["sent_id"], x["offset"][0]))
        # self.event2id = {e["id"]:idx for idx, e in enumerate(self.events)} # sorted events to index
        self.sorted_event_spans = [(event["sent_id"], event["offset"]) for event in self.events]
    
    def get_coref_labels(self, data):
        label_group = []
        events_only = [e for e in self.events if not e["id"].startswith("TIME")]
        self.events_idx = [i for i, e in enumerate(self.events) if not e["id"].startswith("TIME")]
        mid2index = {e["id"]:i for i, e in enumerate(events_only)}
        for event in data["events"]:
            label_group.append([mid2index[m["id"]] for m in event["mentions"]])


        # labels = []
        # for i, e1 in enumerate(self.events):
        #     for j, e2 in enumerate(self.events):
        #         if e1["id"] == e2["id"]:
        #             continue
        #         if i > j:
        #             labels.append(-100)
        #             continue
        #         if e1["id"].startswith("TIMEX") or e2["id"].startswith("TIMEX"):
        #             labels.append(-100)
        #             continue
        #         labels.append(self.coref_relations.get((e1["id"], e2["id"]), COREFREL2ID["NONE"]))
                
        # assert len(labels) == len(self.events) ** 2 - len(self.events)
        return label_group

    # def get_relation_labels_for_eval(self, relations, REL2ID, ignore_timex=True):
    #     pair2rel = {}
    #     for rel in relations:
    #         for pair in relations[rel]:
    #             pair2rel[tuple(pair)] = REL2ID[rel]
    #             if rel in BIDIRECTIONAL_REL:
    #                 pair2rel[(pair[1], pair[0])] = REL2ID[rel]
    #     labels = []
    #     for i, e1 in enumerate(self.events):
    #         for j, e2 in enumerate(self.events):
    #             if e1["id"] == e2["id"]:
    #                 continue
    #             if "eid" not in e1 or "eid" not in e2:
    #                 labels.append(-100)
    #                 continue
    #             if ignore_timex:
    #                 if e1["id"].startswith("TIME") or e2["id"].startswith("TIME"):
    #                     labels.append(-100)
    #                     continue

    #             labels.append(pair2rel.get((e1["eid"], e2["eid"]), REL2ID["NONE"]))
    #     assert len(labels) == len(self.events) ** 2 - len(self.events)
    #     return labels
    
    def get_relation_labels(self, relations, REL2ID, ignore_timex=True):
        pair2rel = {}
        for rel in relations:
            for pair in relations[rel]:
                for e1 in self.eid2mentions[pair[0]]:
                    for e2 in self.eid2mentions[pair[1]]:
                        pair2rel[(e1["id"], e2["id"])] = REL2ID[rel]
                        if rel in BIDIRECTIONAL_REL:
                            pair2rel[(e2["id"], e1["id"])] = REL2ID[rel]
        labels = []
        for i, e1 in enumerate(self.events):
            for j, e2 in enumerate(self.events):
                if e1["id"] == e2["id"]:
                    continue
                if ignore_timex:
                    if e1["id"].startswith("TIME") or e2["id"].startswith("TIME"):
                        labels.append(-100)
                        continue
                labels.append(pair2rel.get((e1["id"], e2["id"]), REL2ID["NONE"]))
        assert len(labels) == len(self.events) ** 2 - len(self.events)
        return labels



class myDataset(Dataset):
    def __init__(self, tokenizer, data_dir, split, max_length=512, ignore_nonetype=False, sample_rate=None):
        if sample_rate and split != "train":
            print("sampling test or dev, is it intended?")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ignore_nonetype = ignore_nonetype
        self.load_examples(data_dir, split)
        if sample_rate:
            self.examples = list(random.sample(self.examples, int(sample_rate * len(self.examples))))
        self.tokenize()
        self.to_tensor()
    
    def load_examples(self, data_dir, split):
        self.examples = []
        with open(os.path.join(data_dir, f"{split}.json"))as f:
            lines = f.readlines()
        for line in lines:
            data = json.loads(line.strip())
            doc = Document(data, ignore_nonetype=self.ignore_nonetype)
            if doc.sorted_event_spans:
                self.examples.append(doc)
    
    def tokenize(self):
        # {input_ids, event_spans, event_group}
        # TODO: split articless into part of max_length
        self.tokenized_samples = []
        for example in tqdm(self.examples, desc="tokenizing"):
            event_spans = [] # [[(start, end)], [],...]
            input_ids = [] # [[], [], ...]

            # labels = example.labels
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
                    # start = len(tmp_input_ids) + 1 # account for special type tokens
                    # end = len(tmp_input_ids) + 1 + len(event_ids)
                    start = len(tmp_input_ids)
                    end = len(tmp_input_ids) + len(event_ids)
                    tmp_event_spans.append((start, end))
                    # special_ids = self.tokenizer(type_tokens(sp[2]), is_split_into_words=True, add_special_tokens=False)["input_ids"]
                    # assert len(special_ids) == 2, print(f"special tokens <{sp[2]}> and <{sp[2]}/> may not be added to tokenizer.")
                    # tmp_input_ids += special_ids[:1] + event_ids + special_ids[1:]
                    tmp_input_ids += event_ids


                    i = sp[1][1]
                    event_id += 1
                if word[i:]:
                    tmp_input_ids += self.tokenizer(word[i:], is_split_into_words=True, add_special_tokens=False)["input_ids"]
                
                # add SEP between sentences
                tmp_input_ids.append(self.tokenizer.sep_token_id)

                if len(sub_input_ids) + len(tmp_input_ids) <= self.max_length:
                    # print(len(sub_input_ids) + len(tmp_input_ids))
                    sub_event_spans += [(sp[0]+len(sub_input_ids), sp[1]+len(sub_input_ids)) for sp in tmp_event_spans]
                    sub_input_ids += tmp_input_ids
                else:
                    # print("exceed max length! truncate")
                    assert len(sub_input_ids) <= self.max_length
                    input_ids.append(sub_input_ids)
                    event_spans.append(sub_event_spans)

                    # assert len(tmp_input_ids) < self.max_length, print("A sentence too long!\n %s" % " ".join(words[sent_id])) # 3580:
                    while len(tmp_input_ids) >= self.max_length:
                        split_point = self.max_length - 1
                        while not valid_split(split_point, tmp_event_spans):
                            split_point -= 1
                        tmp_event_spans_part1, tmp_event_spans = split_spans(split_point, tmp_event_spans)
                        tmp_input_ids_part1, tmp_input_ids = tmp_input_ids[:split_point], tmp_input_ids[split_point:]

                        input_ids.append([self.tokenizer.cls_token_id] + tmp_input_ids_part1)
                        event_spans.append([(sp[0]+1, sp[1]+1) for sp in tmp_event_spans_part1])

                        tmp_event_spans = [(sp[0]-len(tmp_input_ids_part1), sp[1]-len(tmp_input_ids_part1)) for sp in tmp_event_spans]
                        # sub_input_ids = [self.tokenizer.cls_token_id] + tmp_input_ids_part2

                    sub_event_spans = [(sp[0]+1, sp[1]+1) for sp in tmp_event_spans]
                    sub_input_ids = [self.tokenizer.cls_token_id] + tmp_input_ids
            if sub_input_ids:
                input_ids.append(sub_input_ids)
                event_spans.append(sub_event_spans)
            
            assert event_id == len(spans)
                
            tokenized = {"input_ids": input_ids, "attention_mask": None, "event_spans": event_spans, "coref_labels": example.coref_labels, "temporal_labels": example.temporal_labels, "causal_labels": example.causal_labels, "subevent_labels": example.subevent_labels, "events_idx": example.events_idx}
            # , "temporal_labels_eval": example.temporal_labels_eval, "causal_labels_eval": example.causal_labels_eval, "subevent_labels_eval": example.subevent_labels_eval}
            self.tokenized_samples.append(tokenized)
    
    def to_tensor(self):
        for item in self.tokenized_samples:
            # print(item)
            attention_mask = []
            for ids in item["input_ids"]:
                mask = [1] * len(ids)
                while len(ids) < self.max_length:
                    ids.append(self.tokenizer.pad_token_id)
                    mask.append(0)
                attention_mask.append(mask)
            item["input_ids"] = torch.LongTensor(item["input_ids"])
            item["attention_mask"] = torch.LongTensor(attention_mask)
            # item["coref_labels"] = torch.LongTensor(item["coref_labels"])
            item["temporal_labels"] = torch.LongTensor(item["temporal_labels"])
            item["causal_labels"] = torch.LongTensor(item["causal_labels"])
            item["subevent_labels"] = torch.LongTensor(item["subevent_labels"])

            # item["temporal_labels_eval"] = torch.LongTensor(item["temporal_labels_eval"])
            # item["causal_labels_eval"] = torch.LongTensor(item["causal_labels_eval"])
            # item["subevent_labels_eval"] = torch.LongTensor(item["subevent_labels_eval"])
            # retain_event_index = [i for i in range(len(item["event_spans"])) if item["event_spans"][i][1] < self.max_length]
            # item["event_spans"] = [item["event_spans"][i] for i in retain_event_index]
            # item["label_groups"] = [[i for i in gr if i in retain_event_index] for gr in item["label_groups"]]
    
    def __getitem__(self, index):
        return self.tokenized_samples[index]

    def __len__(self):
        return len(self.tokenized_samples)

def get_special_tokens(filepath="../../MAVEN/types.txt"):
    types = open(filepath).readlines()
    type_tokens_list = []
    for t in types:
        type_tokens_list += type_tokens(t.strip())
    return type_tokens_list

def collator(data):
    collate_data = {"input_ids": [], "attention_mask": [], "event_spans": [], "splits": [0], "coref_labels": [], "temporal_labels": [],"causal_labels": [], "subevent_labels": [], "events_idx": []}
    # , "temporal_labels_eval": [],"causal_labels_eval": [], "subevent_labels_eval": []}
    for d in data:
        for k in d:
            collate_data[k].append(d[k])
            # if k == "label_groups":
            #     print(collate_data[k])
    lengths = [ids.size(0) for ids in collate_data["input_ids"]]
    for l in lengths:
        collate_data["splits"].append(collate_data["splits"][-1]+l)
    # max_length = max(lengths)
    # for i, ids in enumerate(collate_data["input_ids"]):
    #     if ids.size(0) < max_length:
    #         collate_data["input_ids"][i] = torch.nn.functional.pad(ids, (0,0,0,max_length-ids.size(0)), mode="constant", value=0)
    #         collate_data["attention_mask"][i] = torch.nn.functional.pad(collate_data["attention_mask"][i], (0,0,0,max_length-ids.size(0)), mode="constant", value=0)

    collate_data["input_ids"] = torch.cat(collate_data["input_ids"])
    collate_data["attention_mask"] = torch.cat(collate_data["attention_mask"])
    for label_type in ["temporal_labels", "causal_labels", "subevent_labels"]:
    # , "temporal_labels_eval", "causal_labels_eval", "subevent_labels_eval"]:
        max_label_length = max([len(label) for label in collate_data[label_type]])
        collate_data[label_type] = torch.stack([F.pad(label, pad=(0, max_label_length-len(label)), value=-100) for label in collate_data[label_type]])
    return collate_data

def get_dataloader(tokenizer, split, data_dir="../data", max_length=128, batch_size=8, shuffle=True, ignore_nonetype=False, sample_rate=None):
    dataset = myDataset(tokenizer, data_dir, split, max_length=max_length, ignore_nonetype=ignore_nonetype, sample_rate=sample_rate)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collator)

if __name__ == "__main__":
    from transformers import RobertaTokenizer
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    # tokens = get_special_tokens()
    # print(tokens)
    # tokenizer.add_special_tokens(tokens)
    # n = tokenizer.add_tokens(tokens)
    # print(n)
    # dataset = myDataset(tokenizer, "../data/", "test")
    # print(dataset[0])
    # print(dataset[1])
    # print(dataset[2])
    dataloader = get_dataloader(tokenizer, "test", shuffle=False, max_length=256)
    for data in dataloader:
        print(data["input_ids"].size())
        print(data["attention_mask"].size())
        print(data.keys())
        print(data["events_idx"])
        print(list(sorted(sum(data["coref_labels"][0], []))))
        break
        