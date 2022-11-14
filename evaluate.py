#!/usr/bin/env python

# Scoring program for the AutoML challenge
# Isabelle Guyon and Arthur Pesah, ChaLearn, August 2014-November 2016

# ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS".
# ISABELLE GUYON, CHALEARN, AND/OR OTHER ORGANIZERS OR CODE AUTHORS DISCLAIM
# ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE, AND THE
# WARRANTY OF NON-INFRINGEMENT OF ANY THIRD PARTY'S INTELLECTUAL PROPERTY RIGHTS.
# IN NO EVENT SHALL ISABELLE GUYON AND/OR OTHER ORGANIZERS BE LIABLE FOR ANY SPECIAL,
# INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF SOFTWARE, DOCUMENTS, MATERIALS,
# PUBLICATIONS, OR INFORMATION MADE AVAILABLE FOR THE CHALLENGE.

# Some libraries and options
import os
from sys import argv
import json
import yaml
import numpy as np
from sklearn.metrics import f1_score,precision_score,recall_score
from collections import Counter
from scipy.optimize import linear_sum_assignment


def f1(p_num, p_den, r_num, r_den, beta=1):
    p = 0 if p_den == 0 else p_num / float(p_den)
    r = 0 if r_den == 0 else r_num / float(r_den)
    return 0 if p + r == 0 else (1 + beta * beta) * p * r / (beta * beta * p + r)


class Evaluator:
    def __init__(self, metric, beta=1):
        self.p_num = 0
        self.p_den = 0
        self.r_num = 0
        self.r_den = 0
        self.metric = metric
        self.beta = beta
        # for blanc
        self.rc = 0
        self.wc = 0
        self.rn = 0
        self.wn = 0

    def update(self, document):
        if self.metric == blanc:
            rc, wc, rn, wn = self.metric(document.mention_to_cluster, document.mention_to_gold)
            self.rc += rc
            self.wc += wc
            self.rn += rn
            self.wn += wn
        else:
            if self.metric == ceafe:
                pn, pd, rn, rd = self.metric(document.clusters, document.gold)
            else:
                pn, pd = self.metric(document.clusters, document.mention_to_gold)
                rn, rd = self.metric(document.gold, document.mention_to_cluster)
            self.p_num += pn
            self.p_den += pd
            self.r_num += rn
            self.r_den += rd

    def get_f1(self):
        if self.metric == blanc:
            return (f1(self.rc, self.rc+self.wc, self.rc, self.rc+self.wn, beta=self.beta) + f1(self.rn, self.rn+self.wn, self.rn, self.rn+self.wc, beta=self.beta)) / 2
        return f1(self.p_num, self.p_den, self.r_num, self.r_den, beta=self.beta)

    def get_recall(self):
        if self.metric == blanc:
            return (self.rc/(self.rc+self.wn+1e-6) + self.rn/(self.rn+self.wc+1e-6)) / 2
        return 0 if self.r_num == 0 else self.r_num / float(self.r_den)

    def get_precision(self):
        if self.metric == blanc:
            return (self.rc/(self.rc+self.wc+1e-6) + self.rn/(self.rn+self.wn+1e-6)) / 2
        return 0 if self.p_num == 0 else self.p_num / float(self.p_den)

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()

def evaluate_documents(documents, metric, beta=1):
    evaluator = Evaluator(metric, beta=beta)
    for document in documents:
        evaluator.update(document)
    return evaluator.get_precision(), evaluator.get_recall(), evaluator.get_f1()


def b_cubed(clusters, mention_to_gold):
    num, dem = 0, 0

    for c in clusters:
        gold_counts = Counter()
        correct = 0
        for m in c:
            if m in mention_to_gold:
                gold_counts[tuple(mention_to_gold[m])] += 1
        for c2, count in gold_counts.items():
            correct += count * count

        num += correct / float(len(c))
        dem += len(c)

    return num, dem


def muc(clusters, mention_to_gold):
    tp, p = 0, 0
    for c in clusters:
        p += len(c) - 1
        tp += len(c)
        linked = set()
        for m in c:
            if m in mention_to_gold:
                linked.add(mention_to_gold[m])
            else:
                tp -= 1
        tp -= len(linked)
    return tp, p


def phi4(c1, c2):
    return 2 * len([m for m in c1 if m in c2]) / float(len(c1) + len(c2))


def ceafe(clusters, gold_clusters):
    scores = np.zeros((len(gold_clusters), len(clusters)))
    for i in range(len(gold_clusters)):
        for j in range(len(clusters)):
            scores[i, j] = phi4(gold_clusters[i], clusters[j])
    row_id, col_id = linear_sum_assignment(-scores)
    similarity = sum(scores[row_id, col_id])
    return similarity, len(clusters), similarity, len(gold_clusters)

def lea(clusters, mention_to_gold):
    num, dem = 0, 0

    for c in clusters:
        if len(c) == 1:
            continue
        common_links = 0
        all_links = len(c) * (len(c) - 1) / 2.0
        for i, m in enumerate(c):
            if m in mention_to_gold:
                for m2 in c[i + 1:]:
                    if m2 in mention_to_gold and mention_to_gold[m] == mention_to_gold[m2]:
                        common_links += 1
        num += len(c) * common_links / float(all_links)
        dem += len(c)

    return num, dem

def blanc(mention_to_cluster, mention_to_gold):
    rc = 0
    wc = 0
    rn = 0
    wn = 0
    assert len(mention_to_cluster) == len(mention_to_gold)
    mentions = list(mention_to_cluster.keys())
    for i in range(len(mentions)):
        for j in range(i+1, len(mentions)):
            if mention_to_cluster[mentions[i]] == mention_to_cluster[mentions[j]]:
                if mention_to_gold[mentions[i]] == mention_to_gold[mentions[j]]:
                    rc += 1
                else:
                    wc += 1
            else:
                if mention_to_gold[mentions[i]] == mention_to_gold[mentions[j]]:
                    wn += 1
                else:
                    rn += 1
    return rc, wc, rn, wn

class EvalResult:
    def __init__(self, gold, mention_to_gold, clusters, mention_to_cluster):
        self.gold = gold
        self.mention_to_gold = mention_to_gold
        self.clusters = clusters
        self.mention_to_cluster = mention_to_cluster

REL2ID={
"temporal":{
    "NONE": 0,
    "BEFORE": 1,
    "OVERLAP": 2,
    "CONTAINS": 3,
    "SIMULTANEOUS": 4,
    "ENDS-ON": 5,
    "BEGINS-ON": 6,
},
"causal":{
    "NONE": 0,
    "PRECONDITION": 1,
    "CAUSE": 2
},
"subevent":{
    "NONE": 0,
    "subevent": 1
}
}

def evaluate_coreference(golden, res):
    eval_results=[]
    def inv(clusters):
        res={}
        for clu in clusters:
            for i in clu:
                res[i]=tuple(clu)
        return res
    for id in golden:
        golden_doc=golden[id]
        eid={}
        gold_clusters=[]
        for event in golden_doc['events']:
            clu=[]
            for mention in event['mention']:
                eid[mention['id']]=len(eid)
                clu.append(eid[mention['id']])
            gold_clusters.append(clu)
        pred_clusters=[]
        if id in res and 'coreference' in res[id]:
            pred_doc=res[id]
            for cluster in pred_doc['coreference']:
                clu=[]
                for mention in cluster:
                    if mention in eid:
                        clu.append(eid[mention])
                        eid[mention]=-1
                if len(clu):
                    pred_clusters.append(tuple(clu))
        for m in eid:
            if eid[m]!=-1:
                pred_clusters.append([eid[m]])
        gold_event2cluster = inv(gold_clusters)
        pred_event2cluster = inv(pred_clusters)
        eval_result = EvalResult(gold_clusters, gold_event2cluster, pred_clusters, pred_event2cluster)
        eval_results.append(eval_result)
    res={}
    metrics = [b_cubed, ceafe, muc, blanc]
    metric_names = ["b_cubed", "ceaf", "muc", "blanc"]
    for metric, name in zip(metrics, metric_names):
        p,r,f = evaluate_documents(eval_results, metric)
        res[name+"_precision"] = p*100.0
        res[name+"_recall"] = r*100.0
        res[name+"_f1"] = f*100.0
    return res

def evaluate(golden, res, type):
    labels=[]
    preds=[]
    pair_mp={}
    for id in golden:
        golden_doc=golden[id]
        mentions=[]
        eid={}
        for event in golden_doc['events']:
            for mention in event['mention']:
                mentions.append(mention['id'])
            eid[event['id']]=event['mention']
        if type=="temporal":
            for time in golden_doc['TIMEX']:
                mentions.append(time['id'])
                eid[time['id']]=[time]
        for m1 in mentions:
            for m2 in mentions:
                if m1!=m2:
                    pair_mp[m1+m2]=len(labels)
                    labels.append(0)
                    preds.append(0)
        mentions=set(mentions)
        if type!="subevent":
            for rel in golden_doc[type+"_relations"]:
                for e_pair in golden_doc[type+"_relations"][rel]:
                    for m1 in eid[e_pair[0]]:
                        for m2 in eid[e_pair[1]]:
                            labels[pair_mp[m1['id']+m2['id']]]=REL2ID[type][rel]
        else:
            for e_pair in golden_doc["subevent_relations"]:
                for m1 in eid[e_pair[0]]:
                    for m2 in eid[e_pair[1]]:
                        labels[pair_mp[m1['id']+m2['id']]]=REL2ID[type]["subevent"]
        if id in res:
            pred_doc=res[id]
            if type+"_relations" in pred_doc:
                if type!="subevent":
                    for rel in pred_doc[type+"_relations"]:
                        for pair in pred_doc[type+"_relations"][rel]:
                            if pair[0] in mentions and pair[1] in mentions:
                                preds[pair_mp[pair[0]+pair[1]]]=REL2ID[type][rel]
                else:
                    for pair in pred_doc["subevent_relations"]:
                        if pair[0] in mentions and pair[1] in mentions:
                            preds[pair_mp[pair[0]+pair[1]]]=REL2ID[type]["subevent"]
    pos_labels=list(range(1,len(REL2ID[type])))
    labels=np.array(labels)
    preds=np.array(preds)
    p=precision_score(labels,preds,labels=pos_labels,average='micro')*100.0
    r=recall_score(labels,preds,labels=pos_labels,average='micro')*100.0
    f1=f1_score(labels,preds,labels=pos_labels,average='micro')*100.0
    return {type+"_precision":p, type+"_recall":r, type+"_f1":f1}


if __name__ == "__main__":
    input_dir = argv[1]
    output_dir = argv[2]
    submit_dir = os.path.join(input_dir, 'res')
    truth_dir = os.path.join(input_dir, 'ref')
    # Create the output directory, if it does not already exist and open output files
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    score_file = open(os.path.join(output_dir, 'scores.txt'), 'w')
    html_file = open(os.path.join(output_dir, 'scores.html'), 'w')

    # read ground truth
    grouth_truth={}
    with open(os.path.join(truth_dir,"test.jsonl"),"r") as fin:
        lines=fin.readlines()
        for line in lines:
            doc=json.loads(line.strip())
            grouth_truth[doc['id']]=doc

    # read prediction
    prediction={}
    with open(os.path.join(submit_dir,"test_prediction.jsonl"),"r") as fin:
        lines=fin.readlines()
        for line in lines:
            doc=json.loads(line.strip())
            prediction[doc['id']]=doc

    avg=[]
    ## Coreference
    coreference_scores=evaluate_coreference(grouth_truth, prediction)
    f1=0.0
    for k in coreference_scores:
        print(k + ": %0.2f\n" % coreference_scores[k])
        html_file.write("======= score (" + k + ")=%0.2f =======\n" % coreference_scores[k])
        score_file.write(k + ": %0.2f\n" % coreference_scores[k])
        if k.endswith("f1"):
            f1+=coreference_scores[k]/4.0
    avg.append(f1)
    ## Temporal
    temporal_scores=evaluate(grouth_truth, prediction, "temporal")
    for k in temporal_scores:
        print(k + ": %0.2f\n" % temporal_scores[k])
        html_file.write("======= score (" + k + ")=%0.2f =======\n" % temporal_scores[k])
        score_file.write(k + ": %0.2f\n" % temporal_scores[k])
    avg.append(temporal_scores['temporal_f1'])
    ## Causal
    causal_scores=evaluate(grouth_truth, prediction, "causal")
    for k in causal_scores:
        print(k + ": %0.2f\n" % causal_scores[k])
        html_file.write("======= score (" + k + ")=%0.2f =======\n" % causal_scores[k])
        score_file.write(k + ": %0.2f\n" % causal_scores[k])
    avg.append(causal_scores['causal_f1'])
    ## Subevent
    subevent_scores=evaluate(grouth_truth, prediction, "subevent")
    for k in subevent_scores:
        print(k + ": %0.2f\n" % subevent_scores[k])
        html_file.write("======= score (" + k + ")=%0.2f =======\n" % subevent_scores[k])
        score_file.write(k + ": %0.2f\n" % subevent_scores[k])
    avg.append(subevent_scores['subevent_f1'])
    avg_f1=sum(avg)/4.0
    print("overall_f1: %0.2f\n" % avg_f1)
    html_file.write("======= score (overall_f1)=%0.2f =======\n" % avg_f1)
    score_file.write("overall_f1: %0.2f\n" % avg_f1)

    # Read the execution time and add it to the scores:
    try:
        metadata = yaml.load(open(os.path.join(input_dir, 'res', 'metadata'), 'r'))
        score_file.write("Duration: %0.2f\n" % metadata['elapsedTime'])
    except:
        score_file.write("Duration: 0\n")
        html_file.close()
    score_file.close()
