from pyparsing import col
import numpy as np
from collections import Counter
from .linear_assignment import linear_assignment
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
                
if __name__ == "__main__":
    from utils import get_event2cluster
    class Doc:
        def __init__(self, mention2cluster, mention2gold, clusters, gold):
            self.mention_to_cluster = mention2cluster
            self.mention_to_gold = mention2gold
            self.clusters = clusters
            self.gold = gold

    gold = [[1,2,3,4,5], [6,7], [8,9,10, 11,12], [13]]
    pred = [[1,2,3,4,5], [6,7], [8,9,10,11,12], [13]]
    mention2cluster = get_event2cluster(pred)
    mention2gold = get_event2cluster(gold)
    doc= Doc(mention2cluster, mention2gold, pred, gold)
    p, r, f = evaluate_documents([doc], muc)
    print(p, r, f)
    p, r, f = evaluate_documents([doc], b_cubed)
    print(p, r, f)
    p, r, f = evaluate_documents([doc], ceafe)
    print(p, r, f)
    p, r, f = evaluate_documents([doc], blanc)
    print(p, r, f)