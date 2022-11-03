from unittest import result
from model import Model
from utils import to_cuda, to_var
import torch
import random
import numpy as np
from tqdm import tqdm
from data import myDataset, get_special_tokens, get_dataloader
from transformers import AdamW, RobertaTokenizer, get_linear_schedule_with_warmup
from utils import get_predicted_clusters, get_event2cluster, fill_expand
from metrics import evaluate_documents, b_cubed, ceafe, muc, Evaluator, blanc
import argparse
from torch.optim import Adam
import torch.nn as nn
from sklearn.metrics import classification_report
from data import TEMPREL2ID, ID2TEMPREL, CAUSALREL2ID, ID2CAUSALREL, SUBEVENTREL2ID, ID2SUBEVENTREL
import warnings
import os
import sys
warnings.filterwarnings("ignore")

class EvalResult:
    def __init__(self, gold, mention_to_gold, clusters, mention_to_cluster):
        self.gold = gold
        self.mention_to_gold = mention_to_gold
        self.clusters = clusters
        self.mention_to_cluster = mention_to_cluster


def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def evaluate(model, dataloader, desc=""):
    temporal_pred_list = []
    temporal_label_list = []
    causal_pred_list = []
    causal_label_list = []
    subevent_pred_list = []
    subevent_label_list = []
    coref_train_eval_results = []
    for data in tqdm(dataloader, desc=desc):
        model.train()

        for k in data:
            if isinstance(data[k], torch.Tensor):
                data[k] = to_cuda(data[k])

        coref_scores, temporal_scores, causal_scores, subevent_scores = model(data)

        # coreference ###########################
        for i in range(len(coref_scores)):
            prob = coref_scores[i]
            # print(prob[:10, :10])
            labels = data["coref_labels"][i]

            pred_clusters, pred_event2cluster = get_predicted_clusters(prob)
            # print("pred:", pred_clusters)
            # print("gold:", labels)
            gold_event2cluster = get_event2cluster(labels)

            assert len(pred_event2cluster) == len(gold_event2cluster), print(pred_event2cluster, gold_event2cluster)
            eval_result = EvalResult(labels, gold_event2cluster, pred_clusters, pred_event2cluster)
            coref_train_eval_results.append(eval_result)
        # ###########################################

        # output = [temporal_scores, causal_scores, subevent_scores]
        labels = data["temporal_labels"]
        scores = temporal_scores
        scores = scores.view(-1, scores.size(-1))
        labels = labels.view(-1)
        # print("labels:", labels[:10])
        pred = torch.argmax(scores, dim=-1)
        temporal_pred_list.extend(pred[labels>=0].cpu().numpy().tolist())
        temporal_label_list.extend(labels[labels>=0].cpu().numpy().tolist())


        labels = data["causal_labels"]
        scores = causal_scores
        scores = scores.view(-1, scores.size(-1))
        labels = labels.view(-1)
        # print("labels:", labels[:10])
        pred = torch.argmax(scores, dim=-1)
        causal_pred_list.extend(pred[labels>=0].cpu().numpy().tolist())
        causal_label_list.extend(labels[labels>=0].cpu().numpy().tolist())

        labels = data["subevent_labels"]
        scores = subevent_scores
        scores = scores.view(-1, scores.size(-1))
        labels = labels.view(-1)
        # print("labels:", labels[:10])
        pred = torch.argmax(scores, dim=-1)
        subevent_pred_list.extend(pred[labels>=0].cpu().numpy().tolist())
        subevent_label_list.extend(labels[labels>=0].cpu().numpy().tolist())

    result_collection = {"COREFERENCE": {}}

    print("*"*20 + desc + "*"*20)
    for metric, name in zip(metrics, metric_names):
        res = evaluate_documents(coref_train_eval_results, metric)
        # print("COREFRENCE %s: precision=%.4f, recall=%.4f, f1=%.4f" % (name, *res))
        result_collection["COREFERENCE"][name] = {"precision": res[0], "recall": res[1], "f1": res[2]}
        
    temporal_res = classification_report(temporal_label_list, temporal_pred_list, output_dict=True, target_names=TEMP_REPORT_CLASS_NAMES, labels=TEMP_REPORT_CLASS_LABELS)
    print("TEMPORAL:", temporal_res)
    result_collection["TEMPORAL"] = temporal_res

    causal_res = classification_report(causal_label_list, causal_pred_list, output_dict=True, target_names=CAUSAL_REPORT_CLASS_NAMES, labels=CAUSAL_REPORT_CLASS_LABELS)
    print("CAUSAL:", causal_res)
    result_collection["CAUSAL"] = causal_res


    subevent_res = classification_report(subevent_label_list, subevent_pred_list, output_dict=True, target_names=SUBEVENT_REPORT_CLASS_NAMES, labels=SUBEVENT_REPORT_CLASS_LABELS)
    print("SUBEVENT:", subevent_res)
    result_collection["SUBEVENT"] = subevent_res

    return result_collection



if __name__ == "__main__":
    import json
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_steps", default=50, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--log_steps", default=50, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--bert_lr", default=1e-5, type=float)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--ignore_nonetype", action="store_true")
    parser.add_argument("--sample_rate", default=None, type=float, help="randomly sample a portion of the training data")


    args = parser.parse_args()

    # if args.ignore_nonetype:
    #     label_num = len(ID2REL) - 1
    TEMP_REPORT_CLASS_NAMES = [ID2TEMPREL[i] for i in range(0,len(ID2TEMPREL) - 1)]
    CAUSAL_REPORT_CLASS_NAMES = [ID2CAUSALREL[i] for i in range(1,len(ID2CAUSALREL))]
    SUBEVENT_REPORT_CLASS_NAMES = [ID2SUBEVENTREL[i] for i in range(1,len(ID2SUBEVENTREL))]


    TEMP_REPORT_CLASS_LABELS = list(range(len(ID2TEMPREL) - 1))
    CAUSAL_REPORT_CLASS_LABELS = list(range(1, len(ID2CAUSALREL)))
    SUBEVENT_REPORT_CLASS_LABELS = list(range(1, len(ID2SUBEVENTREL)))


    output_dir = f"../output/maven_ignore_none_{args.ignore_nonetype}_{args.sample_rate}"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
    sys.stdout = open(os.path.join(output_dir, "log.txt"), 'w')


    set_seed(args.seed)
    
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    # tokens = get_special_tokens()

    # n = tokenizer.add_tokens(tokens)

    print("loading data...")
    if not args.eval_only:
        train_dataloader = get_dataloader(tokenizer, "train", max_length=256, shuffle=True, batch_size=args.batch_size, ignore_nonetype=args.ignore_nonetype, sample_rate=args.sample_rate)
        dev_dataloader = get_dataloader(tokenizer, "dev", max_length=256, shuffle=False, batch_size=args.batch_size, ignore_nonetype=args.ignore_nonetype)
    test_dataloader = get_dataloader(tokenizer, "test", max_length=256, shuffle=False, batch_size=args.batch_size, ignore_nonetype=args.ignore_nonetype)

    print("loading model...")
    model = Model(len(tokenizer))
    model = to_cuda(model)
    # inner_model = model.module if isinstance(nn.DataParallel) else model

    if not args.eval_only:
        bert_optimizer = AdamW([p for p in model.encoder.model.parameters() if p.requires_grad], lr=args.bert_lr)

        scorer_param = []
        scorer_param += [p for p in model.temporal_scorer.parameters() if p.requires_grad]
        scorer_param += [p for p in model.causal_scorer.parameters() if p.requires_grad]
        scorer_param += [p for p in model.subevent_scorer.parameters() if p.requires_grad]
        scorer_param += [p for p in model.coref_scorer.parameters() if p.requires_grad]
        optimizer = Adam(scorer_param, lr=args.lr)

        scheduler = get_linear_schedule_with_warmup(bert_optimizer, num_warmup_steps=200, num_training_steps=len(train_dataloader) * args.epochs)
    eps = 1e-8

    metrics = [b_cubed, ceafe, muc, blanc]
    metric_names = ["B-cubed", "CEAF", "MUC", "BLANC"]
    # evaluaters = [Evaluator(metric) for metric in metrics]
    Loss = nn.CrossEntropyLoss(ignore_index=-100)
    glb_step = 0
    if not args.eval_only:
        print("*******************start training********************")

        train_losses = []
        temporal_pred_list = []
        temporal_label_list = []
        causal_pred_list = []
        causal_label_list = []
        subevent_pred_list = []
        subevent_label_list = []
        coref_train_eval_results = []
        best_score = {"COREFERENCE": {name:0.0 for name in metric_names}, "TEMPORAL": 0.0, "CAUSAL": 0.0, "SUBEVENT": 0.0}
        for epoch in range(args.epochs):
            for data in tqdm(train_dataloader, desc=f"Training epoch {epoch}"):
                model.train()
                loss = 0.0

                for k in data:
                    if isinstance(data[k], torch.Tensor):
                        data[k] = to_cuda(data[k])

                coref_scores, temporal_scores, causal_scores, subevent_scores = model(data)

                # coreference ###########################
                for i in range(len(coref_scores)):
                    prob = coref_scores[i]
                    # print(prob[:10, :10])
                    labels = data["coref_labels"][i]
                    # print("labels:", labels)
                    filled_labels = fill_expand(labels)
                    filled_labels = to_cuda(filled_labels)
                    # print(filled_labels.size())
                    # print(prob.size())
                    weight = torch.eye(prob.size(0))
                    weight[weight==0.0] = 0.1
                    weight = weight.to(prob.device)
                    prob_sum = torch.sum(torch.clamp(torch.mul(prob, filled_labels), eps, 1-eps), dim=1)
                    # print(prob_sum)
                    loss = loss + torch.sum(torch.log(prob_sum)) * -1

                    pred_clusters, pred_event2cluster = get_predicted_clusters(prob)
                    # print("pred:", pred_clusters)
                    # print("gold:", labels)
                    gold_event2cluster = get_event2cluster(labels)

                    assert len(pred_event2cluster) == len(gold_event2cluster), print(pred_event2cluster, gold_event2cluster)
                    eval_result = EvalResult(labels, gold_event2cluster, pred_clusters, pred_event2cluster)
                    coref_train_eval_results.append(eval_result)
                # ###########################################

                # output = [temporal_scores, causal_scores, subevent_scores]
                labels = data["temporal_labels"]
                # labels_eval = data["temporal_labels_eval"].view(-1)
                scores = temporal_scores
                scores = scores.view(-1, scores.size(-1))
                labels = labels.view(-1)
                # print("labels:", labels[:10])
                loss += Loss(scores, labels)
                pred = torch.argmax(scores, dim=-1)
                temporal_pred_list.extend(pred[labels>=0].cpu().numpy().tolist())
                temporal_label_list.extend(labels[labels>=0].cpu().numpy().tolist())


                labels = data["causal_labels"]
                # labels_eval = data["causal_labels_eval"].view(-1)
                scores = causal_scores
                scores = scores.view(-1, scores.size(-1))
                labels = labels.view(-1)
                # print("labels:", labels[:10])
                loss += Loss(scores, labels)
                pred = torch.argmax(scores, dim=-1)
                causal_pred_list.extend(pred[labels>=0].cpu().numpy().tolist())
                causal_label_list.extend(labels[labels>=0].cpu().numpy().tolist())

                labels = data["subevent_labels"]
                # labels_eval = data["subevent_labels_eval"].view(-1)
                scores = subevent_scores
                scores = scores.view(-1, scores.size(-1))
                labels = labels.view(-1)
                # print("labels:", labels[:10])
                loss += Loss(scores, labels)
                pred = torch.argmax(scores, dim=-1)
                subevent_pred_list.extend(pred[labels>=0].cpu().numpy().tolist())
                subevent_label_list.extend(labels[labels>=0].cpu().numpy().tolist())



                train_losses.append(loss.item())
                loss.backward()
                optimizer.step()
                bert_optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                bert_optimizer.zero_grad()

                glb_step += 1

                if glb_step % args.log_steps == 0:
                    print("*"*20 + "Train Prediction Examples" + "*"*20)
                    print("Train %d steps: loss=%f" % (glb_step, np.mean(train_losses)))
                    for metric, name in zip(metrics, metric_names):
                        res = evaluate_documents(coref_train_eval_results, metric)
                        print("COREFRENCE %s: precision=%.4f, recall=%.4f, f1=%.4f" % (name, *res))
                    
                    temporal_res = classification_report(temporal_label_list, temporal_pred_list, output_dict=True, target_names=TEMP_REPORT_CLASS_NAMES, labels=TEMP_REPORT_CLASS_LABELS)
                    print("TEMPORAL:", temporal_res)

                    causal_res = classification_report(causal_label_list, causal_pred_list, output_dict=True, target_names=CAUSAL_REPORT_CLASS_NAMES, labels=CAUSAL_REPORT_CLASS_LABELS)
                    print("CAUSAL:", causal_res)

                    subevent_res = classification_report(subevent_label_list, subevent_pred_list, output_dict=True, target_names=SUBEVENT_REPORT_CLASS_NAMES, labels=SUBEVENT_REPORT_CLASS_LABELS)
                    print("SUBEVENT:", subevent_res)
                    

                    train_losses = []
                    temporal_pred_list = []
                    temporal_label_list = []
                    causal_pred_list = []
                    causal_label_list = []
                    subevent_pred_list = []
                    subevent_label_list = []
                    coref_train_eval_results = []


                if glb_step % args.eval_steps == 0:
                    res = evaluate(model, dev_dataloader, desc="Validation")
                    anybetter = False
                    for k in ["TEMPORAL", "CAUSAL", "SUBEVENT"]:
                        if res[k]["micro avg"]["f1-score"] > best_score[k]:
                            anybetter = True
                            best_score[k] = res[k]["micro avg"]["f1-score"]
                    for k in metric_names:
                        if res["COREFERENCE"][k]["f1"] > best_score["COREFERENCE"][k]:
                            best_score["COREFERENCE"][k] = res["COREFERENCE"][k]["f1"]
                            anybetter = True

                    if anybetter:
                        print("better result!")
                        state = {"model":model.state_dict(), "optimizer":optimizer.state_dict(), "scheduler": scheduler.state_dict()}
                        torch.save(state, os.path.join(output_dir, "best"))

    
    print("*" * 30 + "Test"+ "*" * 30)
    if os.path.exists(os.path.join(output_dir, "best")):
        print(f"loading from {os.path.join(output_dir, 'best')}")
        state = torch.load(os.path.join(output_dir, "best"))
        model.load_state_dict(state["model"])
    res = evaluate(model, test_dataloader, desc="Test")
    with open(os.path.join(output_dir, "test_result.txt"), "w", encoding="utf-8")as f:
        f.writelines(json.dumps(res, indent=4))
    # for metric, name in zip(metrics, metric_names):
    #     res = evaluate_documents(train_eval_results, metric)
    #     print("%s: precision=%.4f, recall=%.4f, f1=%.4f" % (name, *res))

    sys.stdout.close()