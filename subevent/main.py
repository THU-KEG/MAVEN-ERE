import sys
sys.path.append("../")
sys.path.append("src/")
from utils.utils import to_cuda
from utils.model import Model
import torch
import random
import numpy as np
from tqdm import tqdm
from data import get_dataloader
from transformers import AdamW, RobertaTokenizer, get_linear_schedule_with_warmup
import argparse
from torch.optim import Adam
import torch.nn as nn
from sklearn.metrics import classification_report
from data import REL2ID, ID2REL
import warnings
import os
import sys
from pathlib import Path 
from dump_result import dump_result

warnings.filterwarnings("ignore")


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
    global REPORT_CLASS_NAMES
    global REPORT_CLASS_LABELS
    pred_list = []
    label_list = []
    model.eval()
    with torch.no_grad():
        for data in tqdm(dataloader, desc=desc):
            for k in data:
                if isinstance(data[k], torch.Tensor):
                    data[k] = to_cuda(data[k])

            scores = model(data)
            labels = data["labels"]
            scores = scores.view(-1, scores.size(-1))
            labels = labels.view(-1)
            pred = torch.argmax(scores, dim=-1)
            pred_list.extend(pred[labels>=0].cpu().numpy().tolist())
            label_list.extend(labels[labels>=0].cpu().numpy().tolist())

    result_collection = classification_report(label_list, pred_list, output_dict=True, target_names=REPORT_CLASS_NAMES, labels=REPORT_CLASS_LABELS)
    print(f"{desc} result:", result_collection)
    return result_collection

def predict(model, dataloader):
    all_preds = []
    model.eval()
    with torch.no_grad():
        for data in tqdm(dataloader, desc="Predict"):
            for k in data:
                if isinstance(data[k], torch.Tensor):
                    data[k] = to_cuda(data[k])
            scores = model(data)
            labels = data["labels"]
            scores = scores.view(-1, scores.size(-1))
            labels = labels.view(-1)
            pred = torch.argmax(scores, dim=-1)
            max_label_length = data["max_label_length"]
            n_doc = len(labels) // max_label_length
            assert len(labels) % max_label_length == 0
            for i in range(n_doc):
                selected_index = labels[i*max_label_length:(i+1)*max_label_length] >= -1 # -1 means no label, -100 means padding
                all_preds.append({
                    "doc_id": data["doc_id"][i],
                    "preds": pred[i*max_label_length:(i+1)*max_label_length][selected_index].cpu().numpy().tolist(),
                })
    return all_preds



if __name__ == "__main__":
    import json
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_steps", default=50, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--log_steps", default=50, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--bert_lr", default=1e-5, type=float)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--sample_rate", default=None, type=float, help="randomly sample a portion of the training data")


    args = parser.parse_args()

    label_num = len(ID2REL)
    REPORT_CLASS_NAMES = [ID2REL[i] for i in range(0,len(ID2REL) - 1)]
    REPORT_CLASS_LABELS = list(range(len(ID2REL) - 1))
    output_dir = Path(f"./output/{args.seed}/maven_{args.sample_rate}")
    output_dir.mkdir(exist_ok=True, parents=True)
    sys.stdout = open(os.path.join(output_dir, "log.txt"), 'w')

    set_seed(args.seed)
    
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    print("loading data...")
    if not args.eval_only:
        train_dataloader = get_dataloader(tokenizer, "train", data_dir="../data/MAVEN_ERE", max_length=256, shuffle=True, batch_size=args.batch_size, sample_rate=args.sample_rate)
        dev_dataloader = get_dataloader(tokenizer, "valid", data_dir="../data/MAVEN_ERE", max_length=256, shuffle=False, batch_size=args.batch_size)
    test_dataloader = get_dataloader(tokenizer, "test", data_dir="../data/MAVEN_ERE", max_length=256, shuffle=False, batch_size=args.batch_size)

    print("loading model...")
    model = Model(len(tokenizer), out_dim=label_num)
    model = to_cuda(model)

    if not args.eval_only:
        bert_optimizer = AdamW([p for p in model.encoder.model.parameters() if p.requires_grad], lr=args.bert_lr)
        optimizer = Adam([p for p in model.scorer.parameters() if p.requires_grad], lr=args.lr)

        scheduler = get_linear_schedule_with_warmup(bert_optimizer, num_warmup_steps=200, num_training_steps=len(train_dataloader) * args.epochs)
    eps = 1e-8

    Loss = nn.CrossEntropyLoss(ignore_index=-100)
    glb_step = 0
    if not args.eval_only:
        print("*******************start training********************")

        train_losses = []
        pred_list = []
        label_list = []
        best_score = 0.0
        for epoch in range(args.epochs):
            for data in tqdm(train_dataloader, desc=f"Training epoch {epoch}"):
                model.train()

                for k in data:
                    if isinstance(data[k], torch.Tensor):
                        data[k] = to_cuda(data[k])

                scores = model(data)
                labels = data["labels"]
                scores = scores.view(-1, scores.size(-1))
                labels = labels.view(-1)
                loss = Loss(scores, labels)
                pred = torch.argmax(scores, dim=-1)
                pred_list.extend(pred[labels>=0].cpu().numpy().tolist())
                label_list.extend(labels[labels>=0].cpu().numpy().tolist())

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
                    print("true:")
                    print(label_list[:20])
                    print("pred:")
                    print(pred_list[:20])
                    print("Train %d steps: loss=%f" % (glb_step, np.mean(train_losses)))
                    res = classification_report(label_list, pred_list, output_dict=True, target_names=REPORT_CLASS_NAMES, labels=REPORT_CLASS_LABELS)
                    print("Train result:", res)
                    train_losses = []
                    pred_list = []
                    label_list = []

                if glb_step % args.eval_steps == 0:
                    res = evaluate(model, dev_dataloader, desc="Validation")
                    if "micro avg" not in res:
                        current_score = res["accuracy"]
                    else:
                        current_score = res["micro avg"]["f1-score"]
                    if current_score > best_score:
                        print("best result!")
                        best_score = current_score
                        state = {"model":model.state_dict(), "optimizer":optimizer.state_dict(), "scheduler": scheduler.state_dict()}
                        torch.save(state, os.path.join(output_dir, "best"))

    print("*" * 30 + "Predict"+ "*" * 30)
    if os.path.exists(os.path.join(output_dir, "best")):
        print("loading from", os.path.join(output_dir, "best"))
        state = torch.load(os.path.join(output_dir, "best"))
        model.load_state_dict(state["model"])
    all_preds = predict(model, test_dataloader)
    dump_result("../data/MAVEN_ERE/test.jsonl", all_preds, output_dir)

    sys.stdout.close()