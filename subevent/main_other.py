import sys
sys.path.append("../")
sys.path.append("src/")
from utils.utils import to_cuda
from utils.model import Model
import torch
import random
import numpy as np
from tqdm import tqdm
from data_other import get_dataloader
from transformers import AdamW, RobertaTokenizer, get_linear_schedule_with_warmup
import argparse
from torch.optim import Adam
import torch.nn as nn
from sklearn.metrics import classification_report
from data_other import REL2ID_DICT, NONE_REL_DICT
import os
import warnings
import sys
from pathlib import Path
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
    result_collection = classification_report(label_list, pred_list, output_dict=True, target_names=[ID2REL[i] for i in range(len(ID2REL)) if i not in EVAL_EXCLUDE_ID], labels=[i for i in range(len(ID2REL)) if i not in EVAL_EXCLUDE_ID])
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
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--dataname", default="hievents", type=str)
    parser.add_argument("--ignore_nonetype", action="store_true")
    parser.add_argument("--load_ckpt", default=None, type=str)


    args = parser.parse_args()

    if args.dataname == "TCR":
        assert args.eval_only == True
    
    if not args.load_ckpt and args.eval_only:
        print("WARNING: load_ckpt is not set! Will eval on pre-finetuned model. Is it intended?")

    REL2ID = REL2ID_DICT[args.dataname.lower()]
    if args.ignore_nonetype and args.dataname.lower() in ["matres", "tcr"]:
        assert REL2ID[NONE_REL_DICT[args.dataname.lower()]] == len(REL2ID) - 1
        del REL2ID[NONE_REL_DICT[args.dataname.lower()]]
    ID2REL = {v:k for k, v in REL2ID.items()}

    EVAL_EXCLUDE_ID = [REL2ID["NONE"], REL2ID["Coref"]]

    OUTPUT_DIR = Path(f"./output/{args.seed}/{args.dataname}_ignore_none_{args.ignore_nonetype}")
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    sys.stdout = open(os.path.join(OUTPUT_DIR, "log.txt"), "w")

    set_seed(args.seed)
    
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    data_dir = f"../data/processed/{args.dataname}"
    print("loading data...")
    if not args.eval_only:
        train_dataloader = get_dataloader(tokenizer, "train", args.dataname, data_dir=data_dir, max_length=256, shuffle=True, batch_size=args.batch_size, ignore_nonetype=args.ignore_nonetype)
        dev_dataloader = get_dataloader(tokenizer, "dev", args.dataname, data_dir=data_dir, max_length=256, shuffle=False, batch_size=args.batch_size, ignore_nonetype=args.ignore_nonetype)
    test_dataloader = get_dataloader(tokenizer, "test", args.dataname, data_dir=data_dir, max_length=256, shuffle=False, batch_size=args.batch_size, ignore_nonetype=args.ignore_nonetype)

    print("loading model...")
    model = Model(len(tokenizer), out_dim=len(REL2ID))
    model = to_cuda(model)

    if not args.eval_only:
        bert_optimizer = AdamW([p for p in model.encoder.model.parameters() if p.requires_grad], lr=args.bert_lr)
        optimizer = Adam([p for p in model.scorer.parameters() if p.requires_grad], lr=args.lr)

        scheduler = get_linear_schedule_with_warmup(bert_optimizer, num_warmup_steps=200, num_training_steps=len(train_dataloader) * args.epochs)
    eps = 1e-8

    if args.load_ckpt:
        print("loading from %s" % (args.load_ckpt))
        state = torch.load(args.load_ckpt)
        model.load_state_dict(state["model"])
        if not args.eval_only:
            print("resume from checkpoint", args.load_ckpt)
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])

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
                    res = classification_report(label_list, pred_list, output_dict=True, target_names=[ID2REL[i] for i in range(len(ID2REL)) if i not in EVAL_EXCLUDE_ID], labels=[i for i in range(len(ID2REL)) if i not in EVAL_EXCLUDE_ID])
                    print("Train result:", res)
                    
                    train_losses = []
                    pred_list = []
                    label_list = []

                if glb_step % args.eval_steps == 0:
                    res = evaluate(model, dev_dataloader, desc="Validation")
                    print("Validation result: ", res)
                    if "micro avg" not in res:
                        current_score = res["accuracy"]
                    else:
                        current_score = res["micro avg"]["f1-score"]
                    if current_score > best_score:
                        print("best result!")
                        best_score = current_score
                        state = {"model":model.state_dict(), "optimizer":optimizer.state_dict(), "scheduler": scheduler.state_dict()}
                        torch.save(state, os.path.join(OUTPUT_DIR, "best"))

    
    print("*" * 30 + "Test"+ "*" * 30)

    if not args.eval_only or not args.load_ckpt:
        if os.path.exists(os.path.join(OUTPUT_DIR, "best")):
            print("loading from", os.path.join(OUTPUT_DIR, "best"))
            state = torch.load(os.path.join(OUTPUT_DIR, "best"))
            model.load_state_dict(state["model"])
    res = evaluate(model, test_dataloader, desc="Test")
    with open(os.path.join(OUTPUT_DIR, "test_result.json"), "w", encoding="utf-8")as f:
        json.dump(res, f, indent=4)

    sys.stdout.close()