import sys
sys.path.append("../")
from utils.utils import to_cuda
from utils.model import Model
import torch
import random
import numpy as np
from tqdm import tqdm
from src.data_other import get_dataloader, myDataset, collator
from transformers import AdamW, RobertaTokenizer, get_linear_schedule_with_warmup
import argparse
from torch.optim import Adam
import torch.nn as nn
from sklearn.metrics import classification_report
from src.data_other import REL2ID_DICT, NONE_REL_DICT
import os
import warnings
import sys
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
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
    result_collection = classification_report(label_list, pred_list, output_dict=True, target_names=[ID2REL[i] for i in range(len(ID2REL)) if i != IGNORE_ID], labels=[i for i in range(len(ID2REL)) if i != IGNORE_ID])
    return result_collection



if __name__ == "__main__":
    import json
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_steps", default=50, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--log_steps", default=50, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--bert_lr", default=1e-5, type=float)
    parser.add_argument("--batch_size", default=8, type=int)
    # parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--dataname", default="EventStoryLine", type=str)
    parser.add_argument("--ignore_nonetype", action="store_true")
    parser.add_argument("--K", default=5, type=int, help="K-fold cross validation")


    args = parser.parse_args()

    REL2ID = REL2ID_DICT[args.dataname.lower()]
    ID2REL = {v:k for k, v in REL2ID.items()}
    IGNORE_ID = REL2ID[NONE_REL_DICT[args.dataname.lower()]]
    OUTPUT_DIR = Path(f"./output/{args.seed}/{args.dataname}_ignore_none_{args.ignore_nonetype}")
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    sys.stdout = open(os.path.join(OUTPUT_DIR, "log.txt"), "w")

    set_seed(args.seed)
    
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    data_dir = f"../data/processed/{args.dataname}"

    print("loading data...")
    if args.dataname == "EventStoryLine":
        train_dataset = myDataset(tokenizer, data_dir=data_dir, split="train", dataname=args.dataname, max_length=256, ignore_nonetype=args.ignore_nonetype).tokenized_samples
        dev_dataloader = get_dataloader(tokenizer, "dev", args.dataname, data_dir=data_dir, max_length=256, shuffle=False, batch_size=args.batch_size, ignore_nonetype=args.ignore_nonetype)
    elif args.dataname == "CausalTimeBank":
        train_dataset = myDataset(tokenizer, data_dir=data_dir, split="CausalTimeBank", dataname=args.dataname, max_length=256, ignore_nonetype=args.ignore_nonetype).tokenized_samples


    test_results = {}

    kf = KFold(n_splits=args.K)
    for split_num, (train_idx, test_idx) in enumerate(kf.split(range(len(train_dataset)))):
        print(f"Training {split_num}-th split")
        print("loading data...")
        train_data = myDataset(tokenized_samples=[train_dataset[i] for i in train_idx])
        test_data = myDataset(tokenized_samples=[train_dataset[i] for i in test_idx])

        train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
        test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
        if args.dataname == "CausalTimeBank":
            print("test as dev")
            dev_dataloader = test_dataloader


        print("loading model...")
        model = Model(len(tokenizer), out_dim=len(REL2ID))
        model = to_cuda(model)
        bert_optimizer = AdamW([p for p in model.encoder.model.parameters() if p.requires_grad], lr=args.bert_lr)
        optimizer = Adam([p for p in model.scorer.parameters() if p.requires_grad], lr=args.lr)
        scheduler = get_linear_schedule_with_warmup(bert_optimizer, num_warmup_steps=200, num_training_steps=len(train_dataloader) * args.epochs)
        eps = 1e-8


        Loss = nn.CrossEntropyLoss(ignore_index=-100)
        glb_step = 0
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
                    res = classification_report(label_list, pred_list, output_dict=True, target_names=[ID2REL[i] for i in range(len(ID2REL)) if i != IGNORE_ID], labels=[i for i in range(len(ID2REL)) if i != IGNORE_ID])
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
                        torch.save(state, os.path.join(OUTPUT_DIR, f"best_split{split_num}"))

        
        print("*" * 30 + "Test"+ "*" * 30)
        if os.path.exists(os.path.join(OUTPUT_DIR, f"best_split{split_num}")):
            state = torch.load(os.path.join(OUTPUT_DIR, f"best_split{split_num}"))
            model.load_state_dict(state["model"])
        else:
            print("no best checkpoint saved, directly evaluate on last epoch!")
        res = evaluate(model, test_dataloader, desc="Test")
        
        if not test_results:
            for k in res:
                test_results[k] = res[k]
        else:
            for k in res:
                for m in res[k]:
                    test_results[k][m] += res[k][m]

        with open(os.path.join(OUTPUT_DIR, f"test_result_split{split_num}.txt"), "w", encoding="utf-8")as f:
            f.writelines(json.dumps(res, indent=4))

    
    for k in test_results:
        for m in test_results[k]:
            test_results[k][m] /= args.K 
    print("**********Test result*********\n", test_results)
    with open(os.path.join(OUTPUT_DIR, f"test_result.json"), "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=4)


    sys.stdout.close()