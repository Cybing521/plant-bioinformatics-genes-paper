#!/usr/bin/env python3
"""CNN 区域重要性：与 AgroNT 相同的滑窗 N 扰动协议（同法归因）。"""
import argparse, json, logging, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
from utils import get_device, set_seed, setup_logging
from dataloader import onehot_to_seq
logger = logging.getLogger("plantdl")

BASE = {"A":0,"C":1,"G":2,"T":3}

def seq_to_oh(seq):
    x = np.zeros((4, len(seq)), np.float32)
    for i,b in enumerate(seq.upper()):
        j = BASE.get(b)
        if j is not None: x[j,i]=1.0
    return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--ckpt", default="results/models/cnn_npz_binary.pt")
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--step", type=int, default=25)
    ap.add_argument("--out-csv", default="results/interpret/cnn_region_mutagenesis.csv")
    args = ap.parse_args()
    import yaml
    with open(args.config) as f: cfg=yaml.safe_load(f)
    setup_logging(); set_seed(cfg["seed"]); device=get_device(cfg["train"]["device"])
    from models import build_cnn
    ck=torch.load(args.ckpt, map_location=device)
    model=build_cnn(cfg,1,"binary").to(device)
    model.load_state_dict(ck["model"]); model.eval()
    data=np.load(f"{cfg['data']['out_dir']}/dataset.npz")
    meta=__import__("pandas").read_csv(f"{cfg['data']['out_dir']}/meta.tsv", sep="\t")
    idx=meta.index[meta["split"]=="test"].values
    X=data["X"][idx]
    n=min(args.n_test, len(X))
    L=X.shape[2]
    profile=np.zeros(L)
    with torch.no_grad():
        for i in range(n):
            seq=onehot_to_seq(X[i])
            x0=torch.from_numpy(seq_to_oh(seq)).unsqueeze(0).to(device)
            base=torch.sigmoid(model(x0)).item()
            for start in range(0, L-args.window+1, args.step):
                end=start+args.window
                mut=seq[:start]+"N"*args.window+seq[end:]
                xm=torch.from_numpy(seq_to_oh(mut)).unsqueeze(0).to(device)
                p=torch.sigmoid(model(xm)).item()
                profile[start:end]+=abs(p-base)
            if (i+1)%20==0: logger.info("CNN mut %d/%d", i+1, n)
    profile/=n
    bounds=json.load(open(f"{cfg['data']['out_dir']}/region_bounds.json"))
    rows={}
    for reg in ["promoter","utr5","gap","utr3","terminator"]:
        s,e=bounds[reg]; Lr=e-s
        if Lr>0: rows[reg]=float(profile[s:e].sum()/Lr)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    import csv
    with open(args.out_csv,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["region","importance_per_bp","method","window","step","n"])
        for k,v in rows.items():
            w.writerow([k,v,"sliding_window_N",args.window,args.step,n])
    logger.info("CNN mutagenesis region: %s -> %s", rows, args.out_csv)

if __name__=="__main__":
    main()
