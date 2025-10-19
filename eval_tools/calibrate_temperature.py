import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse, os, numpy as np, pandas as pd, torch
from sklearn.metrics import log_loss, f1_score, accuracy_score
from eval_tools.shared_eval_utils import make_loader
from models.DenseNet121 import DenseNet121Medical

def _gather_logits(model, loader, device):
    model.eval(); logits=[]; labels=[]
    with torch.inference_mode():
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch)==3: x,y,_ = batch
            else: x,y = batch
            x=x.to(device)
            logits.append(model(x).cpu().numpy())
            labels += y.tolist()
    return np.concatenate(logits, axis=0), np.array(labels)

def _ece(probs, labels, n_bins=15):
    confid = probs.max(1)
    pred = probs.argmax(1)
    acc = (pred==labels).astype(np.float32)
    bins = np.linspace(0,1,n_bins+1)
    ece=0.0
    for i in range(n_bins):
        m = (confid>bins[i]) & (confid<=bins[i+1])
        if m.any():
            ece += m.mean() * abs(acc[m].mean() - confid[m].mean())
    return float(ece)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--ckpt', default='runs/L2/densenet121/best.pt')
    ap.add_argument('--phase', default='L2', choices=['ALL9','L1','L2'])
    ap.add_argument('--img', type=int, default=256)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--runs_dir', default='runs')
    args = ap.parse_args()

    device='cuda' if torch.cuda.is_available() else 'cpu'
    val_loader, classes = make_loader(args.data_dir, args.phase, split='val', img=args.img, batch=args.batch)
    test_loader, _      = make_loader(args.data_dir, args.phase, split='test', img=args.img, batch=args.batch)

    nc = len(classes)
    model = DenseNet121Medical(num_classes=nc).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    val_logits, val_y = _gather_logits(model, val_loader, device)
    Ts = np.linspace(0.5, 3.0, 26)
    bestT, bestNLL = 1.0, 1e9
    for T in Ts:
        pv = torch.softmax(torch.tensor(val_logits/T), dim=1).numpy()
        nll = log_loss(val_y, pv, labels=list(range(pv.shape[1])))
        if nll < bestNLL: bestNLL, bestT = nll, T

    test_logits, test_y = _gather_logits(model, test_loader, device)
    pr_raw = torch.softmax(torch.tensor(test_logits),   dim=1).numpy()
    pr_cal = torch.softmax(torch.tensor(test_logits/bestT), dim=1).numpy()

    out = {
        'T': bestT,
        'acc_raw': float(accuracy_score(test_y, pr_raw.argmax(1))),
        'macroF1_raw': float(f1_score(test_y, pr_raw.argmax(1), average='macro')),
        'ECE_raw': _ece(pr_raw, test_y),
        'acc_cal': float(accuracy_score(test_y, pr_cal.argmax(1))),
        'macroF1_cal': float(f1_score(test_y, pr_cal.argmax(1), average='macro')),
        'ECE_cal': _ece(pr_cal, test_y),
    }
    os.makedirs(os.path.join(args.runs_dir, 'analysis'), exist_ok=True)
    pd.DataFrame([out]).to_csv(os.path.join(args.runs_dir, 'analysis', 'calibration_results.csv'), index=False)
    print(out)
