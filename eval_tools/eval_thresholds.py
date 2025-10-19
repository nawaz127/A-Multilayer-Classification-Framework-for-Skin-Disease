import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse, os, numpy as np, pandas as pd, torch
from eval_tools.shared_eval_utils import make_loader
from models.resnet_model import ResNet50        # L1 gate
from models.DenseNet121 import DenseNet121Medical  # L2

from sklearn.metrics import f1_score

def _load(factory, nc, ckpt, device):
    m = factory(nc).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device))
    m.eval()
    return m

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--l1_ckpt', default='runs/L1/resnet50/best.pt')
    ap.add_argument('--l2_ckpt', default='runs/L2/densenet121/best.pt')
    ap.add_argument('--tau_min', type=float, default=0.50)
    ap.add_argument('--tau_max', type=float, default=0.80)
    ap.add_argument('--tau_step', type=float, default=0.05)
    ap.add_argument('--img', type=int, default=256)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--runs_dir', default='runs')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loader, classes_all9 = make_loader(args.data_dir, 'ALL9', split='test', img=args.img, batch=args.batch)
    normal_idx = classes_all9.index('Normal')
    l2_labels = [c for c in classes_all9 if c != 'Normal']

    l1 = _load(ResNet50, 2, args.l1_ckpt, device)
    l2 = _load(DenseNet121Medical, len(l2_labels), args.l2_ckpt, device)

    taus = np.arange(args.tau_min, args.tau_max + 1e-9, args.tau_step)
    rows=[]
    with torch.inference_mode():
        for tau in taus:
            preds=[]; gts=[]
            for batch in loader:
                if isinstance(batch, (list, tuple)) and len(batch)==3: x,y,_ = batch
                else: x,y = batch
                x = x.to(device)
                p_normal = l1(x).softmax(1)[:,0]
                for i in range(x.size(0)):
                    if p_normal[i] >= tau:
                        preds.append(normal_idx)
                    else:
                        q = l2(x[i:i+1]).softmax(1)[0]
                        cls8 = int(q.argmax().item())
                        preds.append(classes_all9.index(l2_labels[cls8]))
                gts += y.tolist()
            preds = np.array(preds); gts = np.array(gts)
            acc = float((preds == gts).mean())
            macroF1 = float(f1_score(gts, preds, average='macro'))
            print(f"tau={tau:.2f} acc={acc:.4f} macroF1={macroF1:.4f}")
            rows.append({'tau': float(tau), 'acc': acc, 'macroF1': macroF1})

    os.makedirs(os.path.join(args.runs_dir, 'analysis'), exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(args.runs_dir, 'analysis', 'tau_sweep_results.csv'), index=False)
