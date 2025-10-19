import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

from eval_tools.shared_eval_utils import make_loader
from models.resnet_model import ResNet50
from models.DenseNet121 import DenseNet121Medical

def load_model(factory, num_classes, ckpt_path, device):
    m = factory(num_classes).to(device)
    m.load_state_dict(torch.load(ckpt_path, map_location=device))
    m.eval()
    return m

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--l1_ckpt", default="runs/L1/resnet50/best.pt")
    ap.add_argument("--l2_ckpt", default="runs/L2/densenet121/best.pt")
    ap.add_argument("--tau", type=float, default=0.70)
    ap.add_argument("--img", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default="runs/ALL9/multilayer_tau070_preds.csv")
    ap.add_argument("--save_png_cm", action="store_true", help="also save confusion matrix as PNG")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ALL9 evaluation split
    loader, classes_all9 = make_loader(args.data_dir, "ALL9", split="test", img=args.img, batch=args.batch)
    normal_idx = classes_all9.index("Normal")
    l2_labels = [c for c in classes_all9 if c != "Normal"]

    l1 = load_model(ResNet50, 2, args.l1_ckpt, device)
    l2 = load_model(DenseNet121Medical, len(l2_labels), args.l2_ckpt, device)

    # reconstruct paths from Subset → RoutedFolder.samples
    subset = loader.dataset
    rf = getattr(subset, "dataset", None)
    indices = getattr(subset, "indices", list(range(len(subset))))
    all_paths = [(rf.samples[i][0] if rf is not None else "") for i in indices]
    cursor = 0

    rows=[]; y_true=[]; y_pred=[]
    with torch.inference_mode():
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch)==3:
                x,y,batch_paths = batch; batch_paths = list(batch_paths)
            else:
                x,y = batch
                bs = len(y)
                batch_paths = all_paths[cursor:cursor+bs]; cursor += bs

            x = x.to(device)
            p_normal = l1(x).softmax(1)[:,0]
            for i in range(x.size(0)):
                yi = int(y[i])
                if p_normal[i] >= args.tau:
                    pred = normal_idx; prob = float(p_normal[i].cpu())
                else:
                    q = l2(x[i:i+1]).softmax(1)[0].cpu().numpy()
                    k = int(np.argmax(q)); pred = classes_all9.index(l2_labels[k]); prob = float(q[k])

                rows.append({"path": batch_paths[i], "true": yi, "pred": pred, "prob": prob})
                y_true.append(yi); y_pred.append(pred)

    # ensure runs/ paths
    out_csv = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"✅ Saved multilayer predictions to {out_csv} (τ={args.tau})")

    acc = accuracy_score(y_true, y_pred)
    macroF1 = f1_score(y_true, y_pred, average="macro")
    print("\n=== Multilayer (ALL9 test) ===")
    print(f"tau={args.tau:.2f}  acc={acc:.4f}  macroF1={macroF1:.4f}")
    print("\nClassification report (digits=4):")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    runs_dir = os.path.abspath(os.path.join(os.path.dirname(out_csv), '..'))  # → runs/
    os.makedirs(runs_dir, exist_ok=True)

    tau_tag = str(args.tau).replace('.', '')
    cm_csv = os.path.join(runs_dir, f"confusion_ALL9_multilayer_tau{tau_tag}.csv")
    pd.DataFrame(cm, index=classes_all9, columns=classes_all9).to_csv(cm_csv)
    print(f"🧩 Saved confusion matrix to {cm_csv}")

    analysis_dir = os.path.join(runs_dir, "analysis"); os.makedirs(analysis_dir, exist_ok=True)
    metrics_csv = os.path.join(analysis_dir, f"multilayer_tau{tau_tag}_metrics.csv")
    pd.DataFrame([{"tau": args.tau, "test_acc": acc, "test_macroF1": macroF1}]).to_csv(metrics_csv, index=False)
    print(f"📝 Saved metrics to {metrics_csv}")

    if args.save_png_cm:
        plt.figure()
        im = plt.imshow(cm, aspect='auto')
        plt.colorbar(im)
        plt.xticks(range(len(classes_all9)), classes_all9, rotation=45, ha='right')
        plt.yticks(range(len(classes_all9)), classes_all9)
        plt.title(f'confusion_ALL9_multilayer_tau{args.tau:.2f}')
        plt.tight_layout()
        fig_png = os.path.join(runs_dir, f"confusion_ALL9_multilayer_tau{tau_tag}.png")
        plt.savefig(fig_png, dpi=200); plt.close()
        print(f"🖼️  Saved confusion PNG to {fig_png}")
