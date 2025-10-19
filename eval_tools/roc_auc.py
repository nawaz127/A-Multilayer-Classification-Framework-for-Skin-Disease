import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse, os, numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
from utils.datasets import RoutedFolder

from models.resnet_model import ResNet50
from models.DenseNet121 import DenseNet121Medical
from models.mobilenetv3 import MobileNetV3
from models.cnn_model import SimpleCNN
from models.hybridmodel import HybridViTCNNMLP
from models.hybridSwinDenseNetMLP import HybridSwinDenseNetMLP

MODELS = {
    'resnet50': ResNet50,
    'densenet121': DenseNet121Medical,
    'mobilenetv3': MobileNetV3,
    'cnn': SimpleCNN,
    'vit_resnet': HybridViTCNNMLP,
    'swin_densenet': HybridSwinDenseNetMLP,
}

def _get_loader(data_dir, phase, split='test', img=256, batch=32):
    norm = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    T = transforms.Compose([transforms.Resize((img,img)), transforms.ToTensor(), norm])
    ds = RoutedFolder(data_dir, transform=T, phase=phase, keep_paths=True)
    labels = np.array([c for _,c in ds.samples])
    idx = np.arange(len(ds.samples)); rng = np.random.default_rng(42)
    tr,va,te = [],[],[]
    for c in np.unique(labels):
        cidx = idx[labels==c]; rng.shuffle(cidx)
        m=len(cidx); t=int(0.7*m); v=int(0.15*m)
        tr += cidx[:t].tolist(); va += cidx[t:t+v].tolist(); te += cidx[t+v:].tolist()
    indices = tr if split=='train' else (va if split=='val' else te)
    return DataLoader(Subset(ds, indices), batch_size=batch, shuffle=False, num_workers=0), ds.classes

def _collect_logits(model, loader, device):
    model.eval(); logits=[]; labels=[]
    with torch.inference_mode():
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch)==3: x,y,_ = batch
            else: x,y = batch
            x=x.to(device)
            logits.append(model(x).cpu().numpy())
            labels += y.tolist()
    return np.concatenate(logits, axis=0), np.array(labels)

def run_roc_auc(data_dir, phase, model_name, ckpt, img=256, batch=32, runs_dir='runs'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loader, classes = _get_loader(data_dir, phase, 'test', img, batch)
    nc = len(classes)
    Model = MODELS[model_name]
    m = Model(num_classes=nc).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device))

    logits, y = _collect_logits(m, loader, device)
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()

    os.makedirs(os.path.join(runs_dir,'figures'), exist_ok=True)
    os.makedirs(os.path.join(runs_dir,'analysis'), exist_ok=True)

    if nc == 2:
        fpr, tpr, _ = roc_curve(y, probs[:,1], pos_label=1)
        au = auc(fpr, tpr)
        plt.figure(); plt.plot(fpr, tpr, label=f'AUC={au:.3f}'); plt.plot([0,1],[0,1],'--')
        plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'ROC (Phase {phase}, {model_name})')
        plt.legend(loc='lower right'); plt.tight_layout()
        png = os.path.join(runs_dir,'figures', f'roc_{phase}_{model_name}.png')
        plt.savefig(png, dpi=200); plt.close()
        pd.DataFrame([{'phase':phase,'model':model_name,'AUC':au}]).to_csv(
            os.path.join(runs_dir,'analysis', f'auc_{phase}_{model_name}.csv'), index=False)
        return {'AUC': float(au), 'png': png}
    else:
        y_ovr = np.eye(nc)[y]
        aucs = {}
        plt.figure()
        for c in range(nc):
            fpr, tpr, _ = roc_curve(y_ovr[:,c], probs[:,c])
            au = auc(fpr, tpr); aucs[classes[c]] = au
            plt.plot(fpr, tpr, label=f'{classes[c]} (AUC={au:.3f})')
        plt.plot([0,1],[0,1],'--')
        plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'OVR ROC (Phase {phase}, {model_name})')
        plt.legend(fontsize=8); plt.tight_layout()
        png = os.path.join(runs_dir,'figures', f'roc_ovr_{phase}_{model_name}.png')
        plt.savefig(png, dpi=200); plt.close()

        macro = float(np.mean(list(aucs.values())))
        micro = float(roc_auc_score(y, probs, multi_class='ovr'))
        rows = [{'class':k, 'AUC':float(v)} for k,v in aucs.items()] + \
               [{'class':'macro','AUC':macro},{'class':'micro','AUC':micro}]
        pd.DataFrame(rows).to_csv(os.path.join(runs_dir,'analysis', f'auc_ovr_{phase}_{model_name}.csv'), index=False)
        return {'macro': macro, 'micro': micro, 'png': png}

if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--phase', required=True, choices=['ALL9','L1','L2'])
    ap.add_argument('--model', required=True, choices=list(MODELS.keys()))
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--img', type=int, default=256)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--runs_dir', default='runs')
    args=ap.parse_args()
    print(run_roc_auc(args.data_dir, args.phase, args.model, args.ckpt, args.img, args.batch, args.runs_dir))
