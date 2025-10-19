import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os, argparse, numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from utils.datasets import RoutedFolder
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from eval_tools.roc_auc import run_roc_auc

# your backbones
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

def _loader(data_dir, phase, split='test', img=256, batch=32):
    normalize = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    T = transforms.Compose([transforms.Resize((img,img)), transforms.ToTensor(), normalize])
    ds = RoutedFolder(data_dir, transform=T, phase=phase, keep_paths=True)
    labels = np.array([c for _, c in ds.samples])
    idx = np.arange(len(ds.samples)); rng = np.random.default_rng(42)
    tr,va,te = [],[],[]
    for c in np.unique(labels):
        cidx = idx[labels==c]; rng.shuffle(cidx)
        n=len(cidx); t=int(0.7*n); v=int(0.15*n)
        tr += cidx[:t].tolist(); va += cidx[t:t+v].tolist(); te += cidx[t+v:].tolist()
    indices = tr if split=='train' else (va if split=='val' else te)
    loader = DataLoader(Subset(ds, indices), batch_size=batch, shuffle=False, num_workers=0)
    return loader, ds.classes

def _eval_ckpt(model_name, ckpt, data_dir, phase, img=256, batch=32, device=None):
    if device is None: device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loader, classes = _loader(data_dir, phase, 'test', img, batch)
    nc = len(classes)

    # Prepare path reconstruction if dataloader yields only (x,y)
    subset = loader.dataset
    rf = getattr(subset, 'dataset', None)
    indices = getattr(subset, 'indices', list(range(len(subset))))
    paths_in_order = [ (rf.samples[i][0] if rf is not None else '') for i in indices ]
    path_cursor = 0

    Model = MODELS[model_name]
    m = Model(num_classes=nc).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device)); m.eval()

    preds=[]; gts=[]; probs=[]; paths=[]
    with torch.inference_mode():
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch)==3:
                x,y,p = batch
                batch_paths = list(p)
            else:
                x,y = batch
                bs = len(y)
                batch_paths = paths_in_order[path_cursor:path_cursor+bs]
                path_cursor += bs

            x = x.to(device)
            pr = m(x).softmax(1).cpu().numpy()
            pp = pr.argmax(1)
            preds += pp.tolist(); gts += y.numpy().tolist()
            probs += pr.max(1).tolist(); paths += batch_paths

    acc = accuracy_score(gts, preds)
    macroF1 = f1_score(gts, preds, average='macro')
    rep = classification_report(gts, preds, digits=4, zero_division=0)
    cm = confusion_matrix(gts, preds)
    df = pd.DataFrame({'path':paths,'true':gts,'pred':preds,'prob':probs})
    return acc, macroF1, rep, cm, df, classes

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--runs_dir', default='runs')
    ap.add_argument('--img', type=int, default=256)
    ap.add_argument('--batch', type=int, default=32)
    args = ap.parse_args()

    os.makedirs(os.path.join(args.runs_dir,'analysis'), exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    phases = ['ALL9','L1','L2']
    rows=[]
    for phase in phases:
        phase_dir = os.path.join(args.runs_dir, phase)
        if not os.path.isdir(phase_dir):
            continue
        for model_name in MODELS.keys():
            ckpt = os.path.join(phase_dir, model_name, 'best.pt')
            if not os.path.exists(ckpt):  # skip untrained models
                continue
            print(f"[EVAL] phase={phase} model={model_name}")
            acc, f1, rep, cm, df, classes = _eval_ckpt(model_name, ckpt, args.data_dir, phase, args.img, args.batch, device)

            pred_path = os.path.join(args.runs_dir, phase, f'{model_name}_preds.csv')
            df.to_csv(pred_path, index=False)
            pd.DataFrame(cm, index=classes, columns=classes).to_csv(os.path.join(args.runs_dir, f'confusion_{phase}_{model_name}.csv'))
            rows.append({'phase':phase,'model':model_name,'test_acc':acc,'test_macroF1':f1})
            print(rep)

            roc_info = run_roc_auc(args.data_dir, phase, model_name, ckpt, args.img, args.batch, args.runs_dir)
            print(f"ROC/AUC saved: {roc_info}")

    if rows:
        cmp = pd.DataFrame(rows).sort_values(['phase','test_macroF1'], ascending=[True,False])
        cmp.to_csv(os.path.join(args.runs_dir,'analysis','comparison_all_models.csv'), index=False)
        print("Saved comparison_all_models.csv")
    else:
        print("No checkpoints found. Ensure runs/<phase>/<model>/best.pt exist.")
