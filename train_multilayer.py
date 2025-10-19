import argparse, os, random, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from utils.datasets import RoutedFolder
from torch.cuda.amp import autocast, GradScaler

# Import custom model wrappers
from SkinBench.models.resnet_model import ResNet50
from SkinBench.models.DenseNet121 import DenseNet121Medical
from SkinBench.models.mobilenetv3 import MobileNetV3
from SkinBench.models.cnn_model import SimpleCNN
from SkinBench.models.hybridmodel import HybridViTCNNMLP
from SkinBench.models.hybridSwinDenseNetMLP import HybridSwinDenseNetMLP

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

MODELS = {
    'resnet50': lambda nc: ResNet50(num_classes=nc, in_chans=3),
    'densenet121': lambda nc: DenseNet121Medical(num_classes=nc, in_chans=3),
    'mobilenetv3': lambda nc: MobileNetV3(num_classes=nc, in_chans=3),
    'cnn': lambda nc: SimpleCNN(num_classes=nc, in_chans=3),
    'vit_resnet': lambda nc: HybridViTCNNMLP(num_classes=nc),
    'swin_densenet': lambda nc: HybridSwinDenseNetMLP(num_classes=nc),
}

def stratified_split(ds, seed=42, train=0.7, val=0.15):
    set_seed(seed)
    # use labels directly from ds.samples to avoid calling __getitem__ (which applies transforms)
    labels = np.array([cls for _, cls in ds.samples])
    idx = np.arange(len(labels))
    train_idx, val_idx, test_idx = [], [], []
    for c in np.unique(labels):
        cidx = idx[labels == c]
        np.random.shuffle(cidx)
        n = len(cidx); t = int(train*n); v = int(val*n)
        train_idx += cidx[:t].tolist()
        val_idx   += cidx[t:t+v].tolist()
        test_idx  += cidx[t+v:].tolist()
    return train_idx, val_idx, test_idx


def get_loaders(data_dir, phase, img=256, bs=32, seed=42):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # BEFORE (problematic): RandomErasing was before ToTensor()
    # aug = transforms.Compose([
    #     transforms.Resize((img,img)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomVerticalFlip(p=0.2),
    #     transforms.ColorJitter(0.1,0.1,0.1,0.05),
    #     transforms.RandomErasing(p=0.1),   # <-- expects tensor; img is still PIL
    #     transforms.ToTensor(), normalize])

    # AFTER (correct): ToTensor() first, then RandomErasing()
    aug = transforms.Compose([
        transforms.Resize((img, img)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.1),
        normalize
    ])

    plain = transforms.Compose([
        transforms.Resize((img, img)),
        transforms.ToTensor(),
        normalize
    ])

    full = RoutedFolder(data_dir, transform=aug, phase=phase)
    tr,va,te = stratified_split(full, seed=seed)
    train_ds = torch.utils.data.Subset(full, tr)
    val_ds   = torch.utils.data.Subset(RoutedFolder(data_dir, transform=plain, phase=phase), va)
    test_ds  = torch.utils.data.Subset(RoutedFolder(data_dir, transform=plain, phase=phase), te)

    # weights for CE
    ytr = np.array([full[i][1] for i in tr])
    counts = np.bincount(ytr, minlength=len(full.classes))
    w = counts.max()/np.clip(counts,1,None)
    class_w = torch.tensor(w, dtype=torch.float32)

    tl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)
    vl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=2)
    te = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=2)
    return tl, vl, te, class_w, full.classes

def evaluate(model, loader, device):
    model.eval(); preds=[]; gts=[]
    with torch.inference_mode():
        for x,y in loader:
            x=x.to(device); y=y.to(device)
            p = model(x).softmax(1).argmax(1)
            preds += p.cpu().tolist(); gts += y.cpu().tolist()
    macroF1 = f1_score(gts, preds, average='macro')
    acc = (np.array(preds)==np.array(gts)).mean()
    return acc, macroF1, classification_report(gts, preds, digits=4, zero_division=0), confusion_matrix(gts, preds)

def main(args):
    set_seed(args.seed)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    tl,vl,te,class_w,classes = get_loaders(args.data_dir, args.phase, args.img_size, args.batch_size, args.seed)
    model = MODELS[args.model](len(classes)).to(device)

    crit = nn.CrossEntropyLoss(weight=class_w.to(device))
    opt  = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', patience=3, factor=0.3)
    scaler = GradScaler(enabled=(device=='cuda'))

    best=-1; noimp=0
    run_dir = os.path.join(args.out, args.phase, args.model); os.makedirs(run_dir, exist_ok=True)

    for ep in range(args.epochs):
        model.train()
        for x,y in tl:
            x=x.to(device); y=y.to(device)
            opt.zero_grad()
            with autocast(enabled=(device=='cuda')):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
        acc,f1,_,_ = evaluate(model, vl, device)
        sch.step(f1)
        print(f"Epoch {ep+1}: val_acc={acc:.4f} val_macroF1={f1:.4f}")
        if f1>best:
            best=f1; noimp=0
            torch.save(model.state_dict(), os.path.join(run_dir, "best.pt"))
        else:
            noimp+=1
            if noimp>=args.early_stop: break

    # test
    model.load_state_dict(torch.load(os.path.join(run_dir,'best.pt'), map_location=device))
    acc,f1,rep,cm = evaluate(model, te, device)
    print(rep)
    lb_path = os.path.join(args.out,'leaderboard.csv')
    import pandas as pd
    row = pd.DataFrame([{'phase':args.phase,'model':args.model,'img':args.img_size,'batch':args.batch_size,'seed':args.seed,'val_macroF1':best,'test_acc':acc,'test_macroF1':f1}])
    if os.path.exists(lb_path): row.to_csv(lb_path, mode='a', header=False, index=False)
    else: row.to_csv(lb_path, index=False)
    import pandas as pd
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(os.path.join(args.out, f'confusion_{args.phase}_{args.model}.csv'))

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--phase', default='ALL9', choices=['ALL9','L1','L2'])
    ap.add_argument('--model', default='resnet50', choices=list(MODELS.keys()))
    ap.add_argument('--img_size', type=int, default=256)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--early_stop', type=int, default=8)
    ap.add_argument('--out', default='runs')
    ap.add_argument('--seed', type=int, default=42)
    args=ap.parse_args()
    main(args)
