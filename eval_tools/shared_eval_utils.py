import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

"""
Shared helpers for deterministic splits + loaders.
- 70/15/15 stratified split (seed=42)
- ImageNet normalization to match pretrained backbones
- keep_paths=True so we can save file paths in preds
"""
from typing import Tuple, List
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from utils.datasets import RoutedFolder

def make_splits(ds: RoutedFolder, seed: int = 42, train: float = 0.7, val: float = 0.15):
    rng = np.random.default_rng(seed)
    labels = np.array([cls for _, cls in ds.samples])
    idx = np.arange(len(ds.samples))
    tr, va, te = [], [], []
    for c in np.unique(labels):
        cidx = idx[labels == c]
        rng.shuffle(cidx)
        n = len(cidx); t = int(train*n); v = int(val*n)
        tr += cidx[:t].tolist()
        va += cidx[t:t+v].tolist()
        te += cidx[t+v:].tolist()
    return tr, va, te

def make_loader(
    data_dir: str,
    phase: str,
    split: str = "test",
    img: int = 256,
    batch: int = 32,
    seed: int = 42
) -> Tuple[DataLoader, List[str]]:
    normalize = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    T = transforms.Compose([transforms.Resize((img, img)), transforms.ToTensor(), normalize])
    ds = RoutedFolder(data_dir, transform=T, phase=phase, keep_paths=True)
    tr, va, te = make_splits(ds, seed=seed)
    indices = tr if split=="train" else (va if split=="val" else te)
    loader = DataLoader(Subset(ds, indices), batch_size=batch, shuffle=False, num_workers=0)
    return loader, ds.classes
