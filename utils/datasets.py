import os, re, numpy as np
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Dataset

# Canonical label mapping to 9 classes
CANONICAL = {
    'acne': 'Acne',
    'bacterial infections (impetigo)': 'Bacterial/Impetigo',
    'bacterial infections': 'Bacterial/Impetigo',
    'impetigo': 'Bacterial/Impetigo',
    'eczema': 'Eczema',
    'atopic dermatitis': 'Eczema',
    'seborrhoeic dermatitis': 'Eczema',
    'fungal infections (tinea, candidiasis)': 'Fungal',
    'fungal infections': 'Fungal',
    'candidiasis': 'Fungal',
    'tinea': 'Fungal',
    'normal': 'Normal',
    'pigmentation disorders': 'Pigmentation',
    'pigmentation': 'Pigmentation',
    'pox': 'Pox',
    'chickenpox': 'Pox',
    'monkeypox': 'Pox',
    'psoriasis': 'Psoriasis',
    'scabies': 'Scabies',
}

LABELS_9 = ['Acne','Bacterial/Impetigo','Eczema','Fungal','Normal','Pigmentation','Pox','Psoriasis','Scabies']
IDX9 = {c:i for i,c in enumerate(LABELS_9)}

def _canon(name:str)->str:
    n = re.sub(r'\(.*?\)', '', name.lower()).strip()
    n = re.sub(r'\d+.*','', n).strip()
    return CANONICAL.get(n, CANONICAL.get(name.lower(), name))

class RoutedFolder(Dataset):
    """Wrap ImageFolder but map subfolders to 9 canonical labels.
    phase: 'L1' (Normal vs Abnormal), 'L2' (8 diseases), or 'ALL9' (9-class)
    """
    def __init__(self, root, transform=None, phase='ALL9', keep_paths=False):
        from torchvision.datasets import ImageFolder
        self.transform=transform; self.phase=phase; self.keep_paths=keep_paths
        tmp = ImageFolder(root)
        self.samples=[]
        for path,y in tmp.samples:
            parts = [p for p in path.split(os.sep) if p]
            label = None
            # search upward for a known class folder
            for k in reversed(parts[:-1]):
                cname = _canon(k)
                if cname in IDX9:
                    label = cname
                    break
            if label is None:
                label = _canon(tmp.classes[y])
                if label not in IDX9:
                    # skip unknowns
                    continue
            if phase=='ALL9':
                cls = IDX9[label]
            elif phase=='L1':
                cls = 0 if label=='Normal' else 1
            else: # L2
                if label=='Normal':   # L2 excludes normals
                    continue
                l2_labels=[c for c in LABELS_9 if c!='Normal']
                cls = l2_labels.index(label)
            self.samples.append((path, cls))
        self.classes = LABELS_9 if phase=='ALL9' else (['Normal','Abnormal'] if phase=='L1' else [c for c in LABELS_9 if c!='Normal'])

    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        p,y = self.samples[i]
        img = Image.open(p).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, y
