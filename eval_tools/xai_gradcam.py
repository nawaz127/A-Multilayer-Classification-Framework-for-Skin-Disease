import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse, os, numpy as np, torch, matplotlib.pyplot as plt
from eval_tools.shared_eval_utils import make_loader
from models.resnet_model import ResNet50
from models.DenseNet121 import DenseNet121Medical

def gradcam(model, x, target_layer):
    feats = None
    def fwd_hook(m, i, o):  # capture features
        nonlocal feats; feats = o
    h1 = target_layer.register_forward_hook(fwd_hook)

    grads = None
    def bwd_hook(m, gi, go):  # capture grads
        nonlocal grads; grads = go[0]
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    model.eval()
    x = x.clone().requires_grad_(True)
    out = model(x)
    cls = out.argmax(1)
    out.gather(1, cls.view(-1,1)).sum().backward()

    g = grads.detach().cpu().numpy()
    f = feats.detach().cpu().numpy()
    w = g.mean(axis=(2,3), keepdims=True)
    cam = (w * f).sum(axis=1)
    cam = np.maximum(cam, 0)
    cam -= cam.min(axis=(1,2), keepdims=True)
    cam /= (cam.max(axis=(1,2), keepdims=True) + 1e-6)

    h1.remove(); h2.remove()
    return cam

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--ckpt', required=True, help='checkpoint for the given phase/model')
    ap.add_argument('--model', default='densenet', choices=['resnet','densenet'])
    ap.add_argument('--phase', default='L2', choices=['ALL9','L1','L2'], help='match the checkpoint’s training phase')
    ap.add_argument('--img', type=int, default=256)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--out', default='XAI_results')
    args = ap.parse_args()

    device='cuda' if torch.cuda.is_available() else 'cpu'
    # IMPORTANT: build loader for the *same* phase as the checkpoint
    loader, classes = make_loader(args.data_dir, args.phase, split='test', img=args.img, batch=args.batch)
    first = next(iter(loader))
    if isinstance(first, (list, tuple)) and len(first)==3: x,y,paths = first
    else: x,y = first; paths=None
    x = x.to(device)

    nc = len(classes)
    if args.model=='resnet':
        m = ResNet50(num_classes=nc).to(device)
        m.load_state_dict(torch.load(args.ckpt, map_location=device))
        target = m.backbone.layer4
    else:
        m = DenseNet121Medical(num_classes=nc).to(device)
        m.load_state_dict(torch.load(args.ckpt, map_location=device))
        target = m.backbone.features.denseblock4

    cam = gradcam(m, x, target)
    os.makedirs(args.out, exist_ok=True)
    for i in range(min(8, cam.shape[0])):
        heat = cam[i]
        img = x[i].detach().cpu().permute(1,2,0).numpy()
        img = (img - img.min())/(img.max()-img.min()+1e-6)
        plt.figure(); plt.imshow(img); plt.imshow(heat, alpha=0.35); plt.axis('off'); plt.tight_layout()
        plt.savefig(os.path.join(args.out, f'cam_{args.phase}_{args.model}_{i}.png'), dpi=200); plt.close()
    print(f"Saved CAM overlays to {args.out}")
