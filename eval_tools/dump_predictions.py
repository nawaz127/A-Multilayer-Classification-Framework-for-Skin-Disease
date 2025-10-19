import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse, os, torch, pandas as pd
from eval_tools.shared_eval_utils import make_loader
from models.resnet_model import ResNet50
from models.DenseNet121 import DenseNet121Medical

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--phase', default='L2', choices=['ALL9','L1','L2'])
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--model', default='densenet', choices=['resnet','densenet'])
    ap.add_argument('--split', default='test', choices=['train','val','test'])
    ap.add_argument('--out', default='runs/preds.csv')
    args = ap.parse_args()

    device='cuda' if torch.cuda.is_available() else 'cpu'
    loader, classes = make_loader(args.data_dir, args.phase, split=args.split)

    subset = loader.dataset
    rf = getattr(subset, 'dataset', None)
    indices = getattr(subset, 'indices', list(range(len(subset))))
    paths_in_order = [ (rf.samples[i][0] if rf is not None else '') for i in indices ]
    path_cursor = 0

    nc = len(classes)
    m = DenseNet121Medical(num_classes=nc).to(device) if args.model=='densenet' else ResNet50(num_classes=nc).to(device)
    m.load_state_dict(torch.load(args.ckpt, map_location=device)); m.eval()

    rows=[]
    with torch.inference_mode():
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch)==3:
                x,y,p = batch; batch_paths = list(p)
            else:
                x,y = batch; bs = len(y)
                batch_paths = paths_in_order[path_cursor:path_cursor+bs]; path_cursor += bs

            x=x.to(device)
            pr = m(x).softmax(1).cpu().numpy()
            pred = pr.argmax(1)
            for i in range(len(y)):
                rows.append({'path':batch_paths[i], 'true':int(y[i]), 'pred':int(pred[i]), 'prob':float(pr[i,pred[i]])})
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Saved {len(rows)} rows to {args.out}")
