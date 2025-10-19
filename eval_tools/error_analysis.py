import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse, os, pandas as pd

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True)
    ap.add_argument('--out', default='runs/analysis')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.pred)
    mis = df[df['true'] != df['pred']].copy()
    mis['pair'] = mis['true'].astype(str) + '→' + mis['pred'].astype(str)

    top3 = mis['pair'].value_counts().head(3)
    top3.to_csv(os.path.join(args.out, 'confusion_hotspots.txt'))

    mis.head(10).to_csv(os.path.join(args.out, 'error_samples.csv'), index=False)
    print("Wrote confusion_hotspots.txt and error_samples.csv")
