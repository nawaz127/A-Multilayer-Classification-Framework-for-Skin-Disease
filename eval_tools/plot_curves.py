import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse, os, pandas as pd, numpy as np, matplotlib.pyplot as plt, glob
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs_dir', default='runs')
    args = ap.parse_args()
    os.makedirs(os.path.join(args.runs_dir,'figures'), exist_ok=True)

    cmp_path = os.path.join(args.runs_dir, 'analysis', 'comparison_all_models.csv')
    if os.path.exists(cmp_path):
        df = pd.read_csv(cmp_path)
        for phase in df['phase'].unique():
            sub = df[df['phase']==phase].sort_values('test_macroF1', ascending=False)
            plt.figure()
            plt.bar(np.arange(len(sub)), sub['test_macroF1'].values)
            plt.xticks(np.arange(len(sub)), sub['model'].values, rotation=45, ha='right')
            plt.ylabel('Test Macro-F1'); plt.title(f'Phase {phase}: Model Comparison')
            plt.tight_layout()
            plt.savefig(os.path.join(args.runs_dir,'figures',f'bar_{phase}.png'), dpi=200)
            plt.close()

    for csv in glob.glob(os.path.join(args.runs_dir,'confusion_*.csv')):
        cm = pd.read_csv(csv, index_col=0)
        plt.figure()
        im = plt.imshow(cm.values, aspect='auto')
        plt.colorbar(im)
        plt.xticks(range(len(cm.columns)), cm.columns, rotation=45, ha='right')
        plt.yticks(range(len(cm.index)), cm.index)
        plt.title(os.path.basename(csv).replace('.csv',''))
        plt.tight_layout()
        out = os.path.join(args.runs_dir,'figures', os.path.basename(csv).replace('.csv','.png'))
        plt.savefig(out, dpi=200); plt.close()
