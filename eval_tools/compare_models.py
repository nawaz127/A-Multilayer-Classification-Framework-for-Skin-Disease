import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse, os, pandas as pd
if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--table', default='runs/analysis/comparison_all_models.csv')
    args = ap.parse_args()
    if not os.path.exists(args.table):
        print('comparison table not found. Run run_all_evals.py first.')
    else:
        df = pd.read_csv(args.table)
        best = df.sort_values(['phase','test_macroF1'], ascending=[True,False]).groupby('phase').head(3)
        print('\nTop-3 per phase:\n')
        print(best.to_string(index=False))
