import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from pathlib import Path
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
import os

def _coerce_and_norm_paths(df: pd.DataFrame) -> pd.DataFrame:
    """Make 'path' joinable: string, normalized separators, lowercase."""
    if 'path' in df.columns:
        df['path'] = df['path'].astype(str).fillna('').map(
            lambda p: os.path.normpath(p).lower() if p else ''
        )
    return df

def _mcnemar_from_df(df: pd.DataFrame):
    # contingency off-diagonals
    c01 = int(((df['pred_1']==df['true']) & (df['pred_2']!=df['true'])).sum())
    c10 = int(((df['pred_1']!=df['true']) & (df['pred_2']==df['true'])).sum())
    n_pairs = int(len(df))
    if (c01 + c10) == 0:
        return {'n': n_pairs, 'c01': c01, 'c10': c10, 'chi2': 0.0, 'p': 1.0, 'note': 'no off-diagonal counts'}
    res = mcnemar([[0, c01],[c10, 0]], exact=False, correction=True)
    return {'n': n_pairs, 'c01': c01, 'c10': c10, 'chi2': float(res.statistic), 'p': float(res.pvalue), 'note': ''}

def mcnemar_robust(a: pd.DataFrame, b: pd.DataFrame):
    a = _coerce_and_norm_paths(a.copy())
    b = _coerce_and_norm_paths(b.copy())

    # Try path-merge first
    if ('path' in a.columns) and ('path' in b.columns) and (not a['path'].eq('').all()) and (not b['path'].eq('').all()):
        m = a.merge(b, on='path', suffixes=('_1','_2'))
        if not m.empty:
            if 'true_1' in m.columns:
                m = m.rename(columns={'true_1':'true'})
            elif 'true' not in m.columns and 'true_2' in m.columns:
                m = m.rename(columns={'true_2':'true'})
            out = _mcnemar_from_df(m[['true','pred_1','pred_2']])
            out['note'] = (out.get('note','') + ' (path-merge)').strip()
            return out

    # Fallback: row-order alignment (valid since splits are deterministic)
    n = min(len(a), len(b))
    if n == 0:
        return {'n': 0, 'c01': 0, 'c10': 0, 'chi2': 0.0, 'p': 1.0, 'note': 'no rows to compare'}
    df = pd.DataFrame({
        'true':   a['true'].iloc[:n].values,
        'pred_1': a['pred'].iloc[:n].values,
        'pred_2': b['pred'].iloc[:n].values
    })
    out = _mcnemar_from_df(df)
    out['note'] = (out.get('note','') + ' (row-order fallback)').strip()
    return out

def infer_name(path: str) -> str:
    return Path(path).stem

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred1', required=True)
    ap.add_argument('--pred2', required=True)
    ap.add_argument('--name1', default=None)
    ap.add_argument('--name2', default=None)
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--runs_dir', default='runs')
    args = ap.parse_args()

    name1 = args.name1 or infer_name(args.pred1)
    name2 = args.name2 or infer_name(args.pred2)

    a = pd.read_csv(args.pred1)
    b = pd.read_csv(args.pred2)

    out = mcnemar_robust(a, b)
    significant = (out['p'] < args.alpha) if out['n'] > 0 and (out['c01']+out['c10'])>0 else False

    row = {
        'model_1': name1, 'model_2': name2,
        'n': out['n'],
        'c01_(m1_correct_m2_wrong)': out['c01'],
        'c10_(m1_wrong_m2_correct)': out['c10'],
        'chi2': out['chi2'], 'p_value': out['p'],
        'alpha': args.alpha, 'significant_(p<alpha)': significant,
        'note': out.get('note','')
    }
    print(row)

    # Save to runs/analysis and runs/tables
    analysis = Path(args.runs_dir)/'analysis'; tables = Path(args.runs_dir)/'tables'
    analysis.mkdir(parents=True, exist_ok=True); tables.mkdir(parents=True, exist_ok=True)
    base = f"mcnemar_{name1}_VS_{name2}.csv"
    pd.DataFrame([row]).to_csv(analysis/base, index=False)
    pd.DataFrame([row]).to_csv(tables/base, index=False)
