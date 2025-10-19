import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os, glob, pandas as pd

out = ["# Results & Discussion (Auto-draft)"]

p = 'runs/analysis/comparison_all_models.csv'
if os.path.exists(p):
    df = pd.read_csv(p)
    best = df.sort_values(['phase','test_macroF1'], ascending=[True,False]).groupby('phase').head(3)
    out += ["\n## Top Models per Phase\n", best.to_markdown(index=False)]

p = 'runs/analysis/tau_sweep_results.csv'
if os.path.exists(p):
    ts = pd.read_csv(p)
    out += ["\n## Threshold Sensitivity (τ)\n", ts.to_markdown(index=False)]

p = 'runs/analysis/calibration_results.csv'
if os.path.exists(p):
    cr = pd.read_csv(p)
    out += ["\n## Calibration (Temperature Scaling)\n", cr.to_markdown(index=False)]

auc_tables = glob.glob('runs/analysis/auc*.csv')
if auc_tables:
    out.append("\n## ROC-AUC Tables\n")
    for a in auc_tables[:10]:
        out.append(f"**{os.path.basename(a)}**\n")
        out.append(pd.read_csv(a).to_markdown(index=False))
        out.append("\n")

out += [
"\n## Statistical Significance (McNemar)\nInsert chi-square and p-value.",
"\n## Explainability (Grad-CAM)\nInsert 3–5 heatmaps and insights.",
"\n## Error Analysis\nSummarize confusions and show misclassifications.",
"\n## Limitations & Future Work\nDiscuss domain shift and clinical context."
]

with open('runs/RESULTS_DISCUSSION_AUTODRAFT.md','w',encoding='utf-8') as f:
    f.write("\n".join(out))
print("Wrote runs/RESULTS_DISCUSSION_AUTODRAFT.md")
