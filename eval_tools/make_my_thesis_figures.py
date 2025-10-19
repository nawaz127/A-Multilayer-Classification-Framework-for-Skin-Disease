import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import subprocess
from pathlib import Path
import shutil
import glob
import pandas as pd

def run(cmd_list):
    """Run a command (list of args). Raise if it fails, and echo it."""
    print("\n>>>", " ".join(cmd_list))
    subprocess.run(cmd_list, check=False)  # keep going even if a step is skipped/errs

def first_existing(*paths):
    for p in paths:
        if Path(p).exists():
            return str(Path(p))
    return None

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--runs_dir', default='runs')
    args = ap.parse_args()

    # Ensure we run from project root (SkinBench)
    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)

    py = args.python
    data_dir_abs = str(Path(args.data_dir).resolve())
    runs_dir = Path(args.runs_dir)
    (runs_dir / 'figures').mkdir(parents=True, exist_ok=True)
    (runs_dir / 'analysis').mkdir(parents=True, exist_ok=True)
    (runs_dir / 'tables').mkdir(parents=True, exist_ok=True)

    # 1) Evaluate everything (+ ROC/AUC)
    run([py, "-m", "eval_tools.run_all_evals", "--data_dir", data_dir_abs])

    # 2) Bar charts + confusion PNGs
    run([py, "-m", "eval_tools.plot_curves"])

    # 3) τ sweep if both L1 and L2 best checkpoints exist
    l1 = first_existing('runs/L1/resnet50/best.pt')
    l2 = first_existing('runs/L2/densenet121/best.pt', 'runs/L2/resnet50/best.pt')
    if l1 and l2:
        run([py, "-m", "eval_tools.eval_thresholds",
             "--data_dir", data_dir_abs,
             "--l1_ckpt", l1,
             "--l2_ckpt", l2])
    else:
        print("[INFO] Skipping τ sweep (L1/L2 best not both present).")

    # 3.5) Multilayer export (tau=0.70) + McNemar vs best flat ALL9
    ml_preds = Path('runs/ALL9/multilayer_tau070_preds.csv')
    if not ml_preds.exists():
        run([py, "-m", "eval_tools.dump_multilayer_preds",
             "--data_dir", data_dir_abs, "--tau", "0.70",
             "--l1_ckpt", "runs/L1/resnet50/best.pt",
             "--l2_ckpt", "runs/L2/densenet121/best.pt",
             "--out", str(ml_preds),
             "--save_png_cm"])

    flat_csv = Path('runs/ALL9/swin_densenet_preds.csv')
    if flat_csv.exists():
        run([py, "-m", "eval_tools.stats_mcnemar",
             "--pred1", str(ml_preds),
             "--pred2", str(flat_csv),
             "--name1", "multilayer_tau070",
             "--name2", "swin_densenet"])
    else:
        print("[INFO] Skipping McNemar: flat ALL9 preds not found.")

    # 4) Calibration on best L2 from comparison table (if exists)
    cmp_csv = runs_dir / "analysis" / "comparison_all_models.csv"
    if cmp_csv.exists():
        df = pd.read_csv(cmp_csv)
        top = df[df['phase'] == 'L2'].sort_values('test_macroF1', ascending=False)
        if len(top):
            best = top.iloc[0]['model']
            ckpt = Path('runs/L2') / best / 'best.pt'
            if ckpt.exists():
                run([py, "-m", "eval_tools.calibrate_temperature",
                     "--data_dir", data_dir_abs,
                     "--ckpt", str(ckpt),
                     "--phase", "L2"])

    # 5) Auto-draft Results/Discussion
    run([py, "-m", "eval_tools.generate_results_section"])

    # 6) Copy important CSVs into runs/tables
    for p in glob.glob(str(runs_dir / "analysis" / "*.csv")) + glob.glob(str(runs_dir / "confusion_*.csv")):
        dst = runs_dir / "tables" / Path(p).name
        try:
            shutil.copyfile(p, dst)
        except Exception as e:
            print(f"[WARN] {e}")

    print("\nALL DONE — Figures in runs/figures, tables in runs/tables, R&D draft at runs/RESULTS_DISCUSSION_AUTODRAFT.md")
