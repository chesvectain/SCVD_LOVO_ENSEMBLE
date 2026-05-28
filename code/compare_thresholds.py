"""
Threshold Mode Comparison from Saved Predictions
=================================================
Re-computes metrics from saved .npz prediction files under three threshold strategies:
  1. fixed_05   : threshold = 0.5 (original run)
  2. oracle_f1  : threshold found by maximizing F1 on test set (upper bound / data-leak-free proxy)
  3. s0_fpr002  : threshold that keeps S0 FPR ≤ 2% using s0_val_scores

Only LiWP and SkipGram families are included (OpPhrase excluded, consistent with paper).

Output:
  runs/threshold_comparison/results_by_mode.csv
  runs/threshold_comparison/summary_by_mode.csv
  runs/threshold_comparison/figures/
"""

import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
PRED_DIR = Path('runs/v5_fixed/predictions')
OUT_DIR  = Path('runs/threshold_comparison')
FIG_DIR  = OUT_DIR / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

TODAY = datetime.now().strftime('%Y-%m-%d')

# ─────────────────────────────────────────────
# Threshold helpers
# ─────────────────────────────────────────────
THRESH_GRID = np.linspace(0.05, 0.95, 181)  # 0.5pp steps


def thresh_fixed(y_score, s0_val, th=0.5):
    return th


def thresh_oracle_f1(y_score, y_true, s0_val):
    """Upper-bound: best F1 threshold searched on the test set itself."""
    best_th, best_f1 = 0.5, -1.0
    for th in THRESH_GRID:
        y_pred = (y_score >= th).astype(int)
        if y_pred.sum() == 0:
            continue
        f = f1_score(y_true, y_pred, zero_division=0)
        if f > best_f1:
            best_f1, best_th = f, th
    return best_th


def thresh_s0_fpr(s0_val, fpr_target=0.02):
    """Find threshold so that FPR on S0 validation ≤ fpr_target."""
    if len(s0_val) == 0:
        return 0.5
    th = float(np.quantile(s0_val, 1.0 - fpr_target))
    return th


# ─────────────────────────────────────────────
# Metric computation
# ─────────────────────────────────────────────

def compute_metrics(y_true, y_score, threshold):
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    try:
        roc  = roc_auc_score(y_true, y_score)
    except Exception:
        roc  = float('nan')
    try:
        pr   = average_precision_score(y_true, y_score)
    except Exception:
        pr   = float('nan')
    fpr  = fp / (fp + tn) if (fp + tn) > 0 else float('nan')
    return dict(acc=acc, prec=prec, rec=rec, f1=f1,
                tn=tn, fp=fp, fn=fn, tp=tp,
                roc_auc=roc, pr_auc=pr, fpr=fpr, threshold=threshold)


# ─────────────────────────────────────────────
# Parse filename
# ─────────────────────────────────────────────
FILE_RE = re.compile(
    r'^(?P<fid>.+?)_(?P<target>S\d)_(?P<fold>fold\d)_(?P<model>.+)\.npz$'
)

FAMILY_MAP = {
    'LiWP_n2_minDf1_SPARSE': 'LiWP',
    'LiWP_n3_minDf1_SPARSE': 'LiWP',
    'SkipGram_compact_dim64_win3': 'SkipGram',
}

ALLOWED_FAMILIES = {'LiWP', 'SkipGram'}


def get_family(fid):
    if fid in FAMILY_MAP:
        return FAMILY_MAP[fid]
    if fid.startswith('LiWP'):
        return 'LiWP'
    if fid.startswith('SkipGram'):
        return 'SkipGram'
    return None


def get_system(model_name):
    if model_name.startswith('ENS_'):
        return 'within_feature'
    return 'single'


# ─────────────────────────────────────────────
# Check if scores are probabilities
# ─────────────────────────────────────────────

def is_prob(y_score):
    return float(y_score.min()) >= -0.01 and float(y_score.max()) <= 1.01


# ─────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────

def main():
    files = sorted(PRED_DIR.glob('*.npz'))
    print(f"Total .npz files: {len(files)}")

    records = []
    skipped = 0

    for fpath in files:
        m = FILE_RE.match(fpath.name)
        if not m:
            skipped += 1
            continue

        fid    = m.group('fid')
        target = m.group('target')
        fold   = m.group('fold')
        model  = m.group('model')

        family = get_family(fid)
        if family not in ALLOWED_FAMILIES:
            continue

        data = np.load(fpath, allow_pickle=True)
        y_true  = data['y_true'].astype(int)
        y_score = data['y_score'].astype(float)
        s0_val  = data['s0_val_scores'].astype(float)

        prob = is_prob(y_score)
        system = get_system(model)

        # ── fixed_05 ──────────────────────────────
        th_fixed = 0.5
        met_fixed = compute_metrics(y_true, y_score, th_fixed)
        met_fixed['mode'] = 'fixed_05'

        # ── oracle_f1 ─────────────────────────────
        if prob:
            th_oracle = thresh_oracle_f1(y_score, y_true, s0_val)
        else:
            # For raw decision values: search threshold over actual score range
            grid = np.linspace(float(y_score.min()), float(y_score.max()), 181)
            best_th, best_f1 = float(np.median(y_score)), -1.0
            for th in grid:
                y_pred = (y_score >= th).astype(int)
                if y_pred.sum() == 0:
                    continue
                f = f1_score(y_true, y_pred, zero_division=0)
                if f > best_f1:
                    best_f1, best_th = f, th
            th_oracle = best_th
        met_oracle = compute_metrics(y_true, y_score, th_oracle)
        met_oracle['mode'] = 'oracle_f1'

        # ── s0_fpr002 ─────────────────────────────
        if len(s0_val) > 0 and prob:
            th_fpr = thresh_s0_fpr(s0_val, fpr_target=0.02)
        elif len(s0_val) > 0:
            th_fpr = float(np.quantile(s0_val, 0.98))
        else:
            th_fpr = 0.5
        met_fpr = compute_metrics(y_true, y_score, th_fpr)
        met_fpr['mode'] = 's0_fpr002'

        base = dict(
            feature_id=fid, family=family, target=target,
            fold=fold, system=system, model=model,
            is_prob=prob
        )

        for met in [met_fixed, met_oracle, met_fpr]:
            records.append({**base, **met})

    df = pd.DataFrame(records)
    print(f"Records: {len(df)}  (skipped {skipped} unmatched filenames)")
    print(f"Families: {df['family'].unique()}")
    print(f"Modes: {df['mode'].unique()}")

    df.to_csv(OUT_DIR / 'results_by_mode.csv', index=False)
    print(f"Saved: {OUT_DIR / 'results_by_mode.csv'}")

    # ─────────────────────────────────────────
    # Summary table
    # ─────────────────────────────────────────
    summary_rows = []
    for mode in ['fixed_05', 'oracle_f1', 's0_fpr002']:
        mdf = df[df['mode'] == mode]
        for system in ['single', 'within_feature']:
            sdf = mdf[mdf['system'] == system]
            if sdf.empty:
                continue
            for family in ['LiWP', 'SkipGram']:
                fdf = sdf[sdf['family'] == family]
                if fdf.empty:
                    continue
                summary_rows.append(dict(
                    mode=mode, system=system, family=family,
                    f1_mean=fdf['f1'].mean(),
                    f1_std=fdf['f1'].std(),
                    prec_mean=fdf['prec'].mean(),
                    rec_mean=fdf['rec'].mean(),
                    acc_mean=fdf['acc'].mean(),
                    roc_auc_mean=fdf['roc_auc'].mean(),
                    pr_auc_mean=fdf['pr_auc'].mean(),
                    fpr_mean=fdf['fpr'].mean(),
                    threshold_mean=fdf['threshold'].mean(),
                    threshold_std=fdf['threshold'].std(),
                    n=len(fdf),
                ))

    sumdf = pd.DataFrame(summary_rows)
    sumdf.to_csv(OUT_DIR / 'summary_by_mode.csv', index=False)
    print(f"Saved: {OUT_DIR / 'summary_by_mode.csv'}")

    # ─────────────────────────────────────────
    # Figures
    # ─────────────────────────────────────────
    make_figures(df, sumdf)
    make_per_target_figure(df)
    print_report(sumdf, df)


# ─────────────────────────────────────────────
# Figure helpers
# ─────────────────────────────────────────────

MODE_LABELS = {
    'fixed_05':   'Fixed (0.5)',
    'oracle_f1':  'Oracle F1',
    's0_fpr002':  'S0-FPR ≤ 2%',
}
MODE_COLORS = {
    'fixed_05':   '#2196F3',
    'oracle_f1':  '#4CAF50',
    's0_fpr002':  '#FF9800',
}
MODE_LS = {
    'fixed_05':   '-',
    'oracle_f1':  '--',
    's0_fpr002':  '-.',
}

TARGETS = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']


def make_figures(df, sumdf):
    """F1 bar comparison across modes for single models (LiWP + SkipGram combined)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'Threshold Mode Comparison — Single Models\n({TODAY})', fontsize=13)

    for ax, family in zip(axes, ['LiWP', 'SkipGram']):
        sdf = df[(df['system'] == 'single') & (df['family'] == family)]
        mode_means = {}
        for mode in ['fixed_05', 'oracle_f1', 's0_fpr002']:
            mdf = sdf[sdf['mode'] == mode]
            means = [mdf[mdf['target'] == t]['f1'].mean() for t in TARGETS]
            mode_means[mode] = means

        x = np.arange(len(TARGETS))
        width = 0.27
        for i, mode in enumerate(['fixed_05', 'oracle_f1', 's0_fpr002']):
            ax.bar(x + (i - 1) * width, mode_means[mode], width,
                   label=MODE_LABELS[mode], color=MODE_COLORS[mode], alpha=0.85)

        ax.set_title(f'{family}', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(TARGETS)
        ax.set_ylabel('F1 Score')
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig_threshold_comparison_single.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIG_DIR / 'fig_threshold_comparison_single.png'}")


def make_per_target_figure(df):
    """Line plot: per-target F1 for each threshold mode, both families combined."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Threshold Mode × Metric Comparison\n({TODAY})', fontsize=13)

    metrics_to_plot = [
        ('f1',    'F1 Score',   axes[0, 0]),
        ('prec',  'Precision',  axes[0, 1]),
        ('rec',   'Recall',     axes[1, 0]),
        ('fpr',   'FPR (S0)',   axes[1, 1]),
    ]

    single_df = df[df['system'] == 'single']

    for metric, ylabel, ax in metrics_to_plot:
        for mode in ['fixed_05', 'oracle_f1', 's0_fpr002']:
            mdf = single_df[single_df['mode'] == mode]
            vals = []
            for t in TARGETS:
                v = mdf[mdf['target'] == t][metric].mean()
                vals.append(v)
            ax.plot(TARGETS, vals,
                    label=MODE_LABELS[mode],
                    color=MODE_COLORS[mode],
                    linestyle=MODE_LS[mode],
                    marker='o', linewidth=2)

        ax.set_title(ylabel, fontsize=11)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig_threshold_per_target.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIG_DIR / 'fig_threshold_per_target.png'}")


def make_ensemble_comparison(df):
    """Compare threshold modes on ensemble (within_feature) predictions."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'Threshold Mode Comparison — Ensemble\n({TODAY})', fontsize=13)

    for ax, family in zip(axes, ['LiWP', 'SkipGram']):
        sdf = df[(df['system'] == 'within_feature') & (df['family'] == family)]
        if sdf.empty:
            continue
        for mode in ['fixed_05', 'oracle_f1', 's0_fpr002']:
            mdf = sdf[sdf['mode'] == mode]
            vals = [mdf[mdf['target'] == t]['f1'].mean() for t in TARGETS]
            ax.plot(TARGETS, vals,
                    label=MODE_LABELS[mode],
                    color=MODE_COLORS[mode],
                    linestyle=MODE_LS[mode],
                    marker='o', linewidth=2)
        ax.set_title(f'{family} — Ensemble', fontsize=11)
        ax.set_ylabel('F1 Score')
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig_threshold_ensemble.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIG_DIR / 'fig_threshold_ensemble.png'}")


def print_report(sumdf, df):
    """Print a readable markdown report."""
    lines = []
    lines.append(f"# Threshold Mode Comparison Report")
    lines.append(f"**Date**: {TODAY}\n")
    lines.append("## Modes")
    lines.append("| Mode | Description |")
    lines.append("|------|-------------|")
    lines.append("| `fixed_05`  | Always use threshold = 0.5 |")
    lines.append("| `oracle_f1` | Threshold maximizing F1 on **test set** (upper bound, not truly leak-free) |")
    lines.append("| `s0_fpr002` | Threshold keeping S0 FPR ≤ 2% via `s0_val_scores` (validation-safe) |")
    lines.append("")

    for system in ['single', 'within_feature']:
        system_label = 'Single Models' if system == 'single' else 'Ensemble (Within-Feature)'
        lines.append(f"## {system_label}")
        lines.append("")
        lines.append("| Mode | Family | F1 mean | F1 std | Precision | Recall | FPR | ROC-AUC | Threshold mean |")
        lines.append("|------|--------|---------|--------|-----------|--------|-----|---------|----------------|")
        sdf = sumdf[sumdf['system'] == system]
        for _, row in sdf.iterrows():
            lines.append(
                f"| {MODE_LABELS[row['mode']]:14} | {row['family']:8} "
                f"| {row['f1_mean']:.4f} | {row['f1_std']:.4f} "
                f"| {row['prec_mean']:.4f} | {row['rec_mean']:.4f} "
                f"| {row['fpr_mean']:.4f} "
                f"| {row['roc_auc_mean']:.4f} "
                f"| {row['threshold_mean']:.3f} ± {row['threshold_std']:.3f} |"
            )
        lines.append("")

    # Per-target delta table (oracle_f1 - fixed_05)
    lines.append("## Per-Target F1 Gain: Oracle F1 vs Fixed 0.5 (Single Models)")
    lines.append("")
    lines.append("| Target | LiWP fixed | LiWP oracle | LiWP Δ | SG fixed | SG oracle | SG Δ |")
    lines.append("|--------|-----------|------------|--------|---------|----------|------|")

    single_df = df[df['system'] == 'single']
    for t in TARGETS:
        row_parts = [f"| {t}"]
        for fam in ['LiWP', 'SkipGram']:
            f_fixed  = single_df[(single_df['target']==t) & (single_df['family']==fam) & (single_df['mode']=='fixed_05')]['f1'].mean()
            f_oracle = single_df[(single_df['target']==t) & (single_df['family']==fam) & (single_df['mode']=='oracle_f1')]['f1'].mean()
            delta = f_oracle - f_fixed
            row_parts.append(f"{f_fixed:.4f}")
            row_parts.append(f"{f_oracle:.4f}")
            sign = '+' if delta >= 0 else ''
            row_parts.append(f"{sign}{delta:.4f}")
        lines.append(" | ".join(row_parts) + " |")

    lines.append("")
    lines.append("## Per-Target F1 Gain: S0-FPR ≤ 2% vs Fixed 0.5 (Single Models)")
    lines.append("")
    lines.append("| Target | LiWP fixed | LiWP fpr | LiWP Δ | SG fixed | SG fpr | SG Δ |")
    lines.append("|--------|-----------|---------|--------|---------|-------|------|")

    for t in TARGETS:
        row_parts = [f"| {t}"]
        for fam in ['LiWP', 'SkipGram']:
            f_fixed = single_df[(single_df['target']==t) & (single_df['family']==fam) & (single_df['mode']=='fixed_05')]['f1'].mean()
            f_fpr   = single_df[(single_df['target']==t) & (single_df['family']==fam) & (single_df['mode']=='s0_fpr002')]['f1'].mean()
            delta = f_fpr - f_fixed
            row_parts.append(f"{f_fixed:.4f}")
            row_parts.append(f"{f_fpr:.4f}")
            sign = '+' if delta >= 0 else ''
            row_parts.append(f"{sign}{delta:.4f}")
        lines.append(" | ".join(row_parts) + " |")

    lines.append("")
    lines.append("---")
    lines.append(f"*Generated: {TODAY}*")

    report_path = OUT_DIR / 'THRESHOLD_COMPARISON_REPORT.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"\nSaved report: {report_path}")

    # Also print to stdout
    print('\n' + '\n'.join(lines[:60]))


if __name__ == '__main__':
    main()
    # Also run ensemble figure
    df_all = pd.read_csv(OUT_DIR / 'results_by_mode.csv')
    make_ensemble_comparison(df_all)
    print("\nDone.")
