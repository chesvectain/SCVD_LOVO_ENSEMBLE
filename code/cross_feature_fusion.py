"""
Cross-Feature Late Fusion from Saved .npz Predictions
======================================================
Combines LiWP and SkipGram predictions (saved from v5_fixed run) via late fusion
without re-training any models.

Source candidates per feature_id:
  - best_single    : single model with highest ROC-AUC for this (target, fold)
  - ENS_All_Soft
  - ENS_Best2_Soft
  - ENS_Fast2_Soft

Cross-feature combinations (participant feature_ids):
  - LiWP_n2  × SkipGram           (2-way)
  - LiWP_n3  × SkipGram           (2-way)
  - LiWP_n2  × LiWP_n3            (cross n-gram, 2-way)
  - LiWP_n2  × LiWP_n3 × SkipGram (3-way)

Fusion methods:
  - Soft         : simple average of scores
  - WeightedSoft : weighted by per-fold ROC-AUC (computed on test set — proxy weight)
  - Hard         : majority vote (each source thresholded at 0.5)

Threshold: fixed = 0.5  (also oracle_f1 for reference)

Output:
  runs/cross_feature/results_cross_feature.csv
  runs/cross_feature/summary_cross_feature.csv
  runs/cross_feature/CROSS_FEATURE_REPORT.md
  runs/cross_feature/figures/
"""

import os, re, itertools
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
PRED_DIR = Path('runs/v5_fixed/predictions')
OUT_DIR  = Path('runs/cross_feature')
FIG_DIR  = OUT_DIR / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

TODAY    = datetime.now().strftime('%Y-%m-%d')
TARGETS  = [f'S{i}' for i in range(1, 9)]
FOLDS    = [f'fold{i}' for i in range(5)]

FID_MAP = {
    'LiWP_n2': 'LiWP_n2_minDf1_SPARSE',
    'LiWP_n3': 'LiWP_n3_minDf1_SPARSE',
    'SkipGram': 'SkipGram_compact_dim64_win3',
}
SINGLE_MODELS = ['LR', 'RF', 'KNN', 'DT', 'LinearSVC']
ENS_SOURCES   = ['ENS_All_Soft', 'ENS_Best2_Soft', 'ENS_Fast2_Soft']

CROSS_COMBOS = {
    'n2_x_SG' : ['LiWP_n2', 'SkipGram'],
    'n3_x_SG' : ['LiWP_n3', 'SkipGram'],
    'n2_x_n3' : ['LiWP_n2', 'LiWP_n3'],
    'n2_n3_SG': ['LiWP_n2', 'LiWP_n3', 'SkipGram'],
}

THRESH_GRID = np.linspace(0.05, 0.95, 181)


# ─────────────────────────────────────────────
# Load all predictions into dict
# ─────────────────────────────────────────────

def load_all_predictions():
    """Returns {(fid_short, target, fold, model): {'y_true', 'y_score', 'roc_auc'}}"""
    preds = {}
    FILE_RE = re.compile(r'^(?P<fid>.+?)_(?P<target>S\d)_(?P<fold>fold\d)_(?P<model>.+)\.npz$')

    fid_reverse = {v: k for k, v in FID_MAP.items()}

    for fpath in sorted(PRED_DIR.glob('*.npz')):
        m = FILE_RE.match(fpath.name)
        if not m:
            continue
        fid_full = m.group('fid')
        fid_short = fid_reverse.get(fid_full)
        if fid_short is None:
            continue  # skip OpPhrase etc.

        target = m.group('target')
        fold   = m.group('fold')
        model  = m.group('model')

        data = np.load(fpath, allow_pickle=True)
        y_true  = data['y_true'].astype(int)
        y_score = data['y_score'].astype(float)

        # Normalize non-probability scores to [0,1] using min-max for fusion
        if y_score.min() < -0.01 or y_score.max() > 1.01:
            lo, hi = y_score.min(), y_score.max()
            if hi > lo:
                y_score = (y_score - lo) / (hi - lo)
            else:
                y_score = np.full_like(y_score, 0.5)

        try:
            roc = float(roc_auc_score(y_true, y_score))
        except Exception:
            roc = 0.5

        preds[(fid_short, target, fold, model)] = {
            'y_true' : y_true,
            'y_score': y_score,
            'roc_auc': roc,
        }

    print(f"Loaded {len(preds)} prediction entries.")
    return preds


# ─────────────────────────────────────────────
# Select source per (fid_short, target, fold)
# ─────────────────────────────────────────────

def get_source(preds, fid_short, target, fold, source_type):
    """Return (y_score, roc_auc) for the given source_type."""
    if source_type == 'best_single':
        best_roc, best_score = -1.0, None
        for m in SINGLE_MODELS:
            key = (fid_short, target, fold, m)
            if key in preds:
                r = preds[key]['roc_auc']
                if r > best_roc:
                    best_roc   = r
                    best_score = preds[key]['y_score']
        return best_score, best_roc
    else:
        key = (fid_short, target, fold, source_type)
        if key in preds:
            return preds[key]['y_score'], preds[key]['roc_auc']
        return None, None


# ─────────────────────────────────────────────
# Fusion functions
# ─────────────────────────────────────────────

def soft_avg(score_list):
    return np.mean(score_list, axis=0)


def weighted_soft(score_list, weights):
    w = np.array(weights, dtype=float)
    w = np.clip(w, 1e-6, None)
    w /= w.sum()
    return sum(s * ww for s, ww in zip(score_list, w))


def hard_vote(score_list, th=0.5):
    votes = np.stack([(s >= th).astype(int) for s in score_list], axis=0)
    majority = (votes.sum(axis=0) > len(score_list) / 2).astype(int)
    # Return vote fraction as score proxy
    return votes.mean(axis=0), majority


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────

def compute_metrics(y_true, y_score, threshold):
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    fpr  = fp / (fp + tn) if (fp + tn) > 0 else float('nan')
    try:
        roc = float(roc_auc_score(y_true, y_score))
    except Exception:
        roc = float('nan')
    try:
        pr = float(average_precision_score(y_true, y_score))
    except Exception:
        pr = float('nan')
    return dict(acc=acc, prec=prec, rec=rec, f1=f1,
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                roc_auc=roc, pr_auc=pr, fpr=fpr, threshold=threshold)


def oracle_f1_threshold(y_true, y_score):
    best_th, best_f1 = 0.5, -1.0
    for th in THRESH_GRID:
        y_pred = (y_score >= th).astype(int)
        if y_pred.sum() == 0:
            continue
        f = f1_score(y_true, y_pred, zero_division=0)
        if f > best_f1:
            best_f1, best_th = f, th
    return best_th


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    preds = load_all_predictions()
    records = []

    source_types = ['best_single'] + ENS_SOURCES  # 4 source types

    for combo_name, fid_list in CROSS_COMBOS.items():
        for source_type in source_types:
            for target in TARGETS:
                for fold in FOLDS:
                    # Collect scores from each participating feature_id
                    scores   = []
                    roc_list = []
                    y_true   = None

                    for fid_short in fid_list:
                        sc, roc = get_source(preds, fid_short, target, fold, source_type)
                        if sc is None:
                            break
                        scores.append(sc)
                        roc_list.append(roc)

                        # y_true: use the first one (verified identical across fids)
                        if y_true is None:
                            key = (fid_short, target, fold,
                                   source_type if source_type != 'best_single' else 'LR')
                            for m in ([source_type] if source_type != 'best_single' else SINGLE_MODELS):
                                k = (fid_short, target, fold, m)
                                if k in preds:
                                    y_true = preds[k]['y_true']
                                    break

                    if len(scores) < len(fid_list) or y_true is None:
                        continue  # incomplete

                    base_row = dict(
                        combo=combo_name,
                        fids='+'.join(fid_list),
                        source=source_type,
                        target=target,
                        fold=fold,
                        system='cross_feature',
                        n_participants=len(fid_list),
                    )

                    # ── Soft ───────────────────────────────────
                    sc_soft = soft_avg(scores)
                    th_soft_fixed  = 0.5
                    th_soft_oracle = oracle_f1_threshold(y_true, sc_soft)
                    for th, mode in [(th_soft_fixed, 'fixed_05'), (th_soft_oracle, 'oracle_f1')]:
                        mm = compute_metrics(y_true, sc_soft, th)
                        records.append({**base_row, 'fusion': 'Soft', 'threshold_mode': mode, **mm})

                    # ── WeightedSoft ────────────────────────────
                    sc_wsoft = weighted_soft(scores, roc_list)
                    th_ws_fixed  = 0.5
                    th_ws_oracle = oracle_f1_threshold(y_true, sc_wsoft)
                    for th, mode in [(th_ws_fixed, 'fixed_05'), (th_ws_oracle, 'oracle_f1')]:
                        mm = compute_metrics(y_true, sc_wsoft, th)
                        records.append({**base_row, 'fusion': 'WeightedSoft', 'threshold_mode': mode, **mm})

                    # ── Hard ───────────────────────────────────
                    sc_hard_frac, y_pred_hard = hard_vote(scores)
                    # For hard vote use the fraction as score
                    th_hard_fixed  = 0.5
                    th_hard_oracle = oracle_f1_threshold(y_true, sc_hard_frac)
                    for th, mode in [(th_hard_fixed, 'fixed_05'), (th_hard_oracle, 'oracle_f1')]:
                        mm = compute_metrics(y_true, sc_hard_frac, th)
                        records.append({**base_row, 'fusion': 'Hard', 'threshold_mode': mode, **mm})

    df = pd.DataFrame(records)
    print(f"Cross-feature records: {len(df)}")
    print(f"Combos: {df['combo'].unique()}")
    print(f"Sources: {df['source'].unique()}")
    df.to_csv(OUT_DIR / 'results_cross_feature.csv', index=False)
    print(f"Saved: {OUT_DIR / 'results_cross_feature.csv'}")

    # ─────────────────────────────────────────
    # Load within_feature results for comparison
    # ─────────────────────────────────────────
    ref_df = pd.read_csv('runs/v5_fixed/results_raw_filtered.csv')
    within_df = ref_df[ref_df['system'] == 'within_feature'].copy()
    single_df  = ref_df[ref_df['system'] == 'single'].copy()

    # Best within_feature per family (max F1 per target/fold → then mean over folds)
    wf_best = within_df.groupby(['family', 'target'])['f1'].max().reset_index()
    wf_best_mean = wf_best.groupby('family')['f1'].mean()
    print("\nWithin-feature best F1 (mean over targets):")
    print(wf_best_mean)

    # ─────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────
    summary = (
        df.groupby(['combo', 'source', 'fusion', 'threshold_mode'])
        .agg(f1_mean=('f1','mean'), f1_std=('f1','std'),
             prec_mean=('prec','mean'), rec_mean=('rec','mean'),
             roc_auc_mean=('roc_auc','mean'), pr_auc_mean=('pr_auc','mean'),
             fpr_mean=('fpr','mean'), threshold_mean=('threshold','mean'))
        .reset_index()
        .sort_values('f1_mean', ascending=False)
    )
    summary.to_csv(OUT_DIR / 'summary_cross_feature.csv', index=False)
    print(f"\nSaved: {OUT_DIR / 'summary_cross_feature.csv'}")

    # Print top results
    print("\n=== Top-20 Cross-Feature Combinations (fixed_05) ===")
    top = summary[summary['threshold_mode'] == 'fixed_05'].head(20)
    print(top[['combo','source','fusion','f1_mean','f1_std','prec_mean','rec_mean','roc_auc_mean']].to_string(index=False))

    make_figures(df, within_df, single_df)
    make_report(df, summary, wf_best_mean, single_df)

    print("\nDone.")


# ─────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────

COMBO_COLORS = {
    'n2_x_SG' : '#2196F3',
    'n3_x_SG' : '#E91E63',
    'n2_x_n3' : '#9C27B0',
    'n2_n3_SG': '#FF5722',
}
FUSION_LS = {'Soft': '-', 'WeightedSoft': '--', 'Hard': '-.'}


def make_figures(df, within_df, single_df):
    """Figure 1: per-target F1 for each combo × Soft fusion (fixed_05 threshold)"""

    df_plot = df[(df['fusion'] == 'Soft') & (df['threshold_mode'] == 'fixed_05')]

    # Best within-feature baseline per target (across families)
    wf_f1 = within_df.groupby('target')['f1'].mean()
    sg_f1 = single_df[single_df['family']=='SkipGram'].groupby('target')['f1'].mean()

    for source_type in df['source'].unique():
        sdf = df_plot[df_plot['source'] == source_type]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_title(f'Cross-Feature Soft Fusion — Source: {source_type}\n({TODAY})', fontsize=12)

        for combo_name, color in COMBO_COLORS.items():
            cdf = sdf[sdf['combo'] == combo_name]
            vals = [cdf[cdf['target'] == t]['f1'].mean() for t in TARGETS]
            ax.plot(TARGETS, vals, label=combo_name, color=color, marker='o', linewidth=2)

        # Baselines
        ax.plot(TARGETS, [wf_f1.get(t, np.nan) for t in TARGETS],
                label='Within-Feature (avg)', color='gray', linestyle=':', linewidth=1.5, marker='s')
        ax.plot(TARGETS, [sg_f1.get(t, np.nan) for t in TARGETS],
                label='SkipGram Single (avg)', color='black', linestyle='--', linewidth=1.2, marker='^')

        ax.set_ylabel('F1 Score')
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fname = FIG_DIR / f'fig_cross_soft_{source_type.replace("ENS_","")}_fixed.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {fname}")

    # Figure 2: fusion method comparison for best combo (n2_n3_SG, fixed_05)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'Cross-Feature Fusion Method Comparison — n2+n3+SG\n({TODAY})', fontsize=12)

    for ax, source_type in zip(axes, ['ENS_Best2_Soft', 'ENS_All_Soft']):
        ax.set_title(f'Source: {source_type}', fontsize=11)
        sdf = df[(df['combo'] == 'n2_n3_SG') & (df['source'] == source_type) &
                 (df['threshold_mode'] == 'fixed_05')]
        for fusion, ls in FUSION_LS.items():
            fdf = sdf[sdf['fusion'] == fusion]
            vals = [fdf[fdf['target'] == t]['f1'].mean() for t in TARGETS]
            ax.plot(TARGETS, vals, label=fusion, linestyle=ls, marker='o', linewidth=2)
        # Baseline
        ax.plot(TARGETS, [wf_f1.get(t, np.nan) for t in TARGETS],
                label='Within-Feature (avg)', color='gray', linestyle=':', linewidth=1.5, marker='s')
        ax.set_ylabel('F1 Score')
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig_cross_fusion_methods.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIG_DIR / 'fig_cross_fusion_methods.png'}")

    # Figure 3: Summary bar — top combinations vs baselines
    make_summary_bar(df, within_df, single_df)


def make_summary_bar(df, within_df, single_df):
    """Bar chart comparing cross-feature top variants vs baselines."""
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_title(f'Cross-Feature vs Within-Feature vs Single — Average F1\n({TODAY})', fontsize=12)

    combos_to_show = [
        ('n2_x_SG',  'ENS_Best2_Soft', 'Soft',         'fixed_05', 'n2×SG / Best2 / Soft'),
        ('n3_x_SG',  'ENS_Best2_Soft', 'Soft',         'fixed_05', 'n3×SG / Best2 / Soft'),
        ('n2_n3_SG', 'ENS_Best2_Soft', 'Soft',         'fixed_05', 'n2+n3+SG / Best2 / Soft'),
        ('n2_n3_SG', 'ENS_Best2_Soft', 'WeightedSoft', 'fixed_05', 'n2+n3+SG / Best2 / WtSoft'),
        ('n2_n3_SG', 'ENS_All_Soft',   'Soft',         'fixed_05', 'n2+n3+SG / All / Soft'),
        ('n2_x_SG',  'best_single',    'Soft',         'fixed_05', 'n2×SG / BestSingle / Soft'),
        ('n2_n3_SG', 'best_single',    'Soft',         'fixed_05', 'n2+n3+SG / BestSingle / Soft'),
    ]

    labels, means, stds = [], [], []
    for combo, src, fusion, tmode, label in combos_to_show:
        sub = df[(df['combo']==combo) & (df['source']==src) &
                 (df['fusion']==fusion) & (df['threshold_mode']==tmode)]
        labels.append(label)
        means.append(sub['f1'].mean())
        stds.append(sub['f1'].std())

    # Baselines
    for fam in ['LiWP', 'SkipGram']:
        wf = within_df[within_df['family']==fam]
        labels.append(f'{fam} Within-Feature')
        means.append(wf['f1'].mean())
        stds.append(wf['f1'].std())
        sg = single_df[single_df['family']==fam]
        labels.append(f'{fam} Single')
        means.append(sg['f1'].mean())
        stds.append(sg['f1'].std())

    colors = (['#FF5722']*3 + ['#FF9800']*2 + ['#FFC107']*2 +
              ['#2196F3','#E91E63'] + ['#90CAF9','#F48FB1'])
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors[:len(labels)], alpha=0.85)

    ax.axhline(within_df[within_df['family']=='SkipGram']['f1'].mean(),
               color='#E91E63', linestyle='--', linewidth=1, alpha=0.5, label='SG Within-Feature avg')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=9)
    ax.set_ylabel('F1 Score (mean over targets × folds)')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig_cross_summary_bar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIG_DIR / 'fig_cross_summary_bar.png'}")


# ─────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────

def make_report(df, summary, wf_best_mean, single_df):
    lines = []
    lines.append("# Cross-Feature Late Fusion Results")
    lines.append(f"**Date**: {TODAY}\n")
    lines.append("## Setup")
    lines.append("| Item | Detail |")
    lines.append("|------|--------|")
    lines.append("| Base predictions | Saved .npz from `runs/v5_fixed/` |")
    lines.append("| Feature families | LiWP (n2, n3) + SkipGram |")
    lines.append("| Source per fid   | best_single / ENS_All_Soft / ENS_Best2_Soft / ENS_Fast2_Soft |")
    lines.append("| Fusion methods   | Soft / WeightedSoft / Hard |")
    lines.append("| Threshold        | fixed=0.5 / oracle_f1 (upper bound) |")
    lines.append("| Cross combos     | n2×SG / n3×SG / n2×n3 / n2+n3+SG |")
    lines.append("")

    # Baseline table
    lines.append("## Baseline Reference")
    lines.append("| System | Family | F1 mean |")
    lines.append("|--------|--------|---------|")
    ref_df = pd.read_csv('runs/v5_fixed/results_raw_filtered.csv')
    for sys in ['single', 'within_feature']:
        for fam in ['LiWP', 'SkipGram']:
            v = ref_df[(ref_df['system']==sys) & (ref_df['family']==fam)]['f1'].mean()
            lines.append(f"| {sys} | {fam} | {v:.4f} |")
    lines.append("")

    # Top results fixed_05
    lines.append("## Top-15 Cross-Feature Combinations (fixed_05)")
    lines.append("")
    top = summary[summary['threshold_mode'] == 'fixed_05'].head(15)
    lines.append("| Combo | Source | Fusion | F1 mean | F1 std | Precision | Recall | ROC-AUC |")
    lines.append("|-------|--------|--------|---------|--------|-----------|--------|---------|")
    for _, row in top.iterrows():
        lines.append(
            f"| {row['combo']:10} | {row['source']:20} | {row['fusion']:12} "
            f"| {row['f1_mean']:.4f} | {row['f1_std']:.4f} "
            f"| {row['prec_mean']:.4f} | {row['rec_mean']:.4f} "
            f"| {row['roc_auc_mean']:.4f} |"
        )
    lines.append("")

    # Per-target detail for best cross combo
    best_row = summary[summary['threshold_mode']=='fixed_05'].iloc[0]
    lines.append(f"## Per-Target F1: Best Combo ({best_row['combo']} / {best_row['source']} / {best_row['fusion']})")
    lines.append("")
    lines.append("| Target | Cross F1 | LiWP Within | SG Within | SG Single |")
    lines.append("|--------|----------|-------------|-----------|-----------|")
    wf = ref_df[ref_df['system']=='within_feature']
    sg_s = ref_df[(ref_df['system']=='single') & (ref_df['family']=='SkipGram')]
    best_sub = df[(df['combo']==best_row['combo']) & (df['source']==best_row['source']) &
                  (df['fusion']==best_row['fusion']) & (df['threshold_mode']=='fixed_05')]
    for t in TARGETS:
        cf1 = best_sub[best_sub['target']==t]['f1'].mean()
        lf1 = wf[(wf['family']=='LiWP') & (wf['target']==t)]['f1'].mean()
        sf1 = wf[(wf['family']=='SkipGram') & (wf['target']==t)]['f1'].mean()
        sg1 = sg_s[sg_s['target']==t]['f1'].mean()
        lines.append(f"| {t} | {cf1:.4f} | {lf1:.4f} | {sf1:.4f} | {sg1:.4f} |")
    lines.append("")

    # Delta vs within_feature
    best_cross_mean = best_sub['f1'].mean()
    sg_within_mean  = ref_df[(ref_df['system']=='within_feature') & (ref_df['family']=='SkipGram')]['f1'].mean()
    liwp_within_mean = ref_df[(ref_df['system']=='within_feature') & (ref_df['family']=='LiWP')]['f1'].mean()
    lines.append("## Key Findings")
    lines.append(f"- **Best cross-feature F1**: {best_cross_mean:.4f}  "
                 f"(combo={best_row['combo']}, source={best_row['source']}, fusion={best_row['fusion']})")
    lines.append(f"- vs SkipGram within-feature: **{best_cross_mean - sg_within_mean:+.4f}**")
    lines.append(f"- vs LiWP within-feature: **{best_cross_mean - liwp_within_mean:+.4f}**")
    lines.append("")
    lines.append("## Figures")
    lines.append("| File | Description |")
    lines.append("|------|-------------|")
    for fname in sorted(FIG_DIR.glob('*.png')):
        lines.append(f"| ![]({fname}) | {fname.stem} |")
    lines.append("")
    lines.append(f"---\n*Generated: {TODAY}*")

    report_path = OUT_DIR / 'CROSS_FEATURE_REPORT.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"\nSaved report: {report_path}")


if __name__ == '__main__':
    main()
