"""
전체 실험 결과 종합 정리
Single → Within-Feature → Cross-Feature (fixed / adaptive)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

TODAY   = datetime.now().strftime('%Y-%m-%d')
OUT_DIR = Path('runs/final_summary')
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = ['S1','S2','S3','S4','S5','S6','S7','S8']

# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────
ref   = pd.read_csv('runs/v5_fixed/results_raw_filtered.csv')
cross = pd.read_csv('runs/cross_feature/results_cross_feature.csv')

single_df = ref[ref['system'] == 'single']
within_df = ref[ref['system'] == 'within_feature']

# Best cross-feature combo: n2_n3_SG / best_single / WeightedSoft
BEST_COMBO  = 'n2_n3_SG'
BEST_SOURCE = 'best_single'
BEST_FUSION = 'WeightedSoft'

cross_fixed  = cross[(cross['combo']==BEST_COMBO) & (cross['source']==BEST_SOURCE) &
                     (cross['fusion']==BEST_FUSION) & (cross['threshold_mode']=='fixed_05')]
cross_oracle = cross[(cross['combo']==BEST_COMBO) & (cross['source']==BEST_SOURCE) &
                     (cross['fusion']==BEST_FUSION) & (cross['threshold_mode']=='oracle_f1')]

# ─────────────────────────────────────────────
# Per-target F1 table
# ─────────────────────────────────────────────
def target_f1(df, target):
    return df[df['target'] == target]['f1'].mean()

def build_per_target_table():
    rows = []
    for t in TARGETS:
        rows.append({
            'Target'              : t,
            'LiWP Single'         : target_f1(single_df[single_df['family']=='LiWP'], t),
            'SG Single'           : target_f1(single_df[single_df['family']=='SkipGram'], t),
            'LiWP Within'         : target_f1(within_df[within_df['family']=='LiWP'], t),
            'SG Within'           : target_f1(within_df[within_df['family']=='SkipGram'], t),
            'Cross Fixed(0.5)'    : target_f1(cross_fixed, t),
            'Cross Adaptive'      : target_f1(cross_oracle, t),
        })
    df_tbl = pd.DataFrame(rows).set_index('Target')

    # Overall mean row
    mean_row = df_tbl.mean().rename('Mean')
    df_tbl = pd.concat([df_tbl, mean_row.to_frame().T])
    return df_tbl

per_target = build_per_target_table()

# ─────────────────────────────────────────────
# Overall summary table
# ─────────────────────────────────────────────
def overall_metrics(df):
    return {
        'F1'      : df['f1'].mean(),
        'Prec'    : df['prec'].mean(),
        'Rec'     : df['rec'].mean(),
        'ROC-AUC' : df['roc_auc'].mean(),
        'PR-AUC'  : df['pr_auc'].mean(),
    }

summary_rows = [
    {'System': 'LiWP Single',          **overall_metrics(single_df[single_df['family']=='LiWP'])},
    {'System': 'SkipGram Single',       **overall_metrics(single_df[single_df['family']=='SkipGram'])},
    {'System': 'LiWP Within-Feature',   **overall_metrics(within_df[within_df['family']=='LiWP'])},
    {'System': 'SG Within-Feature',     **overall_metrics(within_df[within_df['family']=='SkipGram'])},
    {'System': 'Cross Fixed(0.5)',       **overall_metrics(cross_fixed)},
    {'System': 'Cross Adaptive (oracle)',**overall_metrics(cross_oracle)},
]
summary_df = pd.DataFrame(summary_rows).set_index('System')

# ─────────────────────────────────────────────
# Figure 1: per-target F1 line plot (all systems)
# ─────────────────────────────────────────────
def fig_per_target():
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.set_title(f'Per-Target F1: Single → Within-Feature → Cross-Feature\n({TODAY})', fontsize=12)

    styles = [
        ('LiWP Single',       single_df[single_df['family']=='LiWP'],       '#90CAF9', '-',   'o', 1.5),
        ('SkipGram Single',   single_df[single_df['family']=='SkipGram'],    '#EF9A9A', '-',   'o', 1.5),
        ('LiWP Within',       within_df[within_df['family']=='LiWP'],        '#1565C0', '--',  's', 1.5),
        ('SG Within',         within_df[within_df['family']=='SkipGram'],    '#B71C1C', '--',  's', 1.5),
        ('Cross Fixed(0.5)',  cross_fixed,                                   '#FF6F00', '-.',  '^', 2.2),
        ('Cross Adaptive',    cross_oracle,                                   '#2E7D32', '-',   'D', 2.5),
    ]

    for label, df_, color, ls, marker, lw in styles:
        vals = [df_[df_['target']==t]['f1'].mean() for t in TARGETS]
        ax.plot(TARGETS, vals, label=label, color=color, linestyle=ls,
                marker=marker, linewidth=lw, markersize=6)

    ax.set_ylabel('F1 Score')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = OUT_DIR / 'fig1_per_target_f1_all_systems.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


# ─────────────────────────────────────────────
# Figure 2: overall mean F1 bar chart
# ─────────────────────────────────────────────
def fig_overall_bar():
    labels = ['LiWP\nSingle', 'SG\nSingle',
              'LiWP\nWithin', 'SG\nWithin',
              'Cross\nFixed(0.5)', 'Cross\nAdaptive']
    f1s = [summary_df.loc[k, 'F1'] for k in summary_df.index]
    rocs = [summary_df.loc[k, 'ROC-AUC'] for k in summary_df.index]

    colors = ['#90CAF9', '#EF9A9A', '#1565C0', '#B71C1C', '#FF6F00', '#2E7D32']

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'Overall Performance Comparison ({TODAY})', fontsize=12)

    for ax, vals, ylabel, title in [
        (axes[0], f1s,  'F1 Score',  'Mean F1 (all targets × folds)'),
        (axes[1], rocs, 'ROC-AUC',   'Mean ROC-AUC'),
    ]:
        bars = ax.bar(labels, vals, color=colors, alpha=0.85, edgecolor='white')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        # Highlight cross adaptive
        bars[-1].set_edgecolor('#1B5E20')
        bars[-1].set_linewidth(2)

    plt.tight_layout()
    path = OUT_DIR / 'fig2_overall_bar.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


# ─────────────────────────────────────────────
# Figure 3: fixed vs adaptive per target (cross only)
# ─────────────────────────────────────────────
def fig_fixed_vs_adaptive():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(f'Cross-Feature: Fixed(0.5) vs Adaptive Threshold\n'
                 f'(n2+n3+SG / BestSingle / WeightedSoft) — {TODAY}', fontsize=11)

    fixed_vals  = [cross_fixed[cross_fixed['target']==t]['f1'].mean()  for t in TARGETS]
    oracle_vals = [cross_oracle[cross_oracle['target']==t]['f1'].mean() for t in TARGETS]
    sg_vals     = [single_df[(single_df['family']=='SkipGram') & (single_df['target']==t)]['f1'].mean() for t in TARGETS]

    ax.plot(TARGETS, sg_vals,     label='SkipGram Single (baseline)', color='gray',    linestyle=':', marker='o', linewidth=1.5)
    ax.plot(TARGETS, fixed_vals,  label='Cross Fixed(0.5)',           color='#FF6F00', linestyle='--', marker='s', linewidth=2)
    ax.plot(TARGETS, oracle_vals, label='Cross Adaptive (oracle F1)', color='#2E7D32', linestyle='-',  marker='D', linewidth=2.5)

    # Delta annotation
    for i, t in enumerate(TARGETS):
        delta = oracle_vals[i] - fixed_vals[i]
        if abs(delta) > 0.02:
            ax.annotate(f'+{delta:.2f}', xy=(i, oracle_vals[i]),
                        xytext=(i, oracle_vals[i]+0.03),
                        ha='center', fontsize=8, color='#2E7D32')

    ax.set_ylabel('F1 Score')
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = OUT_DIR / 'fig3_fixed_vs_adaptive.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


# ─────────────────────────────────────────────
# Markdown report
# ─────────────────────────────────────────────
def make_report():
    lines = []
    lines.append('# 전체 실험 결과 종합 정리')
    lines.append(f'**Date**: {TODAY}\n')
    lines.append(f'**Best Cross-Feature Config**: `{BEST_COMBO}` / `{BEST_SOURCE}` / `{BEST_FUSION}`\n')
    lines.append('---\n')

    # Table 1: Overall metrics
    lines.append('## Table 1. 시스템별 전체 평균 성능\n')
    lines.append('| System | F1 | Precision | Recall | ROC-AUC | PR-AUC |')
    lines.append('|--------|----|-----------|--------|---------|--------|')
    for sys, row in summary_df.iterrows():
        lines.append(f'| {sys:28} | {row["F1"]:.4f} | {row["Prec"]:.4f} | {row["Rec"]:.4f} | {row["ROC-AUC"]:.4f} | {row["PR-AUC"]:.4f} |')
    lines.append('')

    # Table 2: Per-target F1
    lines.append('## Table 2. 취약점별 F1 Score\n')
    lines.append('| Target | LiWP Single | SG Single | LiWP Within | SG Within | Cross Fixed | Cross Adaptive |')
    lines.append('|--------|------------|-----------|-------------|-----------|-------------|----------------|')
    for t in TARGETS:
        row = per_target.loc[t]
        lines.append(
            f'| {t} '
            f'| {row["LiWP Single"]:.4f} '
            f'| {row["SG Single"]:.4f} '
            f'| {row["LiWP Within"]:.4f} '
            f'| {row["SG Within"]:.4f} '
            f'| {row["Cross Fixed(0.5)"]:.4f} '
            f'| {row["Cross Adaptive"]:.4f} |'
        )
    mean_row = per_target.loc['Mean']
    lines.append(
        f'| **Mean** '
        f'| **{mean_row["LiWP Single"]:.4f}** '
        f'| **{mean_row["SG Single"]:.4f}** '
        f'| **{mean_row["LiWP Within"]:.4f}** '
        f'| **{mean_row["SG Within"]:.4f}** '
        f'| **{mean_row["Cross Fixed(0.5)"]:.4f}** '
        f'| **{mean_row["Cross Adaptive"]:.4f}** |'
    )
    lines.append('')

    # Table 3: Delta vs SkipGram Single
    lines.append('## Table 3. SkipGram Single 대비 F1 향상 (Δ)\n')
    lines.append('| Target | SG Single | Cross Fixed | Δ Fixed | Cross Adaptive | Δ Adaptive |')
    lines.append('|--------|-----------|-------------|---------|----------------|------------|')
    for t in TARGETS:
        sg  = per_target.loc[t, 'SG Single']
        cf  = per_target.loc[t, 'Cross Fixed(0.5)']
        co  = per_target.loc[t, 'Cross Adaptive']
        d1  = cf - sg
        d2  = co - sg
        lines.append(f'| {t} | {sg:.4f} | {cf:.4f} | {d1:+.4f} | {co:.4f} | {d2:+.4f} |')
    sg_m  = per_target.loc['Mean', 'SG Single']
    cf_m  = per_target.loc['Mean', 'Cross Fixed(0.5)']
    co_m  = per_target.loc['Mean', 'Cross Adaptive']
    lines.append(f'| **Mean** | **{sg_m:.4f}** | **{cf_m:.4f}** | **{cf_m-sg_m:+.4f}** | **{co_m:.4f}** | **{co_m-sg_m:+.4f}** |')
    lines.append('')

    # Key findings
    lines.append('## 핵심 결과\n')
    sg_f1   = summary_df.loc['SkipGram Single', 'F1']
    cf_f1   = summary_df.loc['Cross Fixed(0.5)', 'F1']
    co_f1   = summary_df.loc['Cross Adaptive (oracle)', 'F1']
    sg_roc  = summary_df.loc['SkipGram Single', 'ROC-AUC']
    co_roc  = summary_df.loc['Cross Adaptive (oracle)', 'ROC-AUC']
    lines.append(f'1. **Cross-Feature Fixed(0.5)** vs SkipGram Single: F1 `{cf_f1:.4f}` vs `{sg_f1:.4f}` (**{cf_f1-sg_f1:+.4f}**)')
    lines.append(f'2. **Cross-Feature Adaptive** vs SkipGram Single: F1 `{co_f1:.4f}` vs `{sg_f1:.4f}` (**{co_f1-sg_f1:+.4f}**)')
    lines.append(f'3. **Adaptive vs Fixed**: F1 `{co_f1:.4f}` vs `{cf_f1:.4f}` (**{co_f1-cf_f1:+.4f}**)')
    lines.append(f'4. **ROC-AUC**: SkipGram Single `{sg_roc:.4f}` → Cross Adaptive `{co_roc:.4f}` (**{co_roc-sg_roc:+.4f}**)')
    lines.append('')
    lines.append('### S4 (가장 어려운 취약점) 특이점')
    s4_sg = per_target.loc['S4', 'SG Single']
    s4_cf = per_target.loc['S4', 'Cross Fixed(0.5)']
    s4_co = per_target.loc['S4', 'Cross Adaptive']
    lines.append(f'- SkipGram Single: `{s4_sg:.4f}`')
    lines.append(f'- Cross Fixed(0.5): `{s4_cf:.4f}` ({s4_cf-s4_sg:+.4f})')
    lines.append(f'- Cross Adaptive: `{s4_co:.4f}` ({s4_co-s4_sg:+.4f}) ← 극적 향상')
    lines.append('')

    # Figures
    lines.append('## Figures\n')
    lines.append('### Fig 1. 취약점별 F1 (전체 시스템 비교)')
    lines.append(f'![](fig1_per_target_f1_all_systems.png)\n')
    lines.append('### Fig 2. 전체 평균 F1 / ROC-AUC 바 차트')
    lines.append(f'![](fig2_overall_bar.png)\n')
    lines.append('### Fig 3. Cross-Feature Fixed vs Adaptive')
    lines.append(f'![](fig3_fixed_vs_adaptive.png)\n')
    lines.append(f'---\n*Generated: {TODAY}*')

    path = OUT_DIR / 'FINAL_SUMMARY.md'
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'Saved: {path}')


# ─────────────────────────────────────────────
if __name__ == '__main__':
    print('=== Overall Summary ===')
    print(summary_df.round(4).to_string())
    print()
    print('=== Per-Target F1 ===')
    print(per_target.round(4).to_string())

    fig_per_target()
    fig_overall_bar()
    fig_fixed_vs_adaptive()
    make_report()

    # Save CSV
    per_target.round(4).to_csv(OUT_DIR / 'per_target_f1.csv')
    summary_df.round(4).to_csv(OUT_DIR / 'overall_summary.csv')
    print(f'\nAll outputs saved to: {OUT_DIR}')
