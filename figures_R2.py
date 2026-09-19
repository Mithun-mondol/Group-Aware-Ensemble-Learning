"""Regenerate the manuscript figures from the CORRECTED results.

Why this exists. Every figure in the R1 manuscript was produced by
concrete_strength_pipeline.py, which fits LightGBM and CatBoost with the outer
validation fold as `eval_set` (disclosure A.1). The figures therefore display
the leaked numbers, and after the table corrections they contradict their own
tables -- Figure 2's inset box read R2=0.9251 above a table reading 0.9237.

Everything here is drawn from the stored out-of-fold predictions of the
corrected run, so no model is refitted and the figures agree with the tables by
construction. Label sizes are raised throughout, which is Reviewer 6 item 6.

    python figures_R2.py            # every figure whose inputs exist
    python figures_R2.py --only 3   # one figure
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import uniform_filter1d

from dataset import XLSX, load_raw

DATA = os.environ.get('R2_DATA', 'rerun_R2_out')
OUT = os.environ.get('R2_FIGS', 'figures_R2')
DPI = 600

# The pipeline's palette, unchanged.
C_STD, C_GRP = '#0072B2', '#E69F00'
C_ACC, C_GRN, C_PNK, C_LIME = '#D55E00', '#009E73', '#CC79A7', '#A7D100'

# Pipeline rcParams with every text size raised (R6-6). Times where available,
# Liberation Serif is metric-compatible and stands in elsewhere.
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'axes.linewidth': 0.8,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'legend.frameon': False,
    'figure.dpi': 150,
    'savefig.dpi': DPI,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
})

TARGETS = [
    # key                 label              standard  group   unit
    ('Strength_28d', r"$f'_{c,28}$",        'C2_oof', 'C3_oof', 'MPa'),
    ('Strength_7d',  r"$f'_{c,7}$",         'E4_oof', 'E1_oof', 'MPa'),
    ('Slump_30',     r'Slump$_{30}$',       'E5_oof', 'E2_oof', 'mm'),
    ('Slump_90',     r'Slump$_{90}$',       'E6_oof', 'E3_oof', 'mm'),
]
LEARNERS = ['y_ens', 'y_XGBoost', 'y_LightGBM', 'y_CatBoost', 'y_ExtraTrees']
LEARNER_LABELS = ['Weighted\nensemble', 'XGBoost', 'LightGBM', 'CatBoost', 'ExtraTrees']


# --------------------------------------------------------------------- utils
def have(*names):
    return all(os.path.exists(os.path.join(DATA, f'{n}.csv')) for n in names)


def load(name):
    return pd.read_csv(os.path.join(DATA, f'{name}.csv'))


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return 1.0 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def panel(ax, letter, x=-0.14, y=1.08):
    ax.text(x, y, f'({letter})', transform=ax.transAxes, fontsize=13,
            fontweight='bold', va='top', ha='right')


def despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for fmt in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'{name}.{fmt}'), dpi=DPI,
                    bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f'  saved {name}.png / .pdf')




# ------------------------------------------------------------------ Figure 2
def fig2():
    """Prediction diagnostics, f'c,28 under standard 5-fold CV."""
    d = load('C2_oof')
    yt, yp = d.y_true.values, d.y_ens.values
    res = yt - yp
    rv2, rmse = r2(yt, yp), float(np.sqrt(np.mean(res ** 2)))
    mae = float(np.mean(np.abs(res)))
    mu, sig = res.mean(), res.std()

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))

    ax = axes[0]; panel(ax, 'a')
    ax.scatter(yt, yp, alpha=0.45, s=14, color=C_STD, edgecolors='none', rasterized=True)
    lims = [min(yt.min(), yp.min()) * 0.95, max(yt.max(), yp.max()) * 1.05]
    ax.plot(lims, lims, 'k--', lw=1.0)
    sl, ic, rr, _, _ = stats.linregress(yt, yp)
    xl = np.linspace(*lims, 100)
    ax.plot(xl, sl * xl + ic, color=C_ACC, lw=1.4, label=f'OLS (r = {rr:.3f})')
    ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect('equal')
    ax.set_xlabel(r"Measured $f'_{c,28}$ (MPa)")
    ax.set_ylabel(r"Predicted $f'_{c,28}$ (MPa)")
    ax.text(0.04, 0.96, f'R² = {rv2:.4f}\nRMSE = {rmse:.2f} MPa\nMAE = {mae:.2f} MPa',
            transform=ax.transAxes, fontsize=9.5, va='top',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white', alpha=0.9,
                      edgecolor='0.7'))
    ax.legend(fontsize=9.5, loc='lower right'); ax.grid(True, alpha=0.2); despine(ax)

    ax = axes[1]; panel(ax, 'b')
    ax.hist(res, bins=35, color=C_LIME, alpha=0.7, edgecolor='white',
            linewidth=0.4, density=True)
    xr = np.linspace(res.min() - 1, res.max() + 1, 200)
    ax.plot(xr, stats.norm.pdf(xr, mu, sig), 'k-', lw=1.4,
            label=f'N({mu:.2f}, {sig:.2f}²)')
    ax.axvline(0, color=C_ACC, lw=1.4, ls='--')
    ax.set_xlabel('Residual (MPa)'); ax.set_ylabel('Probability density')
    ax.legend(fontsize=9.5); ax.grid(axis='y', alpha=0.2); despine(ax)

    ax = axes[2]; panel(ax, 'c')
    ax.scatter(yp, res, alpha=0.45, s=14, color=C_GRN, edgecolors='none', rasterized=True)
    ax.axhline(0, color='k', lw=1.0, ls='--')
    for s in (2 * sig, -2 * sig):
        ax.axhline(s, color=C_GRP, lw=0.9, ls=':')
    si = np.argsort(yp)
    ax.plot(yp[si], uniform_filter1d(res[si], size=max(20, len(res) // 25)),
            color=C_ACC, lw=1.8, label='Smooth trend')
    ax.set_xlabel(r"Predicted $f'_{c,28}$ (MPa)"); ax.set_ylabel('Residual (MPa)')
    ax.legend(fontsize=9.5); ax.grid(True, alpha=0.2); despine(ax)

    fig.tight_layout(pad=0.6)
    save(fig, 'Figure2_prediction_diagnostics')


# ------------------------------------------------------------------ Figure 3
def fig3():
    """Generalisation gap: dual-validation bars, LOCO by class, class effect.

    Panel (b) is new. The R1 caption described a leave-one-class-out panel that
    the figure did not contain; rather than only correct the caption, the panel
    the caption promised is supplied, since it is the paper's strongest single
    piece of evidence that the gap is structural.
    """
    t3 = load('TABLE3_corrected').set_index(['target', 'scheme'])
    labels = [lab for _, lab, _, _, _ in TARGETS]
    std = [t3.loc[(k, 'standard'), 'R2'] for k, *_ in TARGETS]
    grp = [t3.loc[(k, 'group'), 'R2'] for k, *_ in TARGETS]
    gap = [s - g for s, g in zip(std, grp)]

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.6))

    # (a) dual validation
    ax = axes[0]; panel(ax, 'a')
    x = np.arange(len(labels)); bw = 0.34
    b1 = ax.bar(x - bw / 2, std, bw, color=C_STD, alpha=0.9, edgecolor='#333', lw=0.5,
                label='Standard 5-fold CV')
    b2 = ax.bar(x + bw / 2, grp, bw, color=C_GRP, alpha=0.9, edgecolor='#333', lw=0.5,
                label='GroupKFold CV')
    for bar, v in zip(b1, std):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.03, f'{v:.3f}',
                ha='center', va='bottom', fontsize=9.5, fontweight='bold')
    for bar, v in zip(b2, grp):
        ax.text(bar.get_x() + bar.get_width() / 2,
                (v + 0.03) if v >= 0 else (v - 0.03), f'{v:.3f}',
                ha='center', va='bottom' if v >= 0 else 'top',
                fontsize=9.5, fontweight='bold',
                color='#CC0000' if v < 0.1 else '#333')
    for i, g in enumerate(gap):
        ax.annotate(f'ΔR² = {g:.2f}', xy=(i, 1.05), fontsize=10, ha='center',
                    color='#CC0000' if g > 0.5 else C_GRP, fontweight='bold')
    ax.axhline(0, color='grey', lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('R²'); ax.set_ylim(-1.0, 1.22)
    ax.legend(fontsize=10, loc='lower left'); ax.grid(axis='y', alpha=0.3); despine(ax)

    # (b) leave-one-class-out
    ax = axes[1]; panel(ax, 'b')
    jk = load('S16_diagnostics')
    jk = jk[(jk.block == 'jackknife') & (jk.es == 'none')].copy()
    jk['cls'] = jk.variant.str.replace('without ', '', regex=False)
    jk = jk.sort_values('cls')
    xc = np.arange(len(jk)); bw = 0.36
    ax.bar(xc - bw / 2, jk.R2_standard, bw, color=C_STD, alpha=0.9,
           edgecolor='#333', lw=0.5, label='Standard 5-fold CV')
    ax.bar(xc + bw / 2, jk.R2_group, bw, color=C_GRP, alpha=0.9,
           edgecolor='#333', lw=0.5, label='GroupKFold CV')
    ax2 = ax.twinx()
    ax2.plot(xc, jk.dR2_group, 'o-', color=C_ACC, lw=1.8, ms=6, label='ΔR²')
    ax2.set_ylim(0, 0.39); ax2.set_ylabel('ΔR² (gap)', color=C_ACC)
    ax2.tick_params(axis='y', colors=C_ACC)
    ax2.spines['top'].set_visible(False)
    lo, hi = jk.dR2_group.min(), jk.dR2_group.max()
    ax2.axhspan(lo, hi, color=C_ACC, alpha=0.10)
    ax2.text(-0.45, 0.372, f'ΔR² band: {lo:.3f}–{hi:.3f}', fontsize=10,
             color=C_ACC, ha='left', va='top', fontweight='bold')
    ax.set_xticks(xc); ax.set_xticklabels(jk.cls, fontsize=10)
    ax.set_xlabel('Design class held out'); ax.set_ylabel('R²')
    ax.set_ylim(0, 1.30)
    ax.legend(fontsize=10, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.02))
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)

    # (c) the mechanism: train-validation covariate shift
    ax = axes[2]; panel(ax, 'c')
    s12 = load('S12_TableS12')
    smd = s12[s12.block == 'SMD'].set_index('variant')
    pairs = [('Mean SMD', 'mean_SMD'), ('Max SMD', 'max_SMD')]
    xx = np.arange(len(pairs)); w = 0.34
    rand = [smd.loc['Random folds', c] for _, c in pairs]
    grpd = [smd.loc['GroupKFold by class', c] for _, c in pairs]
    b1 = ax.bar(xx - w / 2, rand, w, color=C_STD, alpha=0.9, edgecolor='#333',
                lw=0.5, label='Random folds')
    b2 = ax.bar(xx + w / 2, grpd, w, color=C_GRP, alpha=0.9, edgecolor='#333',
                lw=0.5, label='GroupKFold by class')
    for bars, vals in ((b1, rand), (b2, grpd)):
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v * 1.12, f'{v:.2f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    for i, (r_, g_) in enumerate(zip(rand, grpd)):
        ax.annotate(f'×{g_ / r_:.1f}', xy=(i, max(r_, g_) * 2.0), fontsize=11,
                    ha='center', color=C_ACC, fontweight='bold')
    ax.axhline(0.1, color='grey', lw=0.8, ls=':')
    ax.text(0.5, 0.104, 'negligible shift', fontsize=9, color='grey',
            ha='center', va='bottom')
    ax.set_yscale('log'); ax.set_ylim(0.05, 16)
    ax.set_xticks(xx); ax.set_xticklabels([n for n, _ in pairs], fontsize=12)
    ax.set_ylabel('Standardised mean difference\n(train vs. validation, log scale)')
    ax.legend(fontsize=10, loc='upper left'); ax.grid(axis='y', alpha=0.3)
    despine(ax)

    fig.tight_layout(pad=1.0)
    save(fig, 'Figure3_generalisation_gap')


# ------------------------------------------------------------------ Figure 6
def fig6():
    """Per-learner R² heatmap: 5 models x 4 targets x 2 schemes."""
    rows = []
    for key, lab, f_std, f_grp in [(k, l, a, b) for k, l, a, b, _ in TARGETS]:
        for scheme, fname in (('Standard 5-fold', f_std), ('GroupKFold', f_grp)):
            d = load(fname)
            rows.append({'target': lab, 'scheme': scheme,
                         **{m: r2(d.y_true, d[m]) for m in LEARNERS}})
    df = pd.DataFrame(rows)

    mat = np.vstack([df[df.scheme == s][LEARNERS].values
                     for s in ('Standard 5-fold', 'GroupKFold')])
    ylab = [f'{l}\nStandard' for _, l, _, _, _ in TARGETS] + \
           [f'{l}\nGroupKFold' for _, l, _, _, _ in TARGETS]

    fig, ax = plt.subplots(figsize=(9.0, 7.0))
    im = ax.imshow(mat, cmap='RdYlGn', vmin=-1.0, vmax=1.0, aspect='auto')
    ax.set_xticks(range(len(LEARNERS))); ax.set_xticklabels(LEARNER_LABELS, fontsize=11)
    ax.set_yticks(range(len(ylab))); ax.set_yticklabels(ylab, fontsize=11)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            r_, g_, b_, _ = im.cmap(im.norm(v))
            lum = 0.299 * r_ + 0.587 * g_ + 0.114 * b_
            ax.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=11,
                    fontweight='bold', color='white' if lum < 0.55 else '#222')
    ax.axhline(3.5, color='#222', lw=2.2)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label('R²', fontsize=12); cb.ax.tick_params(labelsize=10)
    ax.set_title('Upper block: random folds   ·   Lower block: class-held-out folds',
                 fontsize=11.5, pad=12)
    fig.tight_layout(pad=0.6)
    save(fig, 'Figure6_heatmap_comparison')
    df.to_csv(os.path.join(OUT, 'Figure6_values.csv'), index=False)


# ------------------------------------------------------------------ Figure 9
def fig9():
    """Pareto front, with and without the applicability-domain filter."""
    pub, filt = load('S15_front_published'), load('S15_front_filtered')

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.4))

    ax = axes[0, 0]; panel(ax, 'a')
    ax.scatter(pub.Cement, pub.fc28, s=48, facecolors='none', edgecolors='#999',
               lw=1.1, label=f'Marginal bounds only (n = {len(pub)})')
    ax.scatter(filt.Cement, filt.fc28, s=42, color=C_ACC, alpha=0.9,
               edgecolors='#333', lw=0.4,
               label=f'Within applicability domain (n = {len(filt)})')
    ax.set_xlabel('Cement content (kg/m³)')
    ax.set_ylabel(r"Predicted $f'_{c,28}$ (MPa)")
    ax.legend(fontsize=9.5, loc='lower right'); ax.grid(True, alpha=0.25); despine(ax)

    ax = axes[0, 1]; panel(ax, 'b')
    ax.scatter(filt.Cement, filt.Eff, s=42, color=C_GRN, alpha=0.9,
               edgecolors='#333', lw=0.4)
    ax.set_xlabel('Cement content (kg/m³)')
    ax.set_ylabel('Cement efficiency (MPa per kg/m³)')
    ax.grid(True, alpha=0.25); despine(ax)

    ax = axes[1, 0]; panel(ax, 'c')
    ax.scatter(filt.WC, filt.fc28, s=42, color=C_STD, alpha=0.9,
               edgecolors='#333', lw=0.4)
    ax.set_xlabel('W/C ratio'); ax.set_ylabel(r"Predicted $f'_{c,28}$ (MPa)")
    ax.grid(True, alpha=0.25); despine(ax)

    ax = axes[1, 1]; panel(ax, 'd')
    top = filt.nlargest(10, 'Eff').reset_index(drop=True)
    ax.axis('off')
    cells = [[f'{r.Cement:.0f}', f'{r.WC:.3f}', f'{r.fc28:.1f}', f'{r.Eff:.4f}']
             for r in top.itertuples()]
    tb = ax.table(cellText=cells,
                  colLabels=['Cement\n(kg/m³)', 'W/C', r"$f'_{c,28}$" + '\n(MPa)',
                             'Efficiency\n(MPa per kg/m³)'],
                  loc='center', cellLoc='center')
    tb.auto_set_font_size(False); tb.set_fontsize(10); tb.scale(1.0, 1.55)
    for (row, _), cell in tb.get_celld().items():
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_text_props(fontweight='bold')
            cell.set_facecolor('#EEEEEE')
    ax.set_title('Ten most cement-efficient candidates', fontsize=11.5, pad=16)

    fig.tight_layout(pad=0.9)
    save(fig, 'Figure9_pareto_front')


# ----------------------------------------------------------------- Figure S1
def figS1():
    """Diagnostics for the three secondary targets under standard CV."""
    rows = [(lab, f_std, unit) for _, lab, f_std, _, unit in TARGETS[1:]]
    fig, axes = plt.subplots(3, 3, figsize=(11.0, 10.0))

    for r, (lab, fname, unit) in enumerate(rows):
        d = load(fname)
        yt, yp = d.y_true.values, d.y_ens.values
        res = yt - yp
        sig = res.std()

        ax = axes[r, 0]
        if r == 0:
            panel(ax, 'a')
        ax.scatter(yt, yp, alpha=0.45, s=13, color=C_STD, edgecolors='none',
                   rasterized=True)
        lims = [min(yt.min(), yp.min()) * 0.9, max(yt.max(), yp.max()) * 1.08]
        ax.plot(lims, lims, 'k--', lw=1.0)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel(f'Measured {lab} ({unit})')
        ax.set_ylabel(f'Predicted {lab} ({unit})')
        ax.text(0.04, 0.96, f'R² = {r2(yt, yp):.4f}', transform=ax.transAxes,
                fontsize=10, va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          alpha=0.9, edgecolor='0.7'))
        ax.grid(True, alpha=0.2); despine(ax)

        ax = axes[r, 1]
        if r == 0:
            panel(ax, 'b')
        ax.hist(res, bins=30, color=C_LIME, alpha=0.7, edgecolor='white', lw=0.4,
                density=True)
        ax.axvline(0, color=C_ACC, lw=1.4, ls='--')
        ax.set_xlabel(f'Residual ({unit})'); ax.set_ylabel('Probability density')
        ax.grid(axis='y', alpha=0.2); despine(ax)

        ax = axes[r, 2]
        if r == 0:
            panel(ax, 'c')
        ax.scatter(yp, res, alpha=0.45, s=13, color=C_GRN, edgecolors='none',
                   rasterized=True)
        ax.axhline(0, color='k', lw=1.0, ls='--')
        for s in (2 * sig, -2 * sig):
            ax.axhline(s, color=C_GRP, lw=0.9, ls=':')
        ax.set_xlabel(f'Predicted {lab} ({unit})'); ax.set_ylabel(f'Residual ({unit})')
        ax.grid(True, alpha=0.2); despine(ax)

    fig.tight_layout(pad=0.9)
    save(fig, 'FigureS1_secondary_target_diagnostics')


# --------------------------------------------------------------------- main
FIGURES = {
    '2':  (fig2,  ['C2_oof']),
    '3':  (fig3,  ['TABLE3_corrected', 'S16_diagnostics', 'S12_TableS12']),
    '6':  (fig6,  ['C2_oof', 'C3_oof', 'E1_oof', 'E2_oof', 'E3_oof',
                   'E4_oof', 'E5_oof', 'E6_oof']),
    '9':  (fig9,  ['S15_front_published', 'S15_front_filtered']),
    'S1': (figS1, ['E4_oof', 'E5_oof', 'E6_oof']),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', help='figure key, e.g. 3 or S1')
    a = ap.parse_args()

    keys = [a.only] if a.only else list(FIGURES)
    for k in keys:
        fn, needs = FIGURES[k]
        if not have(*needs):
            missing = [n for n in needs if not have(n)]
            print(f'  SKIP Figure {k} -- missing {missing}')
            continue
        print(f'Figure {k}:')
        fn()




# ------------------------------------------------------------------ Figure 4
FEATURE_LABELS = {
    'Strength_7d': r"$f'_{c,7}$", 'Concrete_Class_sq': 'Class$^2$',
    'Cement_sq': 'Cement$^2$', 'Admix_Freq': 'Admixture (freq.)',
    'Cement_Freq': 'Cement brand (freq.)', 'Class_x_WC': 'Class x W/C',
    'WC_Ratio': 'W/C ratio', 'WC_Ratio_sq': '(W/C)$^2$',
    'Slump_Retention': 'Slump retention', 'log_Cement': 'log cement',
    'Binder_Intensity': 'Binder intensity', 'Bolomey_Feature': 'Bolomey term',
    'Slump_30': r'Slump$_{30}$', 'Class_x_Cement': 'Class x cement',
    'Gel_Space_Ratio_28d': 'Gel/space ratio', 'Cement_x_WC': 'Cement x W/C',
    'FA': 'Fine aggregate', 'WC_Ratio_inv': '1/(W/C)',
    'Slump_Loss_Rate': 'Slump loss rate', 'Cement_Content': 'Cement content',
    'Concrete_Class': 'Design class', 'Admix_per_Cement': 'Admixture / cement',
    'CA_FA_x_WC': '(CA/FA) x W/C', 'Water_Content': 'Water content',
    'Paste_Volume': 'Paste volume', 'Admix_pct_bwoc': 'Admixture % bwoc',
    'CA_FA_Ratio': 'CA/FA ratio', 'Admixture_Dose': 'Admixture dose',
    'CA': 'Coarse aggregate', 'Agg_Volume_Frac': 'Aggregate vol. frac.',
}


def _lbl(f):
    return FEATURE_LABELS.get(f, f.replace('_', ' '))


def fig4():
    """TreeSHAP attribution, out-of-fold, under both validation schemes.

    The R1 figure came from a single in-sample fit on the whole dataset
    (pipeline S7), which §3.9 described as fold-wise. These are genuine
    out-of-fold attributions, and panel (d) is new: it is the comparison
    Reviewer 3 item 4 asked for.
    """
    std = load('X4_shap_standard')
    grp = load('X4_shap_group')
    feats = [c[6:] for c in std.columns if c.startswith('shap__')]

    S = std[[f'shap__{f}' for f in feats]].values
    V = std[[f'val__{f}' for f in feats]].values
    mean_abs = np.abs(S).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:12]          # top 12, strongest first

    fig = plt.figure(figsize=(13.5, 9.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1.0], hspace=0.62, wspace=0.42)

    # (a) beeswarm
    ax = fig.add_subplot(gs[0, 0]); panel(ax, 'a', x=-0.30)
    rng = np.random.default_rng(0)
    for row, fi in enumerate(order):
        y = len(order) - 1 - row
        sv, vv = S[:, fi], V[:, fi]
        rank = (np.argsort(np.argsort(vv)) / max(len(vv) - 1, 1))
        jitter = rng.uniform(-0.17, 0.17, len(sv))
        ax.scatter(sv, y + jitter, c=rank, cmap='coolwarm', s=5, alpha=0.6,
                   edgecolors='none', rasterized=True)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([_lbl(feats[fi]) for fi in order][::-1], fontsize=9)
    ax.axvline(0, color='k', lw=0.8, ls='--')
    ax.set_xlabel('SHAP value (MPa)')
    sm = plt.cm.ScalarMappable(cmap='coolwarm')
    cb = fig.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.05,
                      pad=0.20, ticks=[0, 1])
    cb.ax.set_xticklabels(['low', 'high'], fontsize=9)
    cb.set_label('Feature value', fontsize=10)
    ax.set_ylim(-0.6, len(order) - 0.4)
    despine(ax)

    # (b) global ranking
    ax = fig.add_subplot(gs[0, 1]); panel(ax, 'b', x=-0.30)
    ax.barh(range(len(order)), mean_abs[order][::-1], color=C_STD, alpha=0.9,
            edgecolor='#333', lw=0.4)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([_lbl(feats[fi]) for fi in order][::-1], fontsize=9)
    ax.set_xlabel('Mean |SHAP| (MPa)')
    for i, v in enumerate(mean_abs[order][::-1]):
        ax.text(v + mean_abs.max() * 0.015, i, f'{v:.2f}', va='center', fontsize=8.5)
    ax.set_xlim(0, mean_abs.max() * 1.18)
    ax.grid(axis='x', alpha=0.25); despine(ax)

    # (c) anchor dependence
    ax = fig.add_subplot(gs[0, 2]); panel(ax, 'c', x=-0.26)
    fi = feats.index('Strength_7d')
    ax.scatter(V[:, fi], S[:, fi], s=12, alpha=0.5, color=C_ACC,
               edgecolors='none', rasterized=True)
    ax.set_xlabel(r"$f'_{c,7}$ (MPa)"); ax.set_ylabel('SHAP value (MPa)')
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.grid(True, alpha=0.25); despine(ax)

    # (d) the scheme comparison -- this is the new panel (R3-4)
    ax = fig.add_subplot(gs[1, :]); panel(ax, 'd', x=-0.055)
    ms = np.abs(S).mean(axis=0)
    mg = np.abs(grp[[f'shap__{f}' for f in feats]].values).mean(axis=0)
    rs = len(feats) - np.argsort(np.argsort(ms))
    rg = len(feats) - np.argsort(np.argsort(mg))
    keep = np.argsort(np.maximum(ms, mg))[::-1][:16]
    keep = keep[np.argsort(rs[keep])]
    x = np.arange(len(keep))
    ax.plot(x, rs[keep], 'o-', color=C_STD, lw=1.8, ms=7, label='Standard 5-fold CV')
    ax.plot(x, rg[keep], 's-', color=C_GRP, lw=1.8, ms=7, label='GroupKFold')
    for i, fi2 in enumerate(keep):
        if abs(rs[fi2] - rg[fi2]) >= 8:
            ax.annotate('', xy=(i, rg[fi2]), xytext=(i, rs[fi2]),
                        arrowprops=dict(arrowstyle='->', color=C_ACC, lw=1.6))
    ax.set_xticks(x)
    ax.set_xticklabels([_lbl(feats[i]) for i in keep], rotation=38,
                       ha='right', fontsize=9)
    ax.set_ylim(len(feats) * 0.72, 0.4)          # inverted, rank 1 at the top
    ax.set_ylabel('Rank by mean |SHAP|\n(1 = strongest)')
    ax.legend(fontsize=10, ncol=2, loc='lower center',
              bbox_to_anchor=(0.5, 1.01), frameon=False)
    ax.grid(axis='y', alpha=0.25)
    despine(ax)

    save(fig, 'Figure4_treeshap_attribution')


# ------------------------------------------------------------------ Figure 5
def fig5():
    """Bootstrap distributions of R2, RMSE and MAE over 200 OOB replicates."""
    d = load('X3_bootstrap_replicates')
    panels = [('R2', 'R²', '', C_STD),
              ('RMSE', 'RMSE', ' (MPa)', C_GRN),
              ('MAE', 'MAE', ' (MPa)', C_PNK)]

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))
    for ax, (col, name, unit, colour) in zip(axes, panels):
        v = d[col].values
        med, lo, hi = np.median(v), np.percentile(v, 2.5), np.percentile(v, 97.5)
        ax.hist(v, bins=28, color=colour, alpha=0.65, edgecolor='white', lw=0.5)
        ax.axvline(med, color=C_ACC, lw=2.0, label=f'median {med:.4f}'
                   if col == 'R2' else f'median {med:.2f}')
        for b in (lo, hi):
            ax.axvline(b, color='#444', lw=1.2, ls=':')
        ax.axvspan(lo, hi, color=C_ACC, alpha=0.07)
        ax.set_xlabel(f'{name}{unit}')
        ax.set_ylabel('Replicates' if col == 'R2' else '')
        ax.set_title(f'95 % CI [{lo:.4f}, {hi:.4f}]' if col == 'R2'
                     else f'95 % CI [{lo:.2f}, {hi:.2f}]', fontsize=10.5)
        ax.legend(fontsize=9.5, loc='upper left')
        ax.grid(axis='y', alpha=0.25); despine(ax)
        panel(ax, 'abc'[panels.index((col, name, unit, colour))], x=-0.16)

    fig.suptitle(f'B = {len(d)} out-of-bag replicates, fixed-parameter CatBoost, '
                 'standard-CV metric only', fontsize=10.5, y=1.04)
    fig.tight_layout(pad=0.7)
    save(fig, 'Figure5_bootstrap_distributions')


FIGURES['4'] = (fig4, ['X4_shap_standard', 'X4_shap_group'])
FIGURES['5'] = (fig5, ['X3_bootstrap_replicates'])


# ------------------------------------------------------------------ Figure 8
def fig8():
    """Code-compliance margin attainment by class, measured vs predicted.

    Recomputed from the corrected standard-CV ensemble OOF, because the
    predicted column of the compliance table moved when the early-stopping
    leak was removed.
    """
    d = load('C2_oof')

    def fcr(fck, s):
        return (max(fck + 1.34 * s, fck + 2.33 * s - 3.45) if fck <= 35
                else max(fck + 1.34 * s, 0.90 * fck + 2.33 * s))

    rows = []
    for cls, g in d.groupby('Concrete_Class'):
        thr = fcr(cls, g.y_true.std(ddof=0))
        rows.append(dict(cls=f'M{cls}', n=len(g),
                         aci_m=100 * (g.y_true >= thr).mean(),
                         aci_p=100 * (g.y_ens >= thr).mean(),
                         ec2_m=100 * (g.y_true >= cls + 8).mean(),
                         ec2_p=100 * (g.y_ens >= cls + 8).mean()))
    t = pd.DataFrame(rows)
    x = np.arange(len(t)); bw = 0.36

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    for i, (ax, (mc, pc, name, letter)) in enumerate(zip(
            axes[:2], [('aci_m', 'aci_p', 'ACI 318-19 required average', 'a'),
                       ('ec2_m', 'ec2_p', 'Eurocode 2 indicator', 'b')])):
        panel(ax, letter)
        ax.bar(x - bw / 2, t[mc], bw, color=C_STD, alpha=0.9, edgecolor='#333',
               lw=0.5, label='Measured')
        ax.bar(x + bw / 2, t[pc], bw, color=C_GRP, alpha=0.9, edgecolor='#333',
               lw=0.5, label='Predicted')
        # Vertical, so the two labels never run together where both read 100.
        for xi, (m, p) in enumerate(zip(t[mc], t[pc])):
            ax.text(xi - bw / 2, m + 2, f'{m:.0f}', ha='center', va='bottom',
                    fontsize=9, rotation=90)
            ax.text(xi + bw / 2, p + 2, f'{p:.0f}', ha='center', va='bottom',
                    fontsize=9, rotation=90)
        ax.set_xticks(x); ax.set_xticklabels(t.cls, fontsize=10.5)
        ax.set_ylabel('Batches reaching the target (%)')
        ax.set_ylim(0, 134); ax.set_yticks(np.arange(0, 101, 20))
        ax.set_title(name, fontsize=11.5, pad=8)
        ax.axhline(100, color='grey', lw=0.7, ls=':')
        if i == 0:                       # one legend serves both bar panels
            ax.legend(fontsize=9.5, loc='upper left', ncol=2,
                      columnspacing=1.2, handlelength=1.4)
        ax.grid(axis='y', alpha=0.25)
        despine(ax)

    ax = axes[2]; panel(ax, 'c')
    ax.plot(x, t.aci_m, 'o-', color=C_STD, lw=1.8, ms=7, label='ACI, measured')
    ax.plot(x, t.aci_p, 'o--', color=C_STD, lw=1.4, ms=6, alpha=0.6,
            label='ACI, predicted')
    ax.plot(x, t.ec2_m, 's-', color=C_GRN, lw=1.8, ms=7, label='EC2, measured')
    ax.plot(x, t.ec2_p, 's--', color=C_GRN, lw=1.4, ms=6, alpha=0.6,
            label='EC2, predicted')
    lo = float(t[['aci_m', 'aci_p', 'ec2_m', 'ec2_p']].values.min())
    ax.annotate('M45 carries the\nthinnest margin', xy=(5.82, lo + 0.5),
                xytext=(4.15, lo - 1.5), fontsize=10, color=C_ACC,
                ha='center', va='center',
                arrowprops=dict(arrowstyle='->', color=C_ACC, lw=1.6))
    ax.set_xticks(x); ax.set_xticklabels(t.cls, fontsize=10.5)
    ax.set_ylabel('Batches reaching the target (%)')
    ax.set_ylim(lo - 22, 105)            # empty band below the data for the key
    ax.set_title('Both criteria overlaid', fontsize=11.5, pad=8)
    ax.legend(fontsize=9.5, loc='lower center', ncol=2, columnspacing=1.2,
              handlelength=2.0)
    ax.grid(axis='y', alpha=0.25)
    despine(ax)

    fig.tight_layout(pad=0.8)
    save(fig, 'Figure8_code_compliance')
    t.to_csv(os.path.join(OUT, 'Figure8_values.csv'), index=False)


FIGURES['8'] = (fig8, ['C2_oof'])



# ------------------------------------------------------------------ Figure 7
def fig7():
    """Physical-law diagnostics.

    Panels (a), (c), (d) and (f) plot measured quantities and are unchanged by
    the correction; they are regenerated only for the larger labels (R6-6), and
    reproduce the published fits exactly. Panels (b) and (e) are model-derived
    and are rebuilt from the corrected out-of-fold predictions.

    Regenerating this figure also settled a disagreement inside the R1
    submission: §4.7 reported Abrams constants K1 = 152, K2 = 21.05 and a
    maturity intercept of 9.0, while the figure printed alongside it read
    155 / 22.06 and 8.8. Refitting from the released data reproduces the
    figure, so the prose was corrected to match.
    """
    d, p = load_raw()
    oof = load('C2_oof')
    yt = d.Strength_28d.values.astype(float)
    f7 = d.Strength_7d.values.astype(float)
    yp = oof.y_ens.values
    wc, gsr, bi = p.WC_Ratio.values, p.GSR.values, p.Binder_Intensity.values
    cls = d.Concrete_Class.values.astype(int)
    res = yt - yp

    # Kept compact on purpose: the page shows this at 6.5 in, so a wider
    # canvas would scale the labels back down (R6-6).
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.9))
    sc_kw = dict(cmap='viridis', s=16, alpha=0.55, edgecolors='none',
                 rasterized=True)
    wcf = np.linspace(wc.min(), wc.max(), 100)
    v = (yt > 0) & (wc > 0)

    # (a) Abrams on measured strength
    ax = axes[0, 0]; panel(ax, 'a')
    ax.scatter(wc, yt, c=cls, **sc_kw)
    sl_a, ic_a, r_a, _, _ = stats.linregress(wc[v], np.log(yt[v]))
    ax.plot(wcf, np.exp(ic_a + sl_a * wcf), '--', color=C_ACC, lw=1.8,
            label=f"Abrams': $f_c={np.exp(ic_a):.0f}/{np.exp(-sl_a):.2f}^{{w/c}}$"
                  f'  (r = {r_a:.3f})')
    ax.set_xlabel('W/C ratio'); ax.set_ylabel(r"Measured $f'_{c,28}$ (MPa)")
    ax.legend(fontsize=9.5, loc='upper right'); ax.grid(alpha=0.25); despine(ax)

    # (b) the same fit on the corrected predictions
    ax = axes[0, 1]; panel(ax, 'b')
    ax.scatter(wc, yp, c=cls, **sc_kw)
    sl_p, ic_p, r_p, _, _ = stats.linregress(wc[v], np.log(np.maximum(yp[v], 1e-3)))
    ax.plot(wcf, np.exp(ic_p + sl_p * wcf), ':', color=C_STD, lw=1.8,
            label=f'Model fit  (r = {r_p:.3f})')
    ax.plot(wcf, np.exp(ic_a + sl_a * wcf), '--', color=C_ACC, lw=1.3,
            label="Abrams' curve")
    ax.set_xlabel('W/C ratio'); ax.set_ylabel(r"Predicted $f'_{c,28}$ (MPa)")
    ax.legend(fontsize=9.5, loc='upper right'); ax.grid(alpha=0.25); despine(ax)

    # (c) maturity
    ax = axes[0, 2]; panel(ax, 'c')
    ax.scatter(f7, yt, c=cls, **sc_kw)
    sl_m, ic_m, r_m, _, _ = stats.linregress(f7, yt)
    xf = np.linspace(f7.min(), yt.max(), 100)
    ax.plot(xf, sl_m * xf + ic_m, '-', color=C_ACC, lw=1.8,
            label=f'$f_{{28}} = {sl_m:.2f}f_{{7}} + {ic_m:.1f}$  (r = {r_m:.3f})')
    ax.plot(xf, xf, ':', color='grey', lw=1.0, label='1:1 line')
    ax.set_xlabel(r"$f'_{c,7}$ (MPa)"); ax.set_ylabel(r"$f'_{c,28}$ (MPa)")
    ax.legend(fontsize=9.5, loc='upper left'); ax.grid(alpha=0.25); despine(ax)

    # (d) binder intensity, coloured by W/C
    ax = axes[1, 0]; panel(ax, 'd')
    s = ax.scatter(bi, yt, c=wc, cmap='RdBu_r', s=16, alpha=0.6,
                   edgecolors='none', rasterized=True)
    cb = fig.colorbar(s, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label('W/C ratio', fontsize=10); cb.ax.tick_params(labelsize=9)
    ax.set_xlabel(r'Binder intensity (kg/m³ per MPa of class)')
    ax.set_ylabel(r"$f'_{c,28}$ (MPa)")
    ax.grid(alpha=0.25); despine(ax)

    # (e) residuals by class, from the corrected out-of-fold predictions
    ax = axes[1, 1]; panel(ax, 'e')
    order = sorted(np.unique(cls))
    bp = ax.boxplot([res[cls == c] for c in order], widths=0.62,
                    patch_artist=True, tick_labels=[f'M{c}' for c in order],
                    medianprops=dict(color=C_ACC, lw=1.6),
                    flierprops=dict(marker='o', ms=3, mfc='none',
                                    mec='0.55', alpha=0.7))
    for box, c in zip(bp['boxes'], plt.cm.viridis(np.linspace(0, 1, len(order)))):
        box.set_facecolor(c); box.set_alpha(0.65); box.set_edgecolor('#333')
    ax.axhline(0, color='grey', ls='--', lw=0.9)
    ax.set_xlabel('Design class'); ax.set_ylabel('Residual (MPa)')
    ax.tick_params(axis='x', labelsize=10)
    ax.grid(axis='y', alpha=0.25); despine(ax)

    # (f) Powers' gel/space ratio
    ax = axes[1, 2]; panel(ax, 'f')
    ax.scatter(gsr, yt, c=cls, **sc_kw)
    vg = (gsr > 0) & (yt > 0)
    sl_w, ic_w, r_w, _, _ = stats.linregress(np.log(gsr[vg]), np.log(yt[vg]))
    gf = np.linspace(gsr.min(), gsr.max(), 100)
    ax.plot(gf, np.exp(ic_w) * gf ** sl_w, '--', color=C_ACC, lw=1.8,
            label=f"Powers': $f_c={np.exp(ic_w):.0f}X^{{{sl_w:.2f}}}$"
                  f'  (r = {r_w:.3f})')
    ax.set_xlabel("Gel/space ratio (Powers')"); ax.set_ylabel(r"$f'_{c,28}$ (MPa)")
    ax.legend(fontsize=9.5, loc='upper left'); ax.grid(alpha=0.25); despine(ax)

    fig.tight_layout(pad=0.7, w_pad=1.6, h_pad=1.9)
    save(fig, 'Figure7_physical_law_diagnostics')

    pd.DataFrame([dict(panel='a', quantity='Abrams K1', value=np.exp(ic_a)),
                  dict(panel='a', quantity='Abrams K2', value=np.exp(-sl_a)),
                  dict(panel='a', quantity='Abrams r', value=r_a),
                  dict(panel='b', quantity='model fit r', value=r_p),
                  dict(panel='c', quantity='maturity slope', value=sl_m),
                  dict(panel='c', quantity='maturity intercept', value=ic_m),
                  dict(panel='c', quantity='maturity r', value=r_m),
                  dict(panel='f', quantity='Powers K', value=np.exp(ic_w)),
                  dict(panel='f', quantity='Powers exponent', value=sl_w),
                  dict(panel='f', quantity='Powers r', value=r_w)]
                 ).to_csv(os.path.join(OUT, 'Figure7_values.csv'), index=False)


FIGURES['7'] = (fig7, ['C2_oof'])


if __name__ == '__main__':
    main()
