"""extras2_R2.py -- the three supplementary tables extras_R2.py did not cover.

Same contract as extras_R2.py: the SAME fixed-parameter CatBoost used for every
diagnostic in the paper (CB_ABLATION_PARAMS), fitted WITHOUT the early-stopping
leak (no eval_set anywhere), so nothing here inherits the defect described in
Part A of the response. Each block skips itself if its output already exists.

  X7  Table S4 -- permutation importance for the full 30-feature model, mean
      dR2 over the GroupKFold validation folds. The published S4 came from the
      leaked run, and manuscript 4.3 quotes two numbers from it (f'c,7 costs
      dR2 = 0.96; every other feature <= 0.01).

  X8  Table S5 -- mean |SHAP| for the full and no-anchor models under a
      full-data fit, which is what the published S5 reports and what 3.9
      describes. X4 stored the out-of-fold matrices under each CV scheme,
      which is a different quantity and cannot substitute.

  X9  Table S6b, second column -- the no-anchor variant of the repeated
      class-to-fold assignment experiment. X5 measured the full model only,
      so S6b's no-anchor column is still the leaked one.

Run:
    python extras2_R2.py            # all three (~10 min, most of it X9)
    python extras2_R2.py --jobs X7  # one block
"""
import argparse
import os

import numpy as np
import pandas as pd

import rerun_R2 as R
from extras_R2 import metrics

N_PERM_REPEATS = 10       # permutation repeats per feature per fold
N_ASSIGNMENTS = 10        # replicates for X9, matching the published S6b


def _fit(tr, feats, target='Strength_28d'):
    avail = [f for f in feats if f in tr.columns]
    m = R.CatBoostRegressor(**R.CB_ABLATION_PARAMS, random_seed=R.SEED,
                            verbose=0)
    m.fit(tr[avail].values.astype(np.float32),
          tr[target].values.astype(np.float32), verbose=0)
    return m, avail


def _r2(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    return 1.0 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


# --------------------------------------------------------------------------
def run_X7(df):
    """Table S4: permutation importance over the GroupKFold validation folds."""
    out = os.path.join(R.RES_DIR, 'X7_tableS4_permutation.csv')
    if os.path.exists(out):
        R.log('SKIP X7 (already done)'); return

    feats = list(R.FEAT_28D)
    groups = df['Concrete_Class'].values
    k = min(5, df['Concrete_Class'].nunique())
    splits = list(R.GroupKFold(n_splits=k).split(df, groups=groups))

    per_fold = {f: [] for f in feats}
    for fi, (tr_idx, va_idx) in enumerate(splits):
        tr, va, _ = R.preprocess_fold(df.iloc[tr_idx], df.iloc[va_idx])
        m, avail = _fit(tr, feats)
        Xva = va[avail].values.astype(np.float32)
        yva = va['Strength_28d'].values.astype(np.float32)
        base = _r2(yva, m.predict(Xva))

        for j, f in enumerate(avail):
            rng = np.random.default_rng(R.SEED + 1000 * fi + j)
            drops = []
            for _ in range(N_PERM_REPEATS):
                Xp = Xva.copy()
                Xp[:, j] = Xp[rng.permutation(len(Xp)), j]
                drops.append(base - _r2(yva, m.predict(Xp)))
            per_fold[f].append(float(np.mean(drops)))
        R.log(f'X7  fold {fi + 1}/{len(splits)}  baseline R2 = {base:+.4f}')

    d = pd.DataFrame([
        dict(feature=f,
             mean_dR2=float(np.mean(v)),
             sd_dR2=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
             n_folds=len(v))
        for f, v in per_fold.items() if v
    ]).sort_values('mean_dR2', ascending=False).reset_index(drop=True)
    d.to_csv(out, index=False)
    print('\n' + d.head(10).to_string(index=False))
    R.log(f'DONE  X7 -> {out}')


# --------------------------------------------------------------------------
def run_X8(df):
    """Table S5: mean |SHAP| under a full-data fit, full and no-anchor."""
    from catboost import Pool
    out = os.path.join(R.RES_DIR, 'X8_tableS5_shap_fulldata.csv')
    if os.path.exists(out):
        R.log('SKIP X8 (already done)'); return

    full = list(R.FEAT_28D)
    noanchor = [f for f in full if f != 'Strength_7d']

    # A full-data fit has no held-out rows, so the fold preprocessing is
    # applied with the whole dataset as both sides: winsorisation bounds and
    # frequency maps are fitted on all 683 records, which is what "full-data
    # fit" means in 3.9 and is how the released pipeline built df_full_proc.
    proc, _, _ = R.preprocess_fold(df, df)

    rows = []
    for cfg, feats in [('Full (30 features)', full),
                       ("No f'c,7 (29 feat.)", noanchor)]:
        m, avail = _fit(proc, feats)
        X = proc[avail].values.astype(np.float32)
        sv = m.get_feature_importance(Pool(X), type='ShapValues')[:, :-1]
        mean_abs = np.abs(sv).mean(axis=0)
        order = np.argsort(-mean_abs)
        for rank, j in enumerate(order, 1):
            rows.append(dict(configuration=cfg, rank=rank,
                             feature=avail[j],
                             mean_abs_shap=float(mean_abs[j])))
        R.log(f'X8  {cfg:20s} ({len(avail)} features) '
              f'top = {avail[order[0]]} {mean_abs[order[0]]:.3f}')

    d = pd.DataFrame(rows)
    d.to_csv(out, index=False)
    for cfg in d.configuration.unique():
        print(f'\n  {cfg}')
        print(d[d.configuration == cfg].head(10).to_string(index=False))
    R.log(f'DONE  X8 -> {out}')


# --------------------------------------------------------------------------
def run_X9(df):
    """Table S6b: the no-anchor arm of the fold-assignment spread.

    Identical construction to X5 -- permute the eight classes, deal them
    round-robin into five folds, so no class is ever split -- but with the
    29-feature no-anchor input set, and over 10 assignments rather than 20,
    matching the published S6b.
    """
    out = os.path.join(R.RES_DIR, 'X9_tableS6b_noanchor_spread.csv')
    if os.path.exists(out):
        R.log('SKIP X9 (already done)'); return

    y = df['Strength_28d'].values
    feats = [f for f in R.FEAT_28D if f != 'Strength_7d']
    classes = np.array(sorted(df['Concrete_Class'].unique()))
    rows = []

    for rep in range(N_ASSIGNMENTS):
        # Same seed sequence as X5, so replicate r is the SAME class-to-fold
        # assignment in both experiments and the two columns of S6b are paired.
        rng = np.random.default_rng(R.SEED + rep)
        perm = rng.permutation(len(classes))
        fold_of = {classes[perm[i]]: i % 5 for i in range(len(classes))}
        assign = df['Concrete_Class'].map(fold_of).values

        y_oof = np.zeros(len(df))
        for k in range(5):
            va = np.where(assign == k)[0]
            tr = np.where(assign != k)[0]
            if len(va) == 0:
                continue
            tr_d, va_d, _ = R.preprocess_fold(df.iloc[tr], df.iloc[va])
            m, avail = _fit(tr_d, feats)
            y_oof[va] = m.predict(va_d[avail].values.astype(np.float32))

        rows.append(dict(replicate=rep,
                         classes_per_fold=';'.join(
                             str(sorted(int(c) for c, f in fold_of.items() if f == k))
                             for k in range(5)),
                         **metrics(y, y_oof)))
        R.log(f'X9  replicate {rep + 1}/{N_ASSIGNMENTS}  '
              f'R2 = {rows[-1]["R2"]:+.4f}')

    d = pd.DataFrame(rows)
    d.to_csv(out, index=False)
    v = d['R2']
    print(f'\n  mean {v.mean():.4f}  SD {v.std(ddof=1):.4f}  '
          f'range [{v.min():.4f}, {v.max():.4f}]  (n = {len(d)})')
    R.log(f'DONE  X9 -> {out}')


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', default='ALL', help='ALL | X7 | X8 | X9')
    a = ap.parse_args()

    df = R.load_data()
    R.log(f'Data: n={len(df)}, classes={df.Concrete_Class.nunique()}')
    assert len(df) == 683, f'EXPECTED 683 RECORDS, GOT {len(df)}'

    sel = a.jobs.strip().upper()
    for name, fn in [('X7', run_X7), ('X8', run_X8), ('X9', run_X9)]:
        if sel in ('ALL', name):
            fn(df)


if __name__ == '__main__':
    main()
