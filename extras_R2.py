"""extras_R2.py -- the gaps left in the corrected result set.

Every block re-fits the SAME fixed-parameter CatBoost used for every diagnostic
in the paper (CB_ABLATION_PARAMS), but WITHOUT the early-stopping leak, i.e.
es='none'.  Nothing here is tuned, so it is fast (~25 min total, most of it X3 and X5).

  X1  Table 6 -- anchor ablation with the full metric set.
      S16_diagnostics.csv stored only R2 and RMSE; Table 6 in the manuscript
      also prints MAE and MAPE.  This recomputes all four from OOF predictions
      so the table can be filled from measured numbers rather than carried over
      from the leaked run.

  X2  Table S1 -- winsorisation sensitivity under the corrected protocol.
      The published Table S1 was produced with the leaked early stopping.
      This repeats the three settings under both validation schemes.

  X3  Figure 5 -- the 200 bootstrap replicates themselves, so the R2/RMSE/MAE
      panels can be redrawn. run_DIAG stored only the median and the CI.

  X4  Figure 4 -- the per-sample TreeSHAP matrices under both schemes, so the
      beeswarm and dependence panels can be redrawn. run_SHAP stored only the
      column means.

  X6  The fixed-parameter CatBoost OOF predictions themselves, which §4.8's
      screening statistics are computed from and which run_DIAG discarded.

  X5  The 20 repeated class-to-fold assignments behind the 0.816 +- 0.011
      quoted in §4.5 and §4.10. That figure is from the leaked run, and the
      corrected value cannot be inferred from it -- it has to be measured.

Every block skips itself if its output already exists, so re-running is safe.

Run:
    python extras_R2.py            # every block
    python extras_R2.py --jobs X3  # one block
"""
import argparse
import os

import numpy as np
import pandas as pd

import rerun_R2 as R


# --------------------------------------------------------------------------
def metrics(y_true, y_pred):
    y = np.asarray(y_true, float)
    p = np.asarray(y_pred, float)
    ss_res = float(((y - p) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    mask = y != 0
    return dict(
        R2=1.0 - ss_res / ss_tot,
        RMSE=float(np.sqrt(np.mean((y - p) ** 2))),
        MAE=float(np.mean(np.abs(y - p))),
        MAPE=float(np.mean(np.abs((y[mask] - p[mask]) / y[mask])) * 100.0),
    )


def cb_oof(df, feats, outer, lo=0.01, hi=0.99,
           target='Strength_28d', group_col='Concrete_Class'):
    """Fixed-parameter CatBoost OOF, no eval_set anywhere (es='none').

    A local copy of rerun_R2._cb_oof that also exposes the winsorisation
    bounds, which X2 needs to vary.
    """
    groups = df[group_col].values
    if outer == 'group':
        splits = list(R.GroupKFold(n_splits=min(5, df[group_col].nunique())
                                   ).split(df, groups=groups))
    else:
        splits = list(R.KFold(5, shuffle=True, random_state=R.SEED).split(df))

    y_oof = np.zeros(len(df))
    for tr_idx, va_idx in splits:
        tr, va, _ = R.preprocess_fold(df.iloc[tr_idx], df.iloc[va_idx],
                                      lo=lo, hi=hi)
        avail = [f for f in feats if f in tr.columns]
        Xtr = tr[avail].values.astype(np.float32)
        ytr = tr[target].values.astype(np.float32)
        Xva = va[avail].values.astype(np.float32)
        m = R.CatBoostRegressor(**R.CB_ABLATION_PARAMS,
                                random_seed=R.SEED, verbose=0)
        m.fit(Xtr, ytr, verbose=0)          # no eval_set -- this is the point
        y_oof[va_idx] = m.predict(Xva)
    return y_oof


# --------------------------------------------------------------------------
def run_X1(df):
    """Table 6: anchor ablation, full metric set, corrected protocol."""
    out = os.path.join(R.RES_DIR, 'X1_table6_full_metrics.csv')
    if os.path.exists(out):
        R.log('SKIP X1 (already done)'); return

    y = df['Strength_28d'].values
    full = list(R.FEAT_28D)
    noanchor = [f for f in full if f != 'Strength_7d']

    rows = []
    for cfg, feats in [('Full (30 features)', full),
                       ("No f'c,7 (29 feat.)", noanchor)]:
        for outer, label in [('standard', 'Standard 5-fold'),
                             ('group', 'GroupKFold')]:
            R.log(f'X1  {cfg:20s} {label:16s} ({len(feats)} features)')
            oof = cb_oof(df, feats, outer)
            rows.append(dict(configuration=cfg, scheme=label,
                             n_features=len(feats), **metrics(y, oof)))

    d = pd.DataFrame(rows)
    # deltas against the matching full-feature row
    for label in d['scheme'].unique():
        base = d[(d.scheme == label) &
                 (d.configuration == 'Full (30 features)')]['R2'].iloc[0]
        d.loc[d.scheme == label, 'dR2_vs_full'] = d.loc[d.scheme == label, 'R2'] - base
    d.to_csv(out, index=False)
    print('\n' + d.to_string(index=False))
    R.log(f'DONE  X1 -> {out}')


def run_X2(df):
    """Table S1: winsorisation sensitivity, corrected protocol."""
    out = os.path.join(R.RES_DIR, 'X2_tableS1_winsorisation.csv')
    if os.path.exists(out):
        R.log('SKIP X2 (already done)'); return

    y = df['Strength_28d'].values
    feats = list(R.FEAT_28D)
    rows = []
    for lo, hi in [(0.005, 0.995), (0.01, 0.99), (0.02, 0.98)]:
        rec = dict(lo=lo, hi=hi)
        for outer, key in [('standard', 'standard'), ('group', 'group')]:
            R.log(f'X2  winsor {lo:.3f}/{hi:.3f}  {key}')
            m = metrics(y, cb_oof(df, feats, outer, lo=lo, hi=hi))
            rec[f'R2_{key}'] = m['R2']
            rec[f'RMSE_{key}'] = m['RMSE']
        rows.append(rec)

    d = pd.DataFrame(rows)
    d.to_csv(out, index=False)
    print('\n' + d.to_string(index=False))
    for key in ('standard', 'group'):
        v = d[f'R2_{key}']
        print(f'  spread across settings ({key:8s}): '
              f'{v.max() - v.min():.5f}   [{v.min():.4f}, {v.max():.4f}]')
    R.log(f'DONE  X2 -> {out}')


def run_X3(df):
    """Figure 5: bootstrap replicates, corrected protocol.

    run_DIAG kept only the median and the 2.5/97.5 percentiles of R2, so the
    figure's three panels (R2, RMSE, MAE) cannot be redrawn from it. This
    repeats the identical resampling -- same generator, same seed schedule,
    same OOB rule -- and stores every replicate.
    """
    out = os.path.join(R.RES_DIR, 'X3_bootstrap_replicates.csv')
    if os.path.exists(out):
        R.log('SKIP X3 (already done)'); return

    feats = list(R.FEAT_28D)
    rng = np.random.default_rng(R.SEED)
    n = len(df)
    rows = []
    for b in range(200):
        idx = rng.integers(0, n, size=n)
        oob = np.setdiff1d(np.arange(n), np.unique(idx))
        if len(oob) < 20:
            continue
        tr, va, _ = R.preprocess_fold(df.iloc[idx], df.iloc[oob])
        avail = [f for f in feats if f in tr.columns]
        Xtr = tr[avail].values.astype(np.float32)
        ytr = tr['Strength_28d'].values.astype(np.float32)
        Xva = va[avail].values.astype(np.float32)
        yva = va['Strength_28d'].values.astype(np.float32)
        m = R.CatBoostRegressor(**R.CB_ABLATION_PARAMS,
                                random_seed=R.SEED + b, verbose=0)
        m.fit(Xtr, ytr, verbose=0)              # no eval_set
        rows.append(dict(replicate=b, n_oob=len(oob),
                         **metrics(yva, m.predict(Xva))))
        if (b + 1) % 50 == 0:
            R.log(f'X3  {b + 1}/200 replicates')

    d = pd.DataFrame(rows)
    d.to_csv(out, index=False)
    for c in ('R2', 'RMSE', 'MAE'):
        v = d[c]
        print(f'  {c:5s} median {v.median():.4f}  '
              f'95% CI [{v.quantile(.025):.4f}, {v.quantile(.975):.4f}]')
    R.log(f'DONE  X3 ({len(d)} replicates) -> {out}')


def run_X4(df):
    """Figure 4: per-sample TreeSHAP values under both schemes.

    run_SHAP averaged |SHAP| down to one number per feature and discarded the
    matrix, so the beeswarm and the dependence panels cannot be redrawn. This
    repeats the same out-of-fold attribution and stores the signed values plus
    the feature values they attach to.
    """
    from catboost import Pool
    done = all(os.path.exists(os.path.join(R.RES_DIR, f'X4_shap_{s}.csv'))
               for s in ('standard', 'group'))
    if done:
        R.log('SKIP X4 (already done)'); return

    feats = list(R.FEAT_28D)
    groups = df['Concrete_Class'].values

    for scheme in ('standard', 'group'):
        if scheme == 'group':
            k = min(5, df['Concrete_Class'].nunique())
            splits = list(R.GroupKFold(n_splits=k).split(df, groups=groups))
        else:
            splits = list(R.KFold(5, shuffle=True, random_state=R.SEED).split(df))

        shap = np.zeros((len(df), len(feats)))
        xval = np.zeros((len(df), len(feats)))
        for tr_idx, va_idx in splits:
            tr, va, _ = R.preprocess_fold(df.iloc[tr_idx], df.iloc[va_idx])
            Xtr = tr[feats].values.astype(np.float32)
            ytr = tr['Strength_28d'].values.astype(np.float32)
            Xva = va[feats].values.astype(np.float32)
            m = R.CatBoostRegressor(**R.CB_ABLATION_PARAMS,
                                    random_seed=R.SEED, verbose=0)
            m.fit(Xtr, ytr, verbose=0)
            sv = m.get_feature_importance(Pool(Xva), type='ShapValues')
            shap[va_idx, :] = sv[:, :-1]        # drop the base-value column
            xval[va_idx, :] = Xva

        out = os.path.join(R.RES_DIR, f'X4_shap_{scheme}.csv')
        d = pd.concat([
            pd.DataFrame(shap, columns=[f'shap__{f}' for f in feats]),
            pd.DataFrame(xval, columns=[f'val__{f}' for f in feats]),
            pd.DataFrame({'y_true': df['Strength_28d'].values,
                          'Concrete_Class': df['Concrete_Class'].values}),
        ], axis=1)
        d.to_csv(out, index=False)
        R.log(f'DONE  X4 [{scheme}] {shap.shape} -> {out}')


def run_X5(df):
    """§4.5 / §4.10: the repeated class-to-fold assignment spread.

    The manuscript reports GroupKFold R2 = 0.816 +- 0.011 across 20 random
    assignments of the eight design classes to five folds. That figure came
    from the leaked run and there is no honest way to carry it over, so the
    experiment is repeated here under the corrected protocol.

    Each replicate permutes the eight classes and deals them round-robin into
    five folds, so every fold holds one or two whole classes and no class is
    ever split. Replicate 0 is not the canonical GroupKFold assignment; the
    canonical figure (0.8130) comes from S16_diagnostics.csv.
    """
    out = os.path.join(R.RES_DIR, 'X5_fold_assignment_spread.csv')
    if os.path.exists(out):
        R.log('SKIP X5 (already done)'); return

    y = df['Strength_28d'].values
    feats = list(R.FEAT_28D)
    classes = np.array(sorted(df['Concrete_Class'].unique()))
    rows = []

    for rep in range(20):
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
            avail = [f for f in feats if f in tr_d.columns]
            m = R.CatBoostRegressor(**R.CB_ABLATION_PARAMS,
                                    random_seed=R.SEED, verbose=0)
            m.fit(tr_d[avail].values.astype(np.float32),
                  tr_d['Strength_28d'].values.astype(np.float32), verbose=0)
            y_oof[va] = m.predict(va_d[avail].values.astype(np.float32))

        rows.append(dict(replicate=rep,
                         classes_per_fold=';'.join(
                             str(sorted(int(c) for c, f in fold_of.items() if f == k))
                             for k in range(5)),
                         **metrics(y, y_oof)))
        R.log(f'X5  replicate {rep + 1}/20  R2 = {rows[-1]["R2"]:+.4f}')

    d = pd.DataFrame(rows)
    d.to_csv(out, index=False)
    v = d['R2']
    print(f'\n  mean {v.mean():.4f}  SD {v.std(ddof=1):.4f}  '
          f'range [{v.min():.4f}, {v.max():.4f}]  (n = {len(d)})')
    R.log(f'DONE  X5 -> {out}')


def run_X6(df):
    """§4.8 / §5.4: fixed-parameter CatBoost out-of-fold predictions.

    §4.8 states that its screening statistics use "the GroupKFold out-of-fold
    predictions of the fixed-hyperparameter CatBoost of §4.4". run_DIAG fits
    exactly that model but keeps only the summary metrics, so the sensitivity,
    specificity, precision, the M45 signed error and the predicted attainment
    share cannot be recomputed from what is stored. This saves the predictions
    themselves, under both schemes.
    """
    out = os.path.join(R.RES_DIR, 'X6_cb_oof.csv')
    if os.path.exists(out):
        R.log('SKIP X6 (already done)'); return

    feats = list(R.FEAT_28D)
    cols = {'y_true': df['Strength_28d'].values,
            'Concrete_Class': df['Concrete_Class'].values}
    for outer in ('standard', 'group'):
        R.log(f'X6  fixed-parameter CatBoost OOF, {outer}')
        cols[f'y_cb_{outer}'] = cb_oof(df, feats, outer)

    d = pd.DataFrame(cols)
    d.to_csv(out, index=False)
    for outer in ('standard', 'group'):
        m = metrics(d.y_true, d[f'y_cb_{outer}'])
        print(f'  {outer:9s} R2 {m["R2"]:.4f}  RMSE {m["RMSE"]:.3f}')
    R.log(f'DONE  X6 -> {out}')


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', default='ALL', help='ALL | X1 | X2 | X3 | X4 | X5 | X6')
    a = ap.parse_args()

    df = R.load_data()
    R.log(f'Data: n={len(df)}, classes={df.Concrete_Class.nunique()}')
    assert len(df) == 683, f'EXPECTED 683 RECORDS, GOT {len(df)}'

    sel = a.jobs.strip().upper()
    for name, fn in [('X1', run_X1), ('X2', run_X2),
                     ('X3', run_X3), ('X4', run_X4), ('X5', run_X5), ('X6', run_X6)]:
        if sel in ('ALL', name):
            fn(df)


if __name__ == '__main__':
    main()
