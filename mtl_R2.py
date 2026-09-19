#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
 mtl_R2.py  --  MTL-PINN re-run for RINENG-D-26-11517R1
===============================================================================

 WHY THIS EXISTS
 ---------------
 The tree pipeline's early-stopping defect (D2) has an equivalent in the
 MTL-PINN, and a worse one. In concrete_strength_pipeline.py, fit_mtl_fold does:

     cb_es = callbacks.EarlyStopping(monitor='val_loss', patience=40,
                                     restore_best_weights=True)
     cb_rl = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                         patience=20)
     model.fit(Xtr, ytr, validation_data=(Xva, yva), callbacks=[cb_es, cb_rl])
     preds = model.predict(Xva)

 where (Xva, yva) is the OUTER VALIDATION FOLD. Three things are therefore
 decided by the data the model is then scored on:

   1. when training stops,
   2. the learning-rate schedule,
   3. WHICH PARAMETER STATE IS KEPT -- restore_best_weights=True selects the
      epoch that scored best on the evaluation set. That is model selection on
      the test set, and it has no counterpart in the tree pipeline.

 Correcting the tree ensemble but not this would make the §4.6 architecture
 comparison unfair in the network's favour: corrected trees against a leaked
 network. Reviewer 5's comment 1 is about exactly that comparison.

 THE FIX
 -------
 A neural network genuinely needs a validation signal to stop on, so unlike the
 tree case we cannot simply switch early stopping off. Instead the early-stopping
 split is carved out of the TRAINING fold:

   - outer scheme 'group'    -> GroupShuffleSplit on the training fold's design
                                classes, so whole classes are held out and the
                                inner split is group-aware like the outer one;
   - outer scheme 'standard' -> a random 15 % split of the training fold.

 The outer validation fold is never shown to the model during fitting.

 Feature and target scalers stay fitted on the whole training fold, exactly as
 the published code does -- the inner validation rows are still training-fold
 data, so that is not a leak.

 HOW TO RUN
 ----------
     python mtl_R2.py --list
     python mtl_R2.py --jobs GATE     # reproduce the published MTL-PINN numbers
     python mtl_R2.py --jobs FIXED    # the corrected runs
     python mtl_R2.py --jobs ALL

 RUN THE GATES FIRST. M1 must land near the published standard-CV R² of 0.7493
 for f'c,28 and M2 near the published GroupKFold 0.5302. Note that Table S11
 already records that this model's absolute values move with the TensorFlow
 version, so expect looser agreement here than the tree gates gave.
===============================================================================
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit, KFold,
                                     train_test_split)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

import tensorflow as tf
import keras
from tensorflow.keras import layers, Model, callbacks, optimizers, regularizers
from tensorflow.keras import backend as K

sys.dont_write_bytecode = True
import rerun_R2 as R          # shared data loading, preprocessing, feature sets

SEED = R.SEED

# Three runs of M4 at the identical seed and configuration returned f'c,28
# GroupKFold R2 of +0.0358, +0.1832 and +0.0040 -- an 18-percentage-point
# spread from nothing but nondeterministic op scheduling. tf.random.set_seed
# and K.clear_session() do not prevent that on their own. The manuscript's
# claim that "the pipeline is deterministic (fixed random seed 42)" is false
# for this model unless op determinism is switched on explicitly.
os.environ.setdefault('TF_DETERMINISTIC_OPS', '1')
os.environ.setdefault('TF_CUDNN_DETERMINISTIC', '1')
os.environ.setdefault('PYTHONHASHSEED', str(SEED))
try:
    tf.config.experimental.enable_op_determinism()
    _DETERMINISM = 'enabled'
except Exception as _e:                       # TensorFlow < 2.9
    _DETERMINISM = f'UNAVAILABLE ({type(_e).__name__}) -- results will vary run to run'

np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.get_logger().setLevel('ERROR')

RES_DIR = R.RES_DIR
os.makedirs(RES_DIR, exist_ok=True)

# --- verbatim from the published pipeline ------------------------------------
FEAT_MTL = R.FEAT_MTL                      # 26 features
TARGETS_MTL = ['Strength_28d', 'Strength_7d', 'Slump_30', 'Slump_90']
LOSS_WEIGHTS = {'Strength_28d': 1.0, 'Strength_7d': 0.5,
                'Slump_30': 0.3, 'Slump_90': 0.3}
PHYS_LAMBDA = 0.10
DROPOUT_RATE = 0.20
EPOCHS, BATCH = 300, 32

# published Table 7 / master_results_table.csv, for the gates
PUBLISHED = {
    ('standard', 'Strength_28d'): 0.749231, ('standard', 'Strength_7d'): 0.722372,
    ('standard', 'Slump_30'): 0.495173, ('standard', 'Slump_90'): 0.516303,
    ('group', 'Strength_28d'): 0.5302, ('group', 'Strength_7d'): 0.5407,
    ('group', 'Slump_30'): -0.1243, ('group', 'Slump_90'): -0.2391,
}



def _seed_everything(seed):
    """Reseed every generator Keras actually draws from.

    tf.random.set_seed() was the correct call under Keras 2 / tf.keras. Keras 3
    draws initializer and dropout randomness from its own global SeedGenerator,
    which tf.random.set_seed() does NOT reset -- so after the Keras 3 upgrade
    this code was seeding nothing that mattered, and repeated runs at a fixed
    seed differed by up to 18 percentage points of R2. keras.utils.set_random_seed
    reseeds Python, NumPy, TensorFlow and the Keras generator together; verified
    bit-identical across repeated fits on TensorFlow 2.21 / Keras 3.15.
    """
    try:
        import keras
        keras.utils.set_random_seed(int(seed))
    except Exception:
        import random
        random.seed(int(seed))
        np.random.seed(int(seed))
    tf.random.set_seed(int(seed))

def log(m):
    print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


class AbramsViolationLayer(layers.Layer):
    """relu(y_7d - y_28d): positive when f'c,7 > f'c,28 (Abrams violation)."""
    def call(self, inputs):
        out_7d, out_28d = inputs
        try:
            return keras.ops.relu(out_7d - out_28d)
        except AttributeError:          # older Keras without keras.ops
            return tf.nn.relu(out_7d - out_28d)

    def get_config(self):
        return super().get_config()


def build_mtl_pinn(n_features, dropout=DROPOUT_RATE, l2=1e-5):
    reg = regularizers.l2(l2)
    inp = layers.Input(shape=(n_features,), name='features')
    x = layers.Dense(128, activation='relu', kernel_regularizer=reg)(inp)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=reg)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(32, activation='relu')(x)

    def head(name):
        h = layers.Dense(16, activation='relu')(x)
        return layers.Dense(1, name=name)(h)

    out_28d, out_7d = head('Strength_28d'), head('Strength_7d')
    out_s30, out_s90 = head('Slump_30'), head('Slump_90')
    abrams = AbramsViolationLayer(name='abrams_viol')([out_7d, out_28d])

    model = Model(inp, [out_28d, out_7d, out_s30, out_s90, abrams], name='MTL_PINN')
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss={'Strength_28d': 'mse', 'Strength_7d': 'mse', 'Slump_30': 'mse',
              'Slump_90': 'mse', 'abrams_viol': 'mae'},
        loss_weights={**LOSS_WEIGHTS, 'abrams_viol': PHYS_LAMBDA})
    return model


RUN_SEED = SEED          # varied by the SPREAD job; everything else uses SEED


def _inner_split(n, groups_tr):
    """Early-stopping split taken from the TRAINING fold. Group-aware when the
    outer loop is: whole design classes are held out, never split across."""
    if groups_tr is not None and len(np.unique(groups_tr)) >= 3:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=RUN_SEED)
        return next(gss.split(np.zeros(n), groups=groups_tr))
    return train_test_split(np.arange(n), test_size=0.15, random_state=RUN_SEED)


def fit_mtl_fold(tr_proc, va_proc, es_mode, groups_tr=None):
    feats, targets = FEAT_MTL, TARGETS_MTL
    sx = StandardScaler().fit(tr_proc[feats].values)
    sy = {t: StandardScaler().fit(tr_proc[t].values.reshape(-1, 1)) for t in targets}

    Xtr = sx.transform(tr_proc[feats].values).astype(np.float32)
    Xva = sx.transform(va_proc[feats].values).astype(np.float32)
    ytr = {t: sy[t].transform(tr_proc[t].values.reshape(-1, 1)).ravel() for t in targets}
    yva = {t: sy[t].transform(va_proc[t].values.reshape(-1, 1)).ravel() for t in targets}
    ytr['abrams_viol'] = np.zeros(len(Xtr), dtype=np.float32)
    yva['abrams_viol'] = np.zeros(len(Xva), dtype=np.float32)

    K.clear_session()
    _seed_everything(RUN_SEED)
    model = build_mtl_pinn(len(feats))

    cb_es = callbacks.EarlyStopping(monitor='val_loss', patience=40,
                                    restore_best_weights=True, verbose=0)
    cb_rl = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                        patience=20, min_lr=1e-5, verbose=0)

    if es_mode == 'published':
        # reproduces the defect: stopping, LR schedule and weight restoration
        # are all driven by the fold the model is scored on
        val_data = (Xva, yva)
        fit_X, fit_y = Xtr, ytr
    else:
        itr, iva = _inner_split(len(Xtr), groups_tr)
        fit_X = Xtr[itr]
        fit_y = {k: v[itr] for k, v in ytr.items()}
        val_data = (Xtr[iva], {k: v[iva] for k, v in ytr.items()})

    model.fit(fit_X, fit_y, validation_data=val_data, epochs=EPOCHS,
              batch_size=BATCH, callbacks=[cb_es, cb_rl], verbose=0)

    preds = model.predict(Xva, verbose=0)
    return {t: sy[t].inverse_transform(preds[i].reshape(-1, 1)).ravel()
            for i, t in enumerate(targets)}


def run_mtl(df, outer, es_mode):
    groups = df['Concrete_Class'].values
    n = len(df)
    if outer == 'group':
        k = min(5, df['Concrete_Class'].nunique())
        splits = list(GroupKFold(n_splits=k).split(df, groups=groups))
    else:
        splits = list(KFold(5, shuffle=True, random_state=SEED).split(df))

    oof = {t: np.zeros(n) for t in TARGETS_MTL}
    for fi, (tr_idx, va_idx) in enumerate(splits, 1):
        t0 = time.time()
        tr, va, _ = R.preprocess_fold(df.iloc[tr_idx], df.iloc[va_idx])
        g_tr = groups[tr_idx] if outer == 'group' else None
        p = fit_mtl_fold(tr, va, es_mode, g_tr)
        for t in TARGETS_MTL:
            oof[t][va_idx] = p[t]
        log(f'    fold {fi}/{len(splits)} done in {time.time()-t0:.0f}s')

    res = {}
    for t in TARGETS_MTL:
        yt = df[t].values
        res[f'R2_{t}'] = r2_score(yt, oof[t])
        res[f'RMSE_{t}'] = float(np.sqrt(np.mean((yt - oof[t]) ** 2)))
        res[f'MAE_{t}'] = float(np.mean(np.abs(yt - oof[t])))
    return res, oof


JOBS = {
    'M1': dict(group='GATE', outer='standard', es='published',
               desc='REPRODUCE published MTL-PINN, standard CV (expect 0.7493)'),
    'M2': dict(group='GATE', outer='group', es='published',
               desc='REPRODUCE published MTL-PINN, GroupKFold (expect 0.5302)'),
    'M3': dict(group='FIXED', outer='standard', es='inner',
               desc='CORRECTED: early stopping on an inner split, standard CV'),
    'M4': dict(group='FIXED', outer='group', es='inner',
               desc='CORRECTED: group-aware inner split, GroupKFold'),
}
ORDER = ['M1', 'M2', 'M3', 'M4']


def path(j):
    return os.path.join(RES_DIR, f'{j}.csv')


def run_job(j, df):
    if os.path.exists(path(j)):
        log(f'SKIP {j} (already done) -- delete {j}.csv to force a re-run'); return
    s = JOBS[j]
    log(f'START {j}: {s["desc"]}')
    log(f'      outer={s["outer"]} early_stopping={s["es"]} n_feats={len(FEAT_MTL)}')
    t0 = time.time()
    res, oof = run_mtl(df, s['outer'], s['es'])
    res.update(jid=j, group=s['group'], outer=s['outer'], es=s['es'],
               n_features=len(FEAT_MTL), minutes=round((time.time() - t0) / 60, 1))
    pd.DataFrame([res]).to_csv(path(j), index=False)
    pd.DataFrame({**{f'true_{t}': df[t].values for t in TARGETS_MTL},
                  **{f'pred_{t}': oof[t] for t in TARGETS_MTL},
                  'Concrete_Class': df['Concrete_Class'].values}
                 ).to_csv(os.path.join(RES_DIR, f'{j}_oof.csv'), index=False)

    log(f'DONE  {j}  ({res["minutes"]} min)')
    for t in TARGETS_MTL:
        pub = PUBLISHED.get((s['outer'], t))
        extra = ''
        if s['group'] == 'GATE' and pub is not None:
            extra = f'   published {pub:+.4f}   delta {res[f"R2_{t}"]-pub:+.4f}'
        log(f'    {t:13s} R2 {res[f"R2_{t}"]:+.4f}{extra}')


def run_SPREAD(df, n_seeds=5):
    """The M2 gate reproduced the published GroupKFold figure badly (0.379 vs
    0.530) while the M1 standard-CV gate landed within 0.002. That asymmetry
    says the network's group-aware result is unstable, not that the replication
    is wrong -- so a single number from it should not go in a paper.

    This repeats the two GroupKFold configurations across several seeds and
    reports the spread, so §4.6 can quote a range rather than a point estimate.
    """
    global RUN_SEED
    out = os.path.join(RES_DIR, 'M_SPREAD.csv')
    if os.path.exists(out):
        log('SKIP SPREAD (already done)'); return
    rows, original = [], RUN_SEED
    try:
        for k in range(n_seeds):
            RUN_SEED = SEED + k
            for lab, es in [('leaked (as published)', 'published'),
                            ('corrected', 'inner')]:
                t0 = time.time()
                res, _ = run_mtl(df, 'group', es)
                rows.append(dict(seed=RUN_SEED, mode=lab, es=es,
                                 **{f'R2_{t}': res[f'R2_{t}'] for t in TARGETS_MTL}))
                log(f'   seed {RUN_SEED} {lab:22s} '
                    f'f\'c,28 R2 = {res["R2_Strength_28d"]:+.4f}  '
                    f'({time.time()-t0:.0f}s)')
    finally:
        RUN_SEED = original

    d = pd.DataFrame(rows)
    d.to_csv(out, index=False)
    print()
    for lab in d['mode'].unique():
        s = d[d['mode'] == lab]
        for t in TARGETS_MTL:
            v = s[f'R2_{t}']
            print(f'  {lab:22s} {t:13s} mean {v.mean():+.4f}  sd {v.std():.4f}  '
                  f'range [{v.min():+.4f}, {v.max():+.4f}]')
    log(f'DONE  SPREAD -> {out}')



def run_LAMBDA(df, lambdas=(0.0, 0.05, 0.10, 0.50)):
    """Table S11: physics-penalty weight sensitivity, correctly seeded.

    The published sweep gave GroupKFold R2 of 0.44 / 0.45 / 0.49 / 0.38 and
    §4.10 concluded that lambda = 0.10 is locally optimal. That 11-pp spread
    was smaller than the run-to-run variation the baseline had at the time, so
    the conclusion was unsupported. With the Keras 3 seeding fixed the runs are
    reproducible, so the sweep can be measured rather than withdrawn -- and
    whatever it shows is now a property of lambda rather than of the seeding.
    """
    global PHYS_LAMBDA
    out = os.path.join(RES_DIR, 'M_LAMBDA.csv')
    if os.path.exists(out):
        log('SKIP LAMBDA (already done)'); return

    original, rows = PHYS_LAMBDA, []
    try:
        for lam in lambdas:
            PHYS_LAMBDA = float(lam)
            log(f'LAMBDA  lambda = {lam:.2f}  (GroupKFold, corrected early stopping)')
            t0 = time.time()
            res, oof = run_mtl(df, 'group', 'inner')
            viol = float(np.mean(oof['Strength_7d'] > oof['Strength_28d']) * 100.0)
            rows.append(dict(phys_lambda=lam,
                             **{f'R2_{t}': res[f'R2_{t}'] for t in TARGETS_MTL},
                             pct_monotonicity_violations=viol,
                             minutes=round((time.time() - t0) / 60, 1)))
            log(f'        f\'c,28 R2 = {res["R2_Strength_28d"]:+.4f}   '
                f'violations {viol:.1f} %')
    finally:
        PHYS_LAMBDA = original

    d = pd.DataFrame(rows)
    d.to_csv(out, index=False)
    print('\n' + d.to_string(index=False))
    v = d['R2_Strength_28d']
    print(f'\n  spread across lambda: {v.max() - v.min():.4f} '
          f'[{v.min():+.4f}, {v.max():+.4f}]   best at lambda = '
          f'{d.loc[v.idxmax(), "phys_lambda"]:.2f}')
    log(f'DONE  LAMBDA -> {out}')


def main():
    global RUN_SEED
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', default='GATE',
                    help='ALL | GATE | FIXED | SPREAD (repeat the GroupKFold '
                         'runs over several seeds) | LAMBDA | M1,M2,...')
    ap.add_argument('--seeds', type=int, default=5,
                    help='number of seeds for the SPREAD job (default 5)')
    ap.add_argument('--list', action='store_true')
    a = ap.parse_args()

    if a.list:
        for j in ORDER:
            print(f'{j:4s} {JOBS[j]["group"]:6s} '
                  f'{"yes" if os.path.exists(path(j)) else "no":4s} {JOBS[j]["desc"]}')
        return

    df = R.load_data()
    log(f'TensorFlow {getattr(tf, "__version__", "?")} | '
        f'op determinism: {_DETERMINISM}')
    log(f'Data: n={len(df)}, classes={df.Concrete_Class.nunique()}')
    assert len(df) == 683, f'EXPECTED 683 RECORDS, GOT {len(df)}'

    sel = a.jobs.strip()
    if sel == 'SPREAD':
        run_SPREAD(df, a.seeds); return
    if sel == 'LAMBDA':
        run_LAMBDA(df); return
    ids = (ORDER if sel == 'ALL'
           else [j for j in ORDER if JOBS[j]['group'] == sel] if sel in ('GATE', 'FIXED')
           else [x.strip() for x in sel.split(',') if x.strip()])
    for j in ids:
        if j in JOBS:
            run_job(j, df)
        else:
            log(f'unknown job {j}')

    parts = [pd.read_csv(path(j)) for j in ORDER if os.path.exists(path(j))]
    if parts:
        s = pd.concat(parts, ignore_index=True)
        cols = ['jid', 'group', 'outer', 'es'] + [f'R2_{t}' for t in TARGETS_MTL] + ['minutes']
        s = s[[c for c in cols if c in s.columns]]
        p = os.path.join(RES_DIR, 'SUMMARY_MTL.csv')
        s.to_csv(p, index=False)
        print('\n' + s.to_string(index=False))
        log(f'Summary -> {p}')


if __name__ == '__main__':
    main()
