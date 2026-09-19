#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
 rerun_R2.py  --  Revision-2 re-analysis for RINENG-D-26-11517R1
===============================================================================

 WHAT THIS IS
 ------------
 A standalone re-run harness that answers Reviewer 3's methodological items
 using the SAME data, preprocessing, learners and search spaces as
 concrete_strength_pipeline.py.  Every function below marked "[VERBATIM]" is
 copied unchanged from that file so the numbers are comparable.

 It does NOT modify concrete_strength_pipeline.py and does NOT overwrite any
 existing output.  Everything it writes goes into ./rerun_R2_out/ .

 WHY IT EXISTS
 -------------
 Three defects were found in the published pipeline:

   (D1)  R3-2.  inner_hpo() uses KFold(shuffle=True) for the inner tuning loop,
         and run_outer_fold() uses plain KFold for the inner OOF that fits the
         ensemble weights -- even when the OUTER loop is GroupKFold.  Group
         separation is therefore not preserved during tuning.

   (D2)  NOT RAISED BY ANY REVIEWER -- found on 2026-09-11.  In
         run_outer_fold(), the final per-fold base models are fitted with
             fit_model(m, mname, Xtr, ytr, Xva, yva)
         where (Xva, yva) is the OUTER VALIDATION FOLD.  fit_model() passes
         that as eval_set and early-stops LightGBM (50 rounds) and CatBoost
         (50 rounds) on it.  The number of boosting rounds is therefore chosen
         using the very data the model is then scored on.  This inflates BOTH
         the standard-CV and the GroupKFold figures.

   (D3)  Table S12 (encoding, grouping variable, class-feature ablation, SMD)
         and Tables S3-S11 are not generated anywhere in the released pipeline,
         which the Data and Code Availability statement says "reproduces every
         table and figure in this paper without modification."  Block S12 below
         regenerates the Table S12 rows so that claim becomes true.

 HOW TO RUN
 ----------
     python rerun_R2.py --list            # show all jobs, mark which are done
     python rerun_R2.py --jobs GATE       # run the two reproduction gates FIRST
     python rerun_R2.py --jobs S12        # fast block (~minutes)
     python rerun_R2.py --jobs R3_2       # the R3-2 correction
     python rerun_R2.py --jobs ALL        # everything, in priority order
     python rerun_R2.py --jobs A1,B1,C1   # named jobs

 Resumable: each job writes its own CSV and is skipped if that CSV exists.
 Kill it any time; re-run the same command and it picks up where it stopped.
 Per-fold out-of-fold predictions are dumped alongside each result so any
 metric can be recomputed later without re-fitting.

 RUN THE GATES FIRST.  If A1/A2 do not reproduce the published 0.8527 / 0.9251,
 stop and tell Claude -- nothing downstream is trustworthy until they do.
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
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold

warnings.filterwarnings('ignore')

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import optuna
from optuna.samplers import TPESampler

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ----------------------------------------------------------------- config
SEED = 42
np.random.seed(SEED)

N_OUTER_FOLDS = 5
N_INNER_FOLDS = 5
N_HPO_TRIALS = 50          # matches the published pipeline

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(OUT_DIR, 'Dhaka-Sylhet_mix_design.xlsx')
RES_DIR = os.path.join(OUT_DIR, 'rerun_R2_out')
os.makedirs(RES_DIR, exist_ok=True)

CB_ABLATION_PARAMS = dict(iterations=800, depth=6, learning_rate=0.05,
                          l2_leaf_reg=1.0, subsample=0.8, min_data_in_leaf=5)

# Powers' gel/space hydration degree. 0.75 is the published value; run_ALPHA()
# sweeps it for Reviewer 3 item 6.
ALPHA_28 = 0.75

MODEL_NAMES = ['XGBoost', 'LightGBM', 'CatBoost', 'ExtraTrees']


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


# ============================================================== [VERBATIM]
# S2. DATA LOADING & QUALITY CONTROL  (copied from concrete_strength_pipeline.py)
# =========================================================================
def load_data():
    df_raw = pd.read_excel(DATA_FILE, header=2)
    df_raw.columns = [
        'SN', 'Concrete_Class', 'Cement', 'Admixture', 'CA', 'FA',
        'Cement_Content', 'WC_Ratio', 'Admixture_Dose',
        'Strength_7d', 'Strength_28d', 'Slump_30', 'Slump_60', 'Slump_90',
    ]
    for col in ['CA', 'FA', 'Cement_Content', 'WC_Ratio', 'Admixture_Dose',
                'Strength_7d', 'Strength_28d', 'Slump_30', 'Slump_60',
                'Slump_90', 'Concrete_Class']:
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
    df_raw['Cement'] = df_raw['Cement'].astype(str).str.strip().str.title()
    df_raw['Admixture'] = df_raw['Admixture'].astype(str).str.strip().str.title()

    CRITICAL_COLS = ['Concrete_Class', 'Cement', 'Admixture', 'Cement_Content',
                     'WC_Ratio', 'Admixture_Dose', 'Strength_7d',
                     'Strength_28d', 'Slump_30', 'Slump_90']
    df_raw = df_raw.dropna(subset=CRITICAL_COLS).reset_index(drop=True)
    df_raw = df_raw[df_raw['Strength_28d'] >= df_raw['Strength_7d']].reset_index(drop=True)
    DUP_COLS = ['Cement', 'Admixture', 'CA', 'FA', 'Cement_Content', 'WC_Ratio',
                'Admixture_Dose', 'Strength_7d', 'Strength_28d',
                'Slump_30', 'Slump_60', 'Slump_90']
    df_raw = df_raw.drop_duplicates(subset=DUP_COLS, keep='first').reset_index(drop=True)
    return df_raw


# ============================================================== [VERBATIM]
# S3. FEATURE DEFINITIONS AND FOLD-LOCAL PREPROCESSING
# =========================================================================
WINSOR_COLS = ['Cement_Content', 'WC_Ratio', 'Admixture_Dose', 'CA', 'FA',
               'Strength_7d', 'Slump_30', 'Slump_60', 'Slump_90']

FEAT_28D = [
    'Concrete_Class', 'CA', 'FA', 'Cement_Content', 'WC_Ratio',
    'Admixture_Dose', 'Strength_7d', 'Slump_30',
    'Water_Content', 'Paste_Volume', 'CA_FA_Ratio', 'Agg_Volume_Frac',
    'Admix_pct_bwoc', 'Admix_per_Cement',
    'WC_Ratio_sq', 'WC_Ratio_inv', 'log_Cement', 'Cement_x_WC',
    'Bolomey_Feature',
    'Concrete_Class_sq', 'Class_x_WC', 'Class_x_Cement', 'Cement_sq',
    'CA_FA_x_WC',
    'Binder_Intensity', 'Gel_Space_Ratio_28d',
    'Slump_Retention', 'Slump_Loss_Rate',
    'Cement_Freq', 'Admix_Freq',
]
FEAT_SLUMP = [f for f in FEAT_28D if f not in
              {'Strength_7d', 'Slump_30', 'Slump_Retention',
               'Slump_Loss_Rate', 'Gel_Space_Ratio_28d'}]
# The MTL-PINN's own 26-feature input set (pipeline line ~686) -- used for the
# matched-input comparison Reviewer 5 asks for.
FEAT_MTL = [f for f in FEAT_28D if f not in
            {'Strength_7d', 'Slump_30', 'Slump_Retention', 'Slump_Loss_Rate'}]

assert len(FEAT_28D) == 30
assert len(FEAT_SLUMP) == 25
assert len(FEAT_MTL) == 26

# Reviewer 3 item 1: the per-target feature matrices, stated explicitly.
TARGETS_CONFIG = {
    'Strength_28d': FEAT_28D,
    'Strength_7d': [f for f in FEAT_28D if f != 'Strength_7d'],
    'Slump_30': FEAT_SLUMP,
    'Slump_90': FEAT_SLUMP,
}


def fit_winsor_bounds(train_df, cols=WINSOR_COLS, lo=0.01, hi=0.99):
    bounds = {}
    for c in cols:
        if c in train_df:
            bounds[c] = (train_df[c].quantile(lo), train_df[c].quantile(hi))
    return bounds


def apply_winsor_bounds(df, bounds):
    d = df.copy()
    for c, (l, h) in bounds.items():
        d[c] = d[c].clip(l, h)
    return d


def engineer_features(df):
    d = df.copy()
    d['Water_Content'] = d['Cement_Content'] * d['WC_Ratio']
    d['Paste_Volume'] = (d['Cement_Content'] / 3150 + d['Water_Content'] / 1000) * 1000
    d['CA_FA_Ratio'] = d['CA'] / d['FA']
    d['Admix_pct_bwoc'] = d['Admixture_Dose'] / 100
    d['Admix_per_Cement'] = d['Admixture_Dose'] / d['Cement_Content'] * 1000
    d['Agg_Volume_Frac'] = (d['CA'] + d['FA']) / 100
    d['WC_Ratio_sq'] = d['WC_Ratio'] ** 2
    d['WC_Ratio_inv'] = 1.0 / d['WC_Ratio']
    d['log_Cement'] = np.log(d['Cement_Content'])
    d['Cement_x_WC'] = d['Cement_Content'] * d['WC_Ratio']
    d['Bolomey_Feature'] = (1.0 / d['WC_Ratio']) - 0.5
    d['Concrete_Class_sq'] = d['Concrete_Class'] ** 2
    d['Class_x_WC'] = d['Concrete_Class'] * d['WC_Ratio']
    d['Class_x_Cement'] = d['Concrete_Class'] * d['Cement_Content']
    d['Cement_sq'] = d['Cement_Content'] ** 2
    d['CA_FA_x_WC'] = d['CA_FA_Ratio'] * d['WC_Ratio']
    d['Binder_Intensity'] = d['Cement_Content'] / d['Concrete_Class']
    a = ALPHA_28             # R3-6: fixed hydration degree; see run_ALPHA()
    d['Gel_Space_Ratio_28d'] = (0.68 * a) / (0.32 * a + d['WC_Ratio'])
    d['Slump_Retention'] = d['Slump_90'] / d['Slump_30']
    d['Slump_Loss_Rate'] = (d['Slump_30'] - d['Slump_90']) / 60.0
    return d


def add_fold_freq(train_df, valid_df, brand='Cement', admix='Admixture'):
    fb = train_df[brand].value_counts(normalize=True)
    fa = train_df[admix].value_counts(normalize=True)

    def _apply(df_):
        df_ = df_.copy()
        df_['Cement_Freq'] = df_[brand].map(fb).fillna(0.0).values
        df_['Admix_Freq'] = df_[admix].map(fa).fillna(0.0).values
        return df_
    return _apply(train_df), _apply(valid_df)


def add_fold_onehot(train_df, valid_df, brand='Cement', admix='Admixture'):
    """R3-5 arm. Fold-local one-hot: categories fitted on the TRAINING fold only;
    a category unseen in training yields an all-zero row. Returns the dataframes
    plus the generated column names so the caller can swap them in for
    Cement_Freq / Admix_Freq."""
    cats_b = sorted(train_df[brand].unique())
    cats_a = sorted(train_df[admix].unique())
    cols = ([f'OH_B_{c}' for c in cats_b] + [f'OH_A_{c}' for c in cats_a])

    def _apply(df_):
        df_ = df_.copy()
        for c in cats_b:
            df_[f'OH_B_{c}'] = (df_[brand] == c).astype(float).values
        for c in cats_a:
            df_[f'OH_A_{c}'] = (df_[admix] == c).astype(float).values
        return df_
    return _apply(train_df), _apply(valid_df), cols


def preprocess_fold(train_df, valid_df, lo=0.01, hi=0.99, encoding='freq'):
    bounds = fit_winsor_bounds(train_df, lo=lo, hi=hi)
    tr = apply_winsor_bounds(train_df, bounds)
    va = apply_winsor_bounds(valid_df, bounds)
    tr = engineer_features(tr)
    va = engineer_features(va)
    if encoding == 'freq':
        tr, va = add_fold_freq(tr, va)
        return tr, va, None
    elif encoding == 'onehot':
        tr, va = add_fold_freq(tr, va)      # keep cols present, then replace
        tr, va, oh_cols = add_fold_onehot(tr, va)
        return tr, va, oh_cols
    elif encoding == 'none':
        tr, va = add_fold_freq(tr, va)
        return tr, va, []
    raise ValueError(encoding)


def resolve_features(base_feats, encoding, oh_cols):
    """Swap the two frequency columns for the chosen categorical representation."""
    feats = [f for f in base_feats if f not in ('Cement_Freq', 'Admix_Freq')]
    if encoding == 'freq':
        return base_feats
    if encoding == 'onehot':
        return feats + list(oh_cols)
    if encoding == 'none':
        return feats
    raise ValueError(encoding)


# ============================================================== [VERBATIM]
# S4. MODELS, HPO, ENSEMBLE  (search spaces copied unchanged)
# =========================================================================
def make_base_models(xgb_p, lgb_p, cb_p):
    return {
        'XGBoost': xgb.XGBRegressor(**xgb_p, random_state=SEED, n_jobs=-1, verbosity=0),
        'LightGBM': lgb.LGBMRegressor(**lgb_p, random_state=SEED, n_jobs=-1, verbose=-1),
        'CatBoost': CatBoostRegressor(**cb_p, random_seed=SEED, verbose=0),
        'ExtraTrees': ExtraTreesRegressor(n_estimators=500, max_depth=20,
                                          min_samples_leaf=2, random_state=SEED,
                                          n_jobs=-1),
    }


def fit_model(model, name, Xtr, ytr, Xva, yva, es_mode='published'):
    """es_mode='published' reproduces the released pipeline EXACTLY, including
    defect D2 (early stopping on whatever fold is passed in as (Xva, yva)).
    es_mode='none' disables early stopping entirely and honours the tuned
    n_estimators / iterations instead -- no eval_set is ever shown to the model.
    """
    if es_mode == 'none':
        model.fit(Xtr, ytr)          # no eval_set is ever shown to the model
        return model
    if name == 'XGBoost':
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    elif name == 'LightGBM':
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    elif name == 'CatBoost':
        model.fit(Xtr, ytr, eval_set=(Xva, yva), early_stopping_rounds=50, verbose=0)
    else:
        model.fit(Xtr, ytr)
    return model


def make_inner_splits(X, groups, inner_mode):
    """THE R3-2 FIX.  inner_mode='kfold' reproduces the published pipeline;
    inner_mode='group' preserves group separation inside the tuning loop."""
    if inner_mode == 'kfold' or groups is None:
        return list(KFold(n_splits=N_INNER_FOLDS, shuffle=True,
                          random_state=SEED).split(X))
    k = min(N_INNER_FOLDS, len(np.unique(groups)))
    if k < 2:
        return list(KFold(n_splits=N_INNER_FOLDS, shuffle=True,
                          random_state=SEED).split(X))
    return list(GroupKFold(n_splits=k).split(X, groups=groups))


def inner_hpo(X_inner, y_inner, groups_inner, inner_mode, es_mode,
              n_trials=N_HPO_TRIALS):
    """Search spaces are VERBATIM from concrete_strength_pipeline.py."""
    splits = make_inner_splits(X_inner, groups_inner, inner_mode)

    def _cv_score(model_fn, model_name):
        scores = []
        for tr, va in splits:
            m = model_fn()
            m = fit_model(m, model_name, X_inner[tr], y_inner[tr],
                          X_inner[va], y_inner[va], es_mode=es_mode)
            scores.append(r2_score(y_inner[va], m.predict(X_inner[va])))
        return float(np.mean(scores))

    def obj_xgb(trial):
        p = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 1500),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.15, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.9),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-7, 5.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-7, 5.0, log=True),
        }
        kw = {} if es_mode == 'none' else {'early_stopping_rounds': 30}
        return _cv_score(lambda: xgb.XGBRegressor(**p, random_state=SEED, n_jobs=-1,
                                                  verbosity=0, **kw), 'XGBoost')

    def obj_lgb(trial):
        p = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 1500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.15, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.9),
            'min_child_samples': trial.suggest_int('min_child_samples', 3, 40),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-7, 5.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-7, 5.0, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
        }
        return _cv_score(lambda: lgb.LGBMRegressor(**p, random_state=SEED,
                                                   n_jobs=-1, verbose=-1), 'LightGBM')

    def obj_cb(trial):
        p = {
            'iterations': trial.suggest_int('iterations', 300, 1500),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.15, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.001, 10.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 30),
        }
        return _cv_score(lambda: CatBoostRegressor(**p, random_seed=SEED,
                                                   verbose=0), 'CatBoost')

    best_params = {}
    for name, obj in [('xgb', obj_xgb), ('lgb', obj_lgb), ('cb', obj_cb)]:
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=SEED))
        study.optimize(obj, n_trials=n_trials)
        best_params[name] = study.best_params
    return best_params


def optimise_ensemble_weights(oof_stack, y_true):
    def neg_r2(w):
        w = np.abs(w); w /= w.sum()
        return -r2_score(y_true, (oof_stack * w).sum(axis=1))
    n_models = oof_stack.shape[1]
    res = minimize(neg_r2, x0=np.ones(n_models) / n_models, method='Nelder-Mead')
    w = np.abs(res.x); w /= w.sum()
    return w


def run_outer_fold(df_fold, target, base_feats, tr_idx, va_idx,
                   groups_tr, inner_mode, es_mode, encoding):
    tr_raw, va_raw = df_fold.iloc[tr_idx], df_fold.iloc[va_idx]
    tr_proc, va_proc, oh_cols = preprocess_fold(tr_raw, va_raw, encoding=encoding)
    feat_list = resolve_features(base_feats, encoding, oh_cols)

    Xtr = tr_proc[feat_list].values.astype(np.float32)
    ytr = tr_proc[target].values.astype(np.float32)
    Xva = va_proc[feat_list].values.astype(np.float32)
    yva = va_proc[target].values.astype(np.float32)

    best_p = inner_hpo(Xtr, ytr, groups_tr, inner_mode, es_mode)

    splits = make_inner_splits(Xtr, groups_tr, inner_mode)
    inner_oof = np.zeros((len(ytr), len(MODEL_NAMES)))
    for itr, iva in splits:
        models_i = make_base_models(best_p['xgb'], best_p['lgb'], best_p['cb'])
        for mi, (mname, m) in enumerate(models_i.items()):
            m = fit_model(m, mname, Xtr[itr], ytr[itr], Xtr[iva], ytr[iva],
                          es_mode=es_mode)
            inner_oof[iva, mi] = m.predict(Xtr[iva])
    weights = optimise_ensemble_weights(inner_oof, ytr)

    models_final = make_base_models(best_p['xgb'], best_p['lgb'], best_p['cb'])
    va_preds = {}
    for mname, m in models_final.items():
        # es_mode='published' reproduces defect D2 (early stopping on the scored
        # fold). es_mode='none' never shows the model the validation fold.
        m = fit_model(m, mname, Xtr, ytr, Xva, yva, es_mode=es_mode)
        va_preds[mname] = m.predict(Xva)

    stack = np.column_stack([va_preds[n] for n in MODEL_NAMES])
    return yva, (stack * weights).sum(axis=1), va_preds, weights, best_p


def run_scheme(df, target, base_feats, outer, inner_mode, es_mode, encoding,
               group_col='Concrete_Class'):
    groups = df[group_col].values
    n = len(df)
    if outer == 'group':
        k = min(N_OUTER_FOLDS, df[group_col].nunique())
        outer_splits = list(GroupKFold(n_splits=k).split(df, groups=groups))
    else:
        outer_splits = list(KFold(n_splits=N_OUTER_FOLDS, shuffle=True,
                                  random_state=SEED).split(df))

    y_oof = np.zeros(n)
    per_model = {m: np.zeros(n) for m in MODEL_NAMES}
    all_w, all_params = [], []
    for fi, (tr_idx, va_idx) in enumerate(outer_splits, 1):
        t0 = time.time()
        g_tr = groups[tr_idx] if outer == 'group' else None
        _, ens, vp, w, best_p = run_outer_fold(df, target, base_feats, tr_idx,
                                               va_idx, g_tr, inner_mode, es_mode,
                                               encoding)
        y_oof[va_idx] = ens
        for m in MODEL_NAMES:
            per_model[m][va_idx] = vp[m]
        all_w.append(w)
        # Per-fold tuned hyperparameters. Needed to tell whether a cross-version
        # discrepancy comes from Optuna sampling a different trial sequence.
        all_params.append({'fold': fi, **{f'{k}.{kk}': vv
                                          for k, d in best_p.items()
                                          for kk, vv in d.items()}})
        log(f'    fold {fi}/{len(outer_splits)} done in {time.time()-t0:.0f}s')

    y_true = df[target].values
    out = {'R2': r2_score(y_true, y_oof),
           'RMSE': float(np.sqrt(np.mean((y_true - y_oof) ** 2))),
           'MAE': float(np.mean(np.abs(y_true - y_oof)))}
    for m in MODEL_NAMES:
        out[f'R2_{m}'] = r2_score(y_true, per_model[m])
    out['weights'] = json.dumps(np.mean(all_w, axis=0).round(4).tolist())
    return out, y_oof, per_model, pd.DataFrame(all_params)


# ============================================================== job registry
JOBS = {}


def job(jid, group, desc, **kw):
    JOBS[jid] = dict(jid=jid, group=group, desc=desc, **kw)


# -- Gates: must reproduce the published headline before anything else counts.
job('A1', 'GATE', 'REPRODUCE published GroupKFold f\'c,28 (expect R2 = 0.8527)',
    target='Strength_28d', outer='group', inner='kfold', es='published', enc='freq')
job('A2', 'GATE', 'REPRODUCE published standard-CV f\'c,28 (expect R2 = 0.9251)',
    target='Strength_28d', outer='standard', inner='kfold', es='published', enc='freq')

# -- R3-2 alone: fix the inner splitter, leave everything else as published.
job('B1', 'R3_2', 'R3-2 fix: inner GroupKFold, f\'c,28 GroupKFold',
    target='Strength_28d', outer='group', inner='group', es='published', enc='freq')

# -- D2 (early-stopping leak) alone, and both fixes together.
job('C1', 'ESLEAK', 'D2 fix only: no early-stop leak, f\'c,28 GroupKFold',
    target='Strength_28d', outer='group', inner='kfold', es='none', enc='freq')
job('C2', 'ESLEAK', 'D2 fix only: no early-stop leak, f\'c,28 standard CV',
    target='Strength_28d', outer='standard', inner='kfold', es='none', enc='freq')
job('C3', 'ESLEAK', 'BOTH fixes: inner GroupKFold + no early-stop leak, GroupKFold',
    target='Strength_28d', outer='group', inner='group', es='none', enc='freq')

# -- R5-1: the ensemble on the MTL-PINN's own 26-feature set (matched comparison).
job('R5a', 'R5_1', 'R5-1 matched inputs: ensemble on FEAT_MTL (26), standard CV',
    target='Strength_28d', outer='standard', inner='kfold', es='none', enc='freq',
    feats='MTL')
job('R5b', 'R5_1', 'R5-1 matched inputs: ensemble on FEAT_MTL (26), GroupKFold',
    target='Strength_28d', outer='group', inner='group', es='none', enc='freq',
    feats='MTL')

# -- Remaining three targets under the corrected protocol (for Table 3).
job('E1', 'TARGETS', 'f\'c,7 GroupKFold, both fixes',
    target='Strength_7d', outer='group', inner='group', es='none', enc='freq')
job('E2', 'TARGETS', 'Slump_30 GroupKFold, both fixes',
    target='Slump_30', outer='group', inner='group', es='none', enc='freq')
job('E3', 'TARGETS', 'Slump_90 GroupKFold, both fixes',
    target='Slump_90', outer='group', inner='group', es='none', enc='freq')
job('E4', 'TARGETS', 'f\'c,7 standard CV, both fixes',
    target='Strength_7d', outer='standard', inner='kfold', es='none', enc='freq')
job('E5', 'TARGETS', 'Slump_30 standard CV, both fixes',
    target='Slump_30', outer='standard', inner='kfold', es='none', enc='freq')
job('E6', 'TARGETS', 'Slump_90 standard CV, both fixes',
    target='Slump_90', outer='standard', inner='kfold', es='none', enc='freq')

# -- R3-5: one-hot vs frequency, on the TUNED ENSEMBLE (not the CatBoost proxy).
job('F1', 'R3_5', 'R3-5: one-hot encoding, f\'c,28 GroupKFold, both fixes',
    target='Strength_28d', outer='group', inner='group', es='none', enc='onehot')
job('F2', 'R3_5', 'R3-5: one-hot encoding, f\'c,28 standard CV, both fixes',
    target='Strength_28d', outer='standard', inner='kfold', es='none', enc='onehot')

# -- R3-10: alternative grouping with the tuned ensemble, not the proxy.
job('G1', 'R3_10', 'R3-10: grouping by cement brand, tuned ensemble',
    target='Strength_28d', outer='group', inner='group', es='none', enc='freq',
    group_col='Cement')

PRIORITY = ['A1', 'A2', 'B1', 'C1', 'C2', 'C3', 'R5a', 'R5b',
            'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'F1', 'F2', 'G1']


def job_path(jid):
    return os.path.join(RES_DIR, f'{jid}.csv')


def run_job(jid, df):
    spec = JOBS[jid]
    if os.path.exists(job_path(jid)):
        log(f'SKIP {jid} (already done) -- delete {jid}.csv to force a re-run')
        return
    base = {'MTL': FEAT_MTL}.get(spec.get('feats'), TARGETS_CONFIG[spec['target']])
    gcol = spec.get('group_col', 'Concrete_Class')
    log(f'START {jid}: {spec["desc"]}')
    log(f'      target={spec["target"]} outer={spec["outer"]} inner={spec["inner"]} '
        f'es={spec["es"]} enc={spec["enc"]} n_feats={len(base)} group_col={gcol}')
    t0 = time.time()
    res, y_oof, per_model, params = run_scheme(df, spec['target'], base,
                                               spec['outer'], spec['inner'],
                                               spec['es'], spec['enc'],
                                               group_col=gcol)
    res.update(jid=jid, group=spec['group'], desc=spec['desc'],
               target=spec['target'], outer=spec['outer'], inner=spec['inner'],
               es=spec['es'], encoding=spec['enc'], group_col=gcol,
               n_features=len(base), minutes=round((time.time() - t0) / 60, 1))
    pd.DataFrame([res]).to_csv(job_path(jid), index=False)
    oof = pd.DataFrame({'y_true': df[spec['target']].values, 'y_ens': y_oof,
                        **{f'y_{m}': per_model[m] for m in MODEL_NAMES},
                        'Concrete_Class': df['Concrete_Class'].values})
    oof.to_csv(os.path.join(RES_DIR, f'{jid}_oof.csv'), index=False)
    params.to_csv(os.path.join(RES_DIR, f'{jid}_params.csv'), index=False)
    log(f'DONE  {jid}: R2={res["R2"]:.4f}  RMSE={res["RMSE"]:.3f}  '
        f'({res["minutes"]} min)')

    if jid == 'A1':
        _gate_check(res['R2'], 0.8527, 'A1 / published GroupKFold')
    if jid == 'A2':
        _gate_check(res['R2'], 0.9251, 'A2 / published standard CV')


def _gate_check(got, expected, label):
    d = abs(got - expected)
    if d <= 0.002:
        log(f'  GATE PASS  {label}: {got:.4f} vs published {expected:.4f}')
    else:
        log(f'  *** GATE FAIL *** {label}: {got:.4f} vs published {expected:.4f} '
            f'(delta {d:.4f}). Stop and report this before trusting anything else.')


# ===================================================== fast block: Table S12
def _proxy_r2(df, target, feats, outer, encoding='freq', group_col='Concrete_Class',
              drop=(), es='published'):
    """es='published' reproduces the released diagnostics exactly, including the
    D2 early-stopping-on-the-validation-fold pattern that quick_cv_r2 also uses.
    es='none' removes it, so the absolute values are clean rather than merely
    comparable."""
    groups = df[group_col].values
    n = len(df)
    if outer == 'group':
        k = min(5, df[group_col].nunique())
        splits = list(GroupKFold(n_splits=k).split(df, groups=groups))
    else:
        splits = list(KFold(5, shuffle=True, random_state=SEED).split(df))
    y_oof = np.zeros(n)
    for tr_idx, va_idx in splits:
        tr, va, oh = preprocess_fold(df.iloc[tr_idx], df.iloc[va_idx], encoding=encoding)
        fl = [f for f in resolve_features(feats, encoding, oh) if f not in drop]
        Xtr = tr[fl].values.astype(np.float32); ytr = tr[target].values.astype(np.float32)
        Xva = va[fl].values.astype(np.float32); yva = va[target].values.astype(np.float32)
        m = CatBoostRegressor(**CB_ABLATION_PARAMS, random_seed=SEED, verbose=0)
        if es == 'none':
            m.fit(Xtr, ytr, verbose=0)
        else:
            m.fit(Xtr, ytr, eval_set=(Xva, yva), early_stopping_rounds=50, verbose=0)
        y_oof[va_idx] = m.predict(Xva)
    return r2_score(df[target].values, y_oof)


def run_S12(df):
    """Regenerates the Table S12 rows, which the released pipeline does not
    produce (defect D3). CatBoost proxy, as published."""
    out = os.path.join(RES_DIR, 'S12_TableS12.csv')
    if os.path.exists(out):
        log('SKIP S12 (already done)'); return
    rows = []
    cls_feats = ('Concrete_Class', 'Concrete_Class_sq', 'Class_x_WC',
                 'Class_x_Cement', 'Binder_Intensity')

    # Every diagnostic is computed twice: once reproducing the released code
    # (es='published', which carries the D2 early-stopping pattern) and once
    # with that removed. The comparisons were always internally valid because
    # the pattern applied equally to every arm; running both shows by how much
    # the absolute values move.
    for es in ('published', 'none'):
        log(f'S12 [es={es}]: categorical encoding')
        for enc, lab in [('freq', 'Frequency (canonical)'), ('onehot', 'One-hot'),
                         ('none', 'No categoricals')]:
            rows.append(dict(block='encoding', variant=lab, es=es,
                             R2_standard=_proxy_r2(df, 'Strength_28d', FEAT_28D,
                                                   'standard', enc, es=es),
                             R2_group=_proxy_r2(df, 'Strength_28d', FEAT_28D,
                                                'group', enc, es=es)))
            log(f'   {lab}: {rows[-1]["R2_standard"]:.4f} / {rows[-1]["R2_group"]:.4f}')

        log(f'S12 [es={es}]: grouping variable')
        rows.append(dict(block='grouping', variant='Random folds (no grouping)', es=es,
                         R2_standard=_proxy_r2(df, 'Strength_28d', FEAT_28D,
                                               'standard', es=es),
                         R2_group=np.nan))
        for gc, lab in [('Concrete_Class', 'Design class (8)'),
                        ('Cement', 'Cement brand (10)'),
                        ('Admixture', 'Admixture type (14)')]:
            rows.append(dict(block='grouping', variant=lab, es=es, R2_standard=np.nan,
                             R2_group=_proxy_r2(df, 'Strength_28d', FEAT_28D, 'group',
                                                group_col=gc, es=es)))
            log(f'   {lab}: {rows[-1]["R2_group"]:.4f}')

        log(f'S12 [es={es}]: class-feature ablation')
        rows.append(dict(block='class_ablation', variant='All class features removed',
                         es=es,
                         R2_standard=_proxy_r2(df, 'Strength_28d', FEAT_28D, 'standard',
                                               drop=cls_feats, es=es),
                         R2_group=_proxy_r2(df, 'Strength_28d', FEAT_28D, 'group',
                                            drop=cls_feats, es=es)))

    log('S12: covariate overlap (SMD)')
    key = ['Cement_Content', 'WC_Ratio', 'CA', 'FA', 'Admixture_Dose']
    for outer, lab in [('standard', 'Random folds'), ('group', 'GroupKFold by class')]:
        g = df['Concrete_Class'].values
        splits = (list(GroupKFold(n_splits=5).split(df, groups=g)) if outer == 'group'
                  else list(KFold(5, shuffle=True, random_state=SEED).split(df)))
        smds = []
        for tr, va in splits:
            for c in key:
                a, b = df[c].iloc[tr], df[c].iloc[va]
                sd = np.sqrt((a.var() + b.var()) / 2)
                if sd > 0:
                    smds.append(abs(a.mean() - b.mean()) / sd)
        rows.append(dict(block='SMD', variant=lab, mean_SMD=float(np.mean(smds)),
                         max_SMD=float(np.max(smds))))
        log(f'   {lab}: mean SMD {np.mean(smds):.3f}, max {np.max(smds):.3f}')

    pd.DataFrame(rows).to_csv(out, index=False)
    log(f'DONE  S12 -> {out}')


def run_ALPHA(df):
    """R3-6: sensitivity of the headline to Powers' fixed hydration degree.

    EXPECTED RESULT: R2 identical to every decimal place at every alpha.
    That is not a bug. For any alpha > 0,

        Gel_Space_Ratio_28d(w) = 0.68*alpha / (0.32*alpha + w),
        d/dw = -0.68*alpha / (0.32*alpha + w)^2  <  0,

    so the feature is a STRICTLY MONOTONE DECREASING function of W/C whatever
    alpha is chosen. Changing alpha re-scales the feature but never reorders
    the samples, and tree ensembles split on rank/threshold only -- they are
    invariant under monotone transforms of a single feature. The choice of
    alpha therefore cannot affect any prediction from this ensemble; it affects
    only the numerical scale on which the feature is reported.

    CAVEAT for the response letter: this invariance covers the tree ensemble
    only. Gel_Space_Ratio_28d is also in FEAT_MTL, and a neural network is NOT
    invariant to a monotone nonlinear rescaling of its inputs, so the MTL-PINN
    would in principle respond to alpha. That is worth stating rather than
    glossing.
    """
    global ALPHA_28
    out = os.path.join(RES_DIR, 'S13_alpha_sensitivity.csv')
    if os.path.exists(out):
        log('SKIP ALPHA (already done)'); return
    original, rows = ALPHA_28, []
    try:
        for a in [0.60, 0.70, 0.75, 0.80, 0.90]:
            ALPHA_28 = a
            r_std = _proxy_r2(df, 'Strength_28d', FEAT_28D, 'standard')
            r_grp = _proxy_r2(df, 'Strength_28d', FEAT_28D, 'group')
            rows.append(dict(alpha=a, R2_standard=r_std, R2_group=r_grp,
                             published=(a == 0.75)))
            log(f'   alpha={a}: {r_std:.4f} / {r_grp:.4f}')
    finally:
        ALPHA_28 = original
    pd.DataFrame(rows).to_csv(out, index=False)
    log(f'DONE  ALPHA -> {out}')


# ============================================== R3-4: TreeSHAP under GroupKFold
def run_SHAP(df):
    """Reviewer 3 item 4. The published attributions were computed on standard
    5-fold splits only, which is inconsistent with treating GroupKFold as the
    deployment-relevant protocol. This recomputes out-of-fold TreeSHAP under
    BOTH schemes on the same fixed-hyperparameter CatBoost used for every other
    diagnostic (§4.4), and compares the two rankings.

    CatBoost's native ShapValues is the same TreeSHAP algorithm as the `shap`
    package's TreeExplainer, and avoids adding a dependency.

    NOTE on what the published attribution actually was. Manuscript §3.9 states
    that attributions were obtained "on every fold of the standard 5-fold CV
    split". The released code does not do that: pipeline S7 fits `cb_full` on
    the entire winsorised dataset and calls shap.TreeExplainer on that same
    data, giving ONE in-sample, full-data attribution and no folds at all.
    Supplementary Table S5's own caption says "full-data fit", contradicting
    §3.9. So this block is not a like-for-like repeat of the published
    procedure -- it is an out-of-fold attribution under both schemes, which is
    what Reviewer 3's comment presupposes and is the more defensible estimate.
    §3.9 needs correcting either way.
    """
    from catboost import Pool
    out = os.path.join(RES_DIR, 'S14_shap_by_scheme.csv')
    if os.path.exists(out):
        log('SKIP SHAP (already done)'); return

    groups = df['Concrete_Class'].values
    feats = FEAT_28D
    rankings = {}

    for scheme in ['standard', 'group']:
        if scheme == 'group':
            k = min(5, df['Concrete_Class'].nunique())
            splits = list(GroupKFold(n_splits=k).split(df, groups=groups))
        else:
            splits = list(KFold(5, shuffle=True, random_state=SEED).split(df))

        abs_shap = np.zeros((len(df), len(feats)))
        for tr_idx, va_idx in splits:
            tr, va, _ = preprocess_fold(df.iloc[tr_idx], df.iloc[va_idx])
            Xtr = tr[feats].values.astype(np.float32)
            ytr = tr['Strength_28d'].values.astype(np.float32)
            Xva = va[feats].values.astype(np.float32)
            m = CatBoostRegressor(**CB_ABLATION_PARAMS, random_seed=SEED, verbose=0)
            m.fit(Xtr, ytr, verbose=0)          # no eval_set: no early-stop leak
            sv = m.get_feature_importance(Pool(Xva), type='ShapValues')
            abs_shap[va_idx, :] = np.abs(sv[:, :-1])   # last column is the base value
        rankings[scheme] = pd.Series(abs_shap.mean(axis=0), index=feats)
        log(f'   {scheme}: top 5 = '
            f'{", ".join(rankings[scheme].nlargest(5).index.tolist())}')

    tbl = pd.DataFrame({
        'feature': feats,
        'meanabs_SHAP_standard': rankings['standard'].values,
        'meanabs_SHAP_group': rankings['group'].values,
    })
    tbl['rank_standard'] = tbl['meanabs_SHAP_standard'].rank(ascending=False).astype(int)
    tbl['rank_group'] = tbl['meanabs_SHAP_group'].rank(ascending=False).astype(int)
    tbl['rank_shift'] = tbl['rank_standard'] - tbl['rank_group']
    tbl = tbl.sort_values('rank_standard')

    rho = tbl['rank_standard'].corr(tbl['rank_group'], method='spearman')
    top10_std = set(tbl.nsmallest(10, 'rank_standard')['feature'])
    top10_grp = set(tbl.nsmallest(10, 'rank_group')['feature'])
    overlap = len(top10_std & top10_grp)

    tbl.to_csv(out, index=False)
    summ = pd.DataFrame([{'spearman_rank_corr': rho, 'top10_overlap': overlap,
                          'max_abs_rank_shift': int(tbl.rank_shift.abs().max())}])
    summ.to_csv(os.path.join(RES_DIR, 'S14_shap_agreement.csv'), index=False)
    log(f'DONE  SHAP: Spearman rho = {rho:.3f}, top-10 overlap {overlap}/10, '
        f'largest rank shift {int(tbl.rank_shift.abs().max())} places -> {out}')


# ================================ R3-7: Pareto applicability-domain filter
def run_PARETO(df):
    """Reviewer 3 item 7. The published search samples the three design
    variables INDEPENDENTLY within their marginal 5-95 % bounds, which can
    generate combinations that are individually plausible but jointly absent
    from the training data (high cement at high W/C being the obvious case).

    This reproduces the published search exactly, then adds a joint-distribution
    applicability-domain filter -- Mahalanobis distance in the three-dimensional
    design space against the training covariance, thresholded at the chi-square
    95th percentile with 3 degrees of freedom -- and reports both fronts.
    """
    from sklearn.linear_model import Ridge
    from scipy.stats import chi2
    out = os.path.join(RES_DIR, 'S15_pareto_applicability_domain.csv')
    if os.path.exists(out):
        log('SKIP PARETO (already done)'); return

    # --- full-data fit, reproducing df_full_proc from pipeline S7 exactly.
    # The winsorisation step matters: omitting it shifts the front by two
    # designs (69 instead of the published 67).
    bounds_full = fit_winsor_bounds(df)
    full = apply_winsor_bounds(df, bounds_full)
    full = engineer_features(full)
    fb = df['Cement'].value_counts(normalize=True)
    fa = df['Admixture'].value_counts(normalize=True)
    full['Cement_Freq'] = df['Cement'].map(fb).fillna(0.0).values
    full['Admix_Freq'] = df['Admixture'].map(fa).fillna(0.0).values
    cb_full = CatBoostRegressor(**CB_ABLATION_PARAMS, random_seed=SEED, verbose=0)
    cb_full.fit(full[FEAT_28D].values.astype(np.float32),
                df['Strength_28d'].values.astype(np.float32), verbose=0)

    proxy_feats = ['Cement_Content', 'WC_Ratio', 'Admixture_Dose']
    pX = full[proxy_feats].values.astype(np.float32)
    proxy_f7 = Ridge(alpha=1.0).fit(pX, df['Strength_7d'].values)
    proxy_sl = Ridge(alpha=1.0).fit(pX, df['Slump_30'].values)

    N_MC = 10000
    rng = np.random.default_rng(SEED)
    lo = {c: float(df[c].quantile(0.05)) for c in proxy_feats}
    hi = {c: float(df[c].quantile(0.95)) for c in proxy_feats}
    log(f'   sampling bounds (5-95 % of data): '
        f'cement {lo["Cement_Content"]:.0f}-{hi["Cement_Content"]:.0f}, '
        f'W/C {lo["WC_Ratio"]:.3f}-{hi["WC_Ratio"]:.3f}, '
        f'admix {lo["Admixture_Dose"]:.2f}-{hi["Admixture_Dose"]:.2f}')

    mc_cem = rng.uniform(lo['Cement_Content'], hi['Cement_Content'], N_MC)
    mc_wc = rng.uniform(lo['WC_Ratio'], hi['WC_Ratio'], N_MC)
    mc_ad = rng.uniform(lo['Admixture_Dose'], hi['Admixture_Dose'], N_MC)

    pin = np.column_stack([mc_cem, mc_wc, mc_ad]).astype(np.float32)
    mc_f7 = proxy_f7.predict(pin).clip(df['Strength_7d'].min(), df['Strength_7d'].max())
    mc_sl = proxy_sl.predict(pin).clip(df['Slump_30'].min(), df['Slump_30'].max())

    mc = pd.DataFrame({
        'Concrete_Class': float(df['Concrete_Class'].median()),
        'CA': float(df['CA'].median()), 'FA': float(df['FA'].median()),
        'Cement_Content': mc_cem, 'WC_Ratio': mc_wc, 'Admixture_Dose': mc_ad,
        'Strength_7d': mc_f7, 'Slump_30': mc_sl,
        'Slump_60': float(df['Slump_60'].median()),
        'Slump_90': float(df['Slump_90'].median()),
        'Cement': 'Median', 'Admixture': 'Median'})
    mc = engineer_features(mc)
    mc['Cement_Freq'] = float(full['Cement_Freq'].median())
    mc['Admix_Freq'] = float(full['Admix_Freq'].median())
    pred = cb_full.predict(mc[FEAT_28D].values.astype(np.float32))

    water = mc_cem * mc_wc
    feasible = (pred > 20) & (water >= 120) & (water <= 220) & (mc_wc >= 0.25)

    # --- the applicability-domain filter
    train = df[proxy_feats].values.astype(float)
    mu, cov = train.mean(axis=0), np.cov(train, rowvar=False)
    inv = np.linalg.inv(cov)
    d = pin.astype(float) - mu
    maha2 = np.einsum('ij,jk,ik->i', d, inv, d)
    thresh = chi2.ppf(0.95, df=3)
    in_domain = maha2 <= thresh
    # how much of the TRAINING data itself passes, as a sanity check
    dt = train - mu
    train_in = (np.einsum('ij,jk,ik->i', dt, inv, dt) <= thresh).mean()

    def front(mask):
        cem, st, wc = mc_cem[mask], pred[mask], mc_wc[mask]
        keep = np.zeros(len(cem), dtype=bool)
        mn = np.inf
        for i in np.argsort(-st):
            if cem[i] < mn:
                keep[i] = True; mn = cem[i]
        return pd.DataFrame({'Cement': cem[keep], 'WC': wc[keep], 'fc28': st[keep],
                             'Eff': st[keep] / cem[keep]}).sort_values(
                                 'fc28', ascending=False).reset_index(drop=True)

    f_pub = front(feasible)
    f_ad = front(feasible & in_domain)

    rows = []
    for lab, f, nc in [('Published (marginal bounds only)', f_pub, int(feasible.sum())),
                       ('With applicability-domain filter', f_ad,
                        int((feasible & in_domain).sum()))]:
        rows.append(dict(front=lab, n_candidates=nc, n_nondominated=len(f),
                         cement_min=f.Cement.min(), cement_max=f.Cement.max(),
                         fc28_min=f.fc28.min(), fc28_max=f.fc28.max(),
                         eff_min=f.Eff.min(), eff_max=f.Eff.max()))
    res = pd.DataFrame(rows)
    res['mahalanobis_threshold_chi2_95_df3'] = thresh
    res['pct_MC_in_domain'] = 100 * in_domain.mean()
    res['pct_training_in_domain'] = 100 * train_in
    res.to_csv(out, index=False)
    f_pub.to_csv(os.path.join(RES_DIR, 'S15_front_published.csv'), index=False)
    f_ad.to_csv(os.path.join(RES_DIR, 'S15_front_filtered.csv'), index=False)

    log(f'   {in_domain.mean()*100:.1f} % of Monte Carlo candidates lie inside the '
        f'domain ({train_in*100:.1f} % of training rows do)')
    log(f'   published front {len(f_pub)} designs, filtered front {len(f_ad)}')
    log(f'DONE  PARETO -> {out}')


# ===== Remaining CatBoost diagnostics that also early-stop on held-out data
def _cb_oof(df, feats, outer, es, target='Strength_28d', group_col='Concrete_Class'):
    groups = df[group_col].values
    if outer == 'group':
        splits = list(GroupKFold(n_splits=min(5, df[group_col].nunique())
                                 ).split(df, groups=groups))
    else:
        splits = list(KFold(5, shuffle=True, random_state=SEED).split(df))
    y_oof = np.zeros(len(df))
    for tr_idx, va_idx in splits:
        tr, va, _ = preprocess_fold(df.iloc[tr_idx], df.iloc[va_idx])
        avail = [f for f in feats if f in tr.columns]
        Xtr = tr[avail].values.astype(np.float32)
        ytr = tr[target].values.astype(np.float32)
        Xva = va[avail].values.astype(np.float32)
        yva = va[target].values.astype(np.float32)
        m = CatBoostRegressor(**CB_ABLATION_PARAMS, random_seed=SEED, verbose=0)
        if es == 'none':
            m.fit(Xtr, ytr, verbose=0)
        else:
            m.fit(Xtr, ytr, eval_set=(Xva, yva), early_stopping_rounds=50, verbose=0)
        y_oof[va_idx] = m.predict(Xva)
    return y_oof


def run_DIAG(df):
    """Table 6 (anchor ablation), Table 5 (class-deletion jackknife) and the
    Figure 5 bootstrap all use CatBoost fitted with eval_set on the very rows
    they are scored on -- the same defect as D2, at patience 50 and 30
    respectively. Each is recomputed here both as published and without it."""
    out = os.path.join(RES_DIR, 'S16_diagnostics.csv')
    if os.path.exists(out):
        log('SKIP DIAG (already done)'); return
    y = df['Strength_28d'].values
    rows = []

    ABL = {
        'Full (30 features)': FEAT_28D,
        "Without f'c,7": [f for f in FEAT_28D if f != 'Strength_7d'],
        'Without slump features': [f for f in FEAT_28D if 'Slump' not in f],
        'Without interactions': [f for f in FEAT_28D if '_x_' not in f
                                 and '_sq' not in f and '_inv' not in f],
        'Raw only (8 features)': ['Concrete_Class', 'CA', 'FA', 'Cement_Content',
                                  'WC_Ratio', 'Admixture_Dose', 'Strength_7d',
                                  'Slump_30'],
    }
    for es in ('published', 'none'):
        log(f'DIAG [es={es}]: Table 6 anchor ablation')
        base = {}
        for name, feats in ABL.items():
            r = {}
            for outer in ('standard', 'group'):
                p = _cb_oof(df, feats, outer, es)
                r[outer] = r2_score(y, p)
                r[f'rmse_{outer}'] = float(np.sqrt(np.mean((y - p) ** 2)))
            if name.startswith('Full'):
                base = dict(r)
            rows.append(dict(block='ablation', variant=name, es=es,
                             n_features=len([f for f in feats if f in FEAT_28D]),
                             R2_standard=r['standard'], R2_group=r['group'],
                             RMSE_standard=r['rmse_standard'],
                             RMSE_group=r['rmse_group'],
                             dR2_standard=r['standard'] - base.get('standard', np.nan),
                             dR2_group=r['group'] - base.get('group', np.nan)))
            log(f'   {name:24s} {r["standard"]:.4f} / {r["group"]:.4f}')

        log(f'DIAG [es={es}]: Table 5 class-deletion jackknife')
        for cls in sorted(df['Concrete_Class'].unique()):
            sub = df[df['Concrete_Class'] != cls].reset_index(drop=True)
            if sub['Concrete_Class'].nunique() < 2:
                continue
            ys = sub['Strength_28d'].values
            ps = _cb_oof(sub, FEAT_28D, 'standard', es)
            pg = _cb_oof(sub, FEAT_28D, 'group', es)
            r2s, r2g = r2_score(ys, ps), r2_score(ys, pg)
            rows.append(dict(block='jackknife', variant=f'without M{cls}', es=es,
                             n_train=len(sub), R2_standard=r2s, R2_group=r2g,
                             dR2_standard=np.nan, dR2_group=r2s - r2g))
        jk = [r for r in rows if r['block'] == 'jackknife' and r['es'] == es]
        log(f'   mean gap across deletions: '
            f'{np.mean([r["dR2_group"] for r in jk]):.4f}')

        log(f'DIAG [es={es}]: Figure 5 bootstrap (B=200)')
        rng = np.random.default_rng(SEED)
        n, r2l = len(df), []
        for b in range(200):
            idx = rng.integers(0, n, size=n)
            oob = np.setdiff1d(np.arange(n), np.unique(idx))
            if len(oob) < 20:
                continue
            tr, va, _ = preprocess_fold(df.iloc[idx], df.iloc[oob])
            avail = [f for f in FEAT_28D if f in tr.columns]
            Xtr = tr[avail].values.astype(np.float32)
            ytr = tr['Strength_28d'].values.astype(np.float32)
            Xva = va[avail].values.astype(np.float32)
            yva = va['Strength_28d'].values.astype(np.float32)
            m = CatBoostRegressor(**CB_ABLATION_PARAMS, random_seed=SEED + b, verbose=0)
            if es == 'none':
                m.fit(Xtr, ytr, verbose=0)
            else:
                m.fit(Xtr, ytr, eval_set=(Xva, yva), early_stopping_rounds=30, verbose=0)
            r2l.append(r2_score(yva, m.predict(Xva)))
        r2l = np.array(r2l)
        rows.append(dict(block='bootstrap', variant=f'B={len(r2l)} OOB replicates',
                         es=es, R2_standard=float(np.median(r2l)),
                         ci_lo=float(np.percentile(r2l, 2.5)),
                         ci_hi=float(np.percentile(r2l, 97.5))))
        log(f'   median R2 {np.median(r2l):.4f} '
            f'[{np.percentile(r2l,2.5):.4f}, {np.percentile(r2l,97.5):.4f}]')

    pd.DataFrame(rows).to_csv(out, index=False)
    log(f'DONE  DIAG -> {out}')


def run_TABLE3(df):
    """Assemble the corrected Table 3 from the OOF dumps the nested-CV jobs
    already wrote. No refitting -- this runs in seconds and supplies the MAPE
    column that run_scheme does not record."""
    out = os.path.join(RES_DIR, 'TABLE3_corrected.csv')
    MAP = {('Strength_28d', 'standard'): 'C2', ('Strength_28d', 'group'): 'C3',
           ('Strength_7d', 'group'): 'E1', ('Slump_30', 'group'): 'E2',
           ('Slump_90', 'group'): 'E3', ('Strength_7d', 'standard'): 'E4',
           ('Slump_30', 'standard'): 'E5', ('Slump_90', 'standard'): 'E6'}
    rows, missing = [], []
    for (t, outer), jid in MAP.items():
        f = os.path.join(RES_DIR, f'{jid}_oof.csv')
        if not os.path.exists(f):
            missing.append(f'{jid} ({t}/{outer})'); continue
        d = pd.read_csv(f)
        yt, yp = d['y_true'].values, d['y_ens'].values
        mask = np.abs(yt) > 1e-9
        rows.append(dict(target=t, scheme=outer, job=jid,
                         R2=r2_score(yt, yp),
                         RMSE=float(np.sqrt(np.mean((yt - yp) ** 2))),
                         MAE=float(np.mean(np.abs(yt - yp))),
                         MAPE=float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100)))
    if missing:
        log(f'   NOTE: no OOF yet for {", ".join(missing)}')
    if not rows:
        log('   nothing to assemble'); return
    tb = pd.DataFrame(rows)
    order = ['Strength_28d', 'Strength_7d', 'Slump_30', 'Slump_90']
    tb['_o'] = tb.target.map({t: i for i, t in enumerate(order)})
    tb = tb.sort_values(['_o', 'scheme'], ascending=[True, False]).drop(columns='_o')
    tb.to_csv(out, index=False)
    print()
    print(tb.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    log(f'DONE  TABLE3 -> {out}')


# ===================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', default='GATE',
                    help='ALL | FAST (S12+ALPHA+SHAP+PARETO) | DIAG (Tables 5,6 '
                         'and the Fig. 5 bootstrap, both with and without the '
                         'early-stopping pattern) | TABLE3 (assemble from OOF, '
                         'seconds) | GATE | S12 | ALPHA | SHAP | PARETO | R3_2 | '
                         'ESLEAK | R5_1 | TARGETS | R3_5 | R3_10 | or ids (A1,B1,...)')
    ap.add_argument('--list', action='store_true')
    a = ap.parse_args()

    if a.list:
        print(f'{"id":6s} {"group":8s} {"done":5s} desc')
        for j in PRIORITY:
            s = JOBS[j]
            print(f'{j:6s} {s["group"]:8s} '
                  f'{"yes" if os.path.exists(job_path(j)) else "no":5s} {s["desc"]}')
        print(f'{"S12":6s} {"S12":8s} '
              f'{"yes" if os.path.exists(os.path.join(RES_DIR,"S12_TableS12.csv")) else "no":5s} '
              'Regenerate Table S12 rows (fast)')
        for nm, f, desc in [
                ('ALPHA', 'S13_alpha_sensitivity.csv',
                 'R3-6 hydration-degree sensitivity (fast)'),
                ('SHAP', 'S14_shap_by_scheme.csv',
                 'R3-4 TreeSHAP under GroupKFold vs standard CV (fast)'),
                ('PARETO', 'S15_pareto_applicability_domain.csv',
                 'R3-7 Pareto applicability-domain filter (fast)')]:
            print(f'{nm:6s} {nm:8s} '
                  f'{"yes" if os.path.exists(os.path.join(RES_DIR, f)) else "no":5s} '
                  f'{desc}')
        return

    df = load_data()
    log(f'Data: n={len(df)}, classes={df.Concrete_Class.nunique()}, '
        f'brands={df.Cement.nunique()}, admixtures={df.Admixture.nunique()}')
    assert len(df) == 683, f'EXPECTED 683 RECORDS, GOT {len(df)} -- stop and report this'

    sel = a.jobs.strip()
    if sel == 'ALL':
        run_S12(df); run_ALPHA(df); run_SHAP(df); run_PARETO(df)
        ids = PRIORITY
    elif sel == 'FAST':
        run_S12(df); run_ALPHA(df); run_SHAP(df); run_PARETO(df); return
    elif sel == 'S12':
        run_S12(df); return
    elif sel == 'ALPHA':
        run_ALPHA(df); return
    elif sel == 'SHAP':
        run_SHAP(df); return
    elif sel == 'PARETO':
        run_PARETO(df); return
    elif sel == 'DIAG':
        run_DIAG(df); return
    elif sel == 'TABLE3':
        run_TABLE3(df); return
    elif sel in {'GATE', 'R3_2', 'ESLEAK', 'R5_1', 'TARGETS', 'R3_5', 'R3_10'}:
        ids = [j for j in PRIORITY if JOBS[j]['group'] == sel]
    else:
        ids = [x.strip() for x in sel.split(',') if x.strip()]

    for jid in ids:
        if jid not in JOBS:
            log(f'unknown job id {jid}'); continue
        run_job(jid, df)

    parts = [pd.read_csv(job_path(j)) for j in PRIORITY if os.path.exists(job_path(j))]
    if parts:
        summ = pd.concat(parts, ignore_index=True)
        cols = ['jid', 'group', 'target', 'outer', 'inner', 'es', 'encoding',
                'group_col', 'n_features', 'R2', 'RMSE', 'MAE', 'minutes']
        summ = summ[[c for c in cols if c in summ.columns]]
        p = os.path.join(RES_DIR, 'SUMMARY.csv')
        summ.to_csv(p, index=False)
        print('\n' + summ.to_string(index=False))
        log(f'Summary -> {p}')


if __name__ == '__main__':
    main()
