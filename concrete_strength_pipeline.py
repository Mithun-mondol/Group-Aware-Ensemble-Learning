"""
ML-Driven Multi-Target Prediction of Concrete Properties
A Dual-Validation Framework with Generalization Gap Analysis

Data source: SASEC Dhaka-Sylhet Corridor Road Investment Project
Dataset:     Dhaka-Sylhet_mix_design_final.xlsx (header row index = 2)
Run:         python concrete_strength_pipeline.py

Sections:
    S1.  Imports and configuration
    S2.  Data loading and quality control
    S3.  Feature definitions and fold-local preprocessing
    S4.  Nested cross-validation (outer CV + inner HPO)
    S5.  Baselines and feature ablation
    S5b. Multi-task physics-informed neural network (MTL-PINN)
    S6.  Bootstrap out-of-bag confidence intervals (B=200, CatBoost)
    S7.  TreeSHAP explainability and stability
    S8.  Supplementary tables (S1 winsorisation, S2 LOCO)
    S9.  Statistical supplement
    S10. ACI 318-19 / Eurocode 2 code compliance
    S11. Multi-objective Pareto optimisation
    S12. Figure generation (Fig 1-8, Fig S1)
    S13. Final results summary and CSV export

Requirements (pin versions for reproducibility):
    numpy==1.26.4        pandas==2.2.2       scipy==1.13.0
    scikit-learn==1.5.0  xgboost==2.0.3      lightgbm==4.3.0
    catboost==1.2.5      optuna==3.6.1       shap==0.45.1
    matplotlib==3.9.0    statsmodels==0.14.2
"""

# =============================================================================
# S1. IMPORTS AND CONFIGURATION
# =============================================================================

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.ndimage import uniform_filter1d
from scipy.stats import shapiro, normaltest, wilcoxon, gaussian_kde

from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import optuna
from optuna.samplers import TPESampler
import shap
import joblib

# TensorFlow/Keras for MTL-PINN (S5b)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import tensorflow as tf
import keras
from tensorflow.keras import layers, Model, callbacks, optimizers, regularizers
from tensorflow.keras import backend as K

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
tf.get_logger().setLevel('ERROR')

# --- Reproducibility ---------------------------------------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# --- Output settings ---------------------------------------------------------
DPI        = 600
OUT_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_FILE  = os.path.join(OUT_DIR, 'Dhaka-Sylhet_mix_design.xlsx')

# --- Colourblind-safe palette (Okabe-Ito) ------------------------------------
PALETTE = {
    'XGBoost'   : '#0072B2',
    'LightGBM'  : '#E69F00',
    'CatBoost'  : '#009E73',
    'ExtraTrees': '#CC79A7',
    'Ensemble'  : '#D55E00',
}

# --- Publication matplotlib style --------------------------------------------
plt.rcParams.update({
    'font.family'      : 'serif',
    'font.serif'       : ['Times New Roman', 'DejaVu Serif'],
    'font.size'        : 9,
    'axes.labelsize'   : 10,
    'axes.titlesize'   : 10,
    'axes.linewidth'   : 0.6,
    'xtick.labelsize'  : 8,
    'ytick.labelsize'  : 8,
    'legend.fontsize'  : 8,
    'legend.frameon'   : False,
    'figure.dpi'       : 150,
    'savefig.dpi'      : DPI,
    'savefig.bbox'     : 'tight',
    'savefig.pad_inches': 0.02,
    'grid.alpha'       : 0.3,
    'grid.linewidth'   : 0.4,
})

# --- Utility functions -------------------------------------------------------
def calc_metrics(y_true, y_pred):
    """Compute R², RMSE (MPa), MAE (MPa), MAPE (%)."""
    mask = np.abs(y_true) > 1e-8
    return {
        'R2'  : r2_score(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE' : mean_absolute_error(y_true, y_pred),
        'MAPE': np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100,
    }

def panel_label(ax, label, x=-0.12, y=1.06):
    ax.text(x, y, f'({label})', transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='top', ha='right')

def clean_spines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def save_fig(fig, name):
    """Save figure as PNG and PDF to OUT_DIR."""
    for fmt in ['png', 'pdf']:
        path = os.path.join(OUT_DIR, f'{name}.{fmt}')
        fig.savefig(path, dpi=DPI, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f'  {name} saved (PNG + PDF)')

print('[S1] Environment ready.')


# =============================================================================
# S2. DATA LOADING & QUALITY CONTROL
# =============================================================================

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

df_raw['Cement']    = df_raw['Cement'].astype(str).str.strip().str.title()
df_raw['Admixture'] = df_raw['Admixture'].astype(str).str.strip().str.title()

CRITICAL_COLS = [
    'Concrete_Class', 'Cement', 'Admixture', 'Cement_Content',
    'WC_Ratio', 'Admixture_Dose', 'Strength_7d', 'Strength_28d',
    'Slump_30', 'Slump_90',
]
n_raw = len(df_raw)
df_raw = df_raw.dropna(subset=CRITICAL_COLS).reset_index(drop=True)
n_after_na = len(df_raw)

# Physical plausibility: remove strength regression (f'c,28 < f'c,7)
df_raw = df_raw[df_raw['Strength_28d'] >= df_raw['Strength_7d']].reset_index(drop=True)
n_after_phys = len(df_raw)

DUP_COLS = [
    'Cement', 'Admixture', 'CA', 'FA', 'Cement_Content', 'WC_Ratio',
    'Admixture_Dose', 'Strength_7d', 'Strength_28d',
    'Slump_30', 'Slump_60', 'Slump_90',
]
df_raw = df_raw.drop_duplicates(subset=DUP_COLS, keep='first').reset_index(drop=True)
n_final = len(df_raw)

# Winsorisation is applied strictly inside each training fold (S4, S6).
# df_raw is kept unmodified for all downstream analyses.

print(f'[S2] Data flow:')
print(f'     Raw import          : {n_raw}')
print(f'     After NA removal    : {n_after_na} (removed {n_raw - n_after_na})')
print(f'     After phys. filter  : {n_after_phys} (removed {n_after_na - n_after_phys})')
print(f'     After deduplication : {n_final} (removed {n_after_phys - n_final})')
print(f'     Final dataset       : {n_final} records, ')
print(f'                           {df_raw["Concrete_Class"].nunique()} strength classes')
print(df_raw['Concrete_Class'].value_counts().sort_index().to_string())


# =============================================================================
# S3. FEATURE DEFINITIONS AND FOLD-LOCAL PREPROCESSING HELPERS
#
# A single canonical feature list is used identically across all analyses.
# Winsorisation and frequency encoding are always fold-local: bounds and
# encodings are fitted on the training fold only.
# =============================================================================

# Columns that must never appear as model inputs
EXCLUDE_COLS = {
    'SN', 'Concrete_Class', 'Cement', 'Admixture',
    'Strength_28d', 'Strength_7d', 'Slump_30', 'Slump_90', 'Slump_60',
}

# Raw numeric columns to winsorise (bounds fitted on training fold only)
WINSOR_COLS = [
    'Cement_Content', 'WC_Ratio', 'Admixture_Dose',
    'CA', 'FA', 'Strength_7d', 'Slump_30', 'Slump_60', 'Slump_90',
]

# Canonical 30-feature set for f'c,28 prediction.
# Strength_7d is included as a legitimate input (available 21 days before the
# 28-day test). Cement_Efficiency and Strength_Gain_Ratio are excluded because
# they are functions of the target (data leakage).
FEAT_28D = [
    # Raw mixture proportions
    'Concrete_Class', 'CA', 'FA', 'Cement_Content', 'WC_Ratio',
    'Admixture_Dose', 'Strength_7d', 'Slump_30',
    # Derived mixture features
    'Water_Content', 'Paste_Volume', 'CA_FA_Ratio', 'Agg_Volume_Frac',
    'Admix_pct_bwoc', 'Admix_per_Cement',
    # Abrams' law / constitutive features
    'WC_Ratio_sq', 'WC_Ratio_inv', 'log_Cement', 'Cement_x_WC',
    'Bolomey_Feature',
    # Polynomial and interaction features
    'Concrete_Class_sq', 'Class_x_WC', 'Class_x_Cement', 'Cement_sq',
    'CA_FA_x_WC',
    # Engineering features
    'Binder_Intensity', 'Gel_Space_Ratio_28d',
    # Slump-derived features
    'Slump_Retention', 'Slump_Loss_Rate',
    # Frequency-encoded categoricals (computed fold-locally — see add_fold_freq)
    'Cement_Freq', 'Admix_Freq',
]

# Feature set for slump targets (excludes Strength_7d and slump-derived features)
FEAT_SLUMP = [f for f in FEAT_28D if f not in
              {'Strength_7d', 'Slump_30', 'Slump_Retention',
               'Slump_Loss_Rate', 'Gel_Space_Ratio_28d'}]

assert len(FEAT_28D) == 30, f"Expected 30 features, got {len(FEAT_28D)}"
assert len(FEAT_SLUMP) == 25, f"Expected 25 slump features, got {len(FEAT_SLUMP)}"

# --- Preprocessing helpers ---------------------------------------------------

def fit_winsor_bounds(train_df, cols=WINSOR_COLS, lo=0.01, hi=0.99):
    """Fit Winsorisation bounds on training data only.
    Returns {col: (lower_bound, upper_bound)}."""
    bounds = {}
    for c in cols:
        if c in train_df.columns:
            bounds[c] = (float(train_df[c].quantile(lo)),
                         float(train_df[c].quantile(hi)))
    return bounds

def apply_winsor_bounds(df, bounds):
    """Apply pre-fitted Winsorisation bounds to any dataframe."""
    out = df.copy()
    for c, (ql, qh) in bounds.items():
        if c in out.columns:
            out[c] = out[c].clip(ql, qh)
    return out

def engineer_features(df):
    """Compute all derived features from raw columns.
    Input df must contain the raw measurement columns.
    Returns df with all 30 engineered features added."""
    d = df.copy()
    d['Water_Content']       = d['Cement_Content'] * d['WC_Ratio']
    d['Paste_Volume']        = (d['Cement_Content'] / 3150 +
                                  d['Water_Content'] / 1000) * 1000
    d['CA_FA_Ratio']         = d['CA'] / d['FA']
    d['Admix_pct_bwoc']      = d['Admixture_Dose'] / 100
    d['Admix_per_Cement']    = d['Admixture_Dose'] / d['Cement_Content'] * 1000
    d['Agg_Volume_Frac']     = (d['CA'] + d['FA']) / 100
    d['WC_Ratio_sq']         = d['WC_Ratio'] ** 2
    d['WC_Ratio_inv']        = 1.0 / d['WC_Ratio']
    d['log_Cement']          = np.log(d['Cement_Content'])
    d['Cement_x_WC']         = d['Cement_Content'] * d['WC_Ratio']
    d['Bolomey_Feature']     = (1.0 / d['WC_Ratio']) - 0.5
    d['Concrete_Class_sq']   = d['Concrete_Class'] ** 2
    d['Class_x_WC']          = d['Concrete_Class'] * d['WC_Ratio']
    d['Class_x_Cement']      = d['Concrete_Class'] * d['Cement_Content']
    d['Cement_sq']           = d['Cement_Content'] ** 2
    d['CA_FA_x_WC']          = d['CA_FA_Ratio'] * d['WC_Ratio']
    d['Binder_Intensity']    = d['Cement_Content'] / d['Concrete_Class']
    alpha_28 = 0.75
    d['Gel_Space_Ratio_28d'] = (0.68 * alpha_28) / (0.32 * alpha_28 + d['WC_Ratio'])
    d['Slump_Retention']     = d['Slump_90'] / d['Slump_30']
    d['Slump_Loss_Rate']     = (d['Slump_30'] - d['Slump_90']) / 60.0
    return d

def add_fold_freq(train_df, valid_df,
                  brand='Cement', admix='Admixture'):
    """Compute frequency encodings on training data only, then map to validation.
    Prevents frequency-encoding leakage across folds."""
    fb = train_df[brand].value_counts(normalize=True)
    fa = train_df[admix].value_counts(normalize=True)

    def _apply(df_):
        df_ = df_.copy()
        df_['Cement_Freq'] = df_[brand].map(fb).fillna(0.0).values
        df_['Admix_Freq']  = df_[admix].map(fa).fillna(0.0).values
        return df_
    return _apply(train_df), _apply(valid_df)

def preprocess_fold(train_df, valid_df, lo=0.01, hi=0.99):
    """Full fold-local preprocessing pipeline:
      1. Fit Winsorisation bounds on train, apply to both.
      2. Engineer features on both.
      3. Fit frequency encodings on train, apply to both.
    Returns (train_processed, valid_processed).
    All fitting is strictly train-only."""
    bounds = fit_winsor_bounds(train_df, lo=lo, hi=hi)
    tr = apply_winsor_bounds(train_df, bounds)
    va = apply_winsor_bounds(valid_df,  bounds)
    tr = engineer_features(tr)
    va = engineer_features(va)
    tr, va = add_fold_freq(tr, va)
    return tr, va

print(f'[S3] Feature sets defined: FEAT_28D={len(FEAT_28D)}, FEAT_SLUMP={len(FEAT_SLUMP)}')
print(f'     Leakage check — Cement_Efficiency in FEAT_28D: {"Cement_Efficiency" in FEAT_28D}')
print(f'     Leakage check — Strength_Gain_Ratio in FEAT_28D: {"Strength_Gain_Ratio" in FEAT_28D}')


# =============================================================================
# S4. NESTED CROSS-VALIDATION (OUTER CV FOR REPORTING, INNER CV FOR HPO)
#
# HPO is performed inside the outer training fold only. Reported metrics are
# from the outer held-out fold. Ensemble weights are also optimised inside the
# outer training fold.
#
# Architecture:
#   Outer loop : 5-fold CV (standard) + GroupKFold (cross-class)
#   Inner loop : 5-fold CV for Optuna HPO (50 trials per outer fold)
#   Ensemble   : Nelder-Mead weight optimisation on inner OOF predictions
# =============================================================================

N_OUTER_FOLDS = 5
N_INNER_FOLDS = 5
N_HPO_TRIALS  = 50   # per outer fold

TARGETS_CONFIG = {
    'Strength_28d': FEAT_28D,
    'Strength_7d' : [f for f in FEAT_28D if f != 'Strength_7d'],
    'Slump_30'    : FEAT_SLUMP,
    'Slump_90'    : FEAT_SLUMP,
}

def make_base_models(xgb_p, lgb_p, cb_p):
    """Instantiate base learners with given hyperparameters."""
    return {
        'XGBoost'   : xgb.XGBRegressor(**xgb_p, random_state=SEED,
                                         n_jobs=-1, verbosity=0),
        'LightGBM'  : lgb.LGBMRegressor(**lgb_p, random_state=SEED,
                                          n_jobs=-1, verbose=-1),
        'CatBoost'  : CatBoostRegressor(**cb_p, random_seed=SEED, verbose=0),
        'ExtraTrees': ExtraTreesRegressor(n_estimators=500, max_depth=20,
                                           min_samples_leaf=2,
                                           random_state=SEED, n_jobs=-1),
    }

def fit_model(model, name, Xtr, ytr, Xva, yva):
    """Fit a single model with early stopping where applicable."""
    if name == 'XGBoost':
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    elif name == 'LightGBM':
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    elif name == 'CatBoost':
        model.fit(Xtr, ytr, eval_set=(Xva, yva),
                  early_stopping_rounds=50, verbose=0)
    else:
        model.fit(Xtr, ytr)
    return model

def inner_hpo(X_inner, y_inner, n_trials=N_HPO_TRIALS):
    """Run Optuna HPO for XGBoost, LightGBM, CatBoost on inner CV.
    Returns best_params dict. Called only on outer-training data."""
    inner_cv = KFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=SEED)

    def _cv_score(model_fn, model_name):
        scores = []
        for tr, va in inner_cv.split(X_inner):
            m = model_fn()
            m = fit_model(m, model_name,
                          X_inner[tr], y_inner[tr],
                          X_inner[va], y_inner[va])
            scores.append(r2_score(y_inner[va], m.predict(X_inner[va])))
        return float(np.mean(scores))

    # XGBoost
    def obj_xgb(trial):
        p = {
            'n_estimators'    : trial.suggest_int('n_estimators', 300, 1500),
            'max_depth'       : trial.suggest_int('max_depth', 3, 8),
            'learning_rate'   : trial.suggest_float('learning_rate', 0.005, 0.15, log=True),
            'subsample'       : trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.9),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'reg_alpha'       : trial.suggest_float('reg_alpha', 1e-7, 5.0, log=True),
            'reg_lambda'      : trial.suggest_float('reg_lambda', 1e-7, 5.0, log=True),
        }
        return _cv_score(lambda: xgb.XGBRegressor(**p, random_state=SEED,
                                                    n_jobs=-1, verbosity=0,
                                                    early_stopping_rounds=30),
                         'XGBoost')

    # LightGBM
    def obj_lgb(trial):
        p = {
            'n_estimators'     : trial.suggest_int('n_estimators', 300, 1500),
            'max_depth'        : trial.suggest_int('max_depth', 3, 10),
            'learning_rate'    : trial.suggest_float('learning_rate', 0.005, 0.15, log=True),
            'subsample'        : trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.3, 0.9),
            'min_child_samples': trial.suggest_int('min_child_samples', 3, 40),
            'reg_alpha'        : trial.suggest_float('reg_alpha', 1e-7, 5.0, log=True),
            'reg_lambda'       : trial.suggest_float('reg_lambda', 1e-7, 5.0, log=True),
            'num_leaves'       : trial.suggest_int('num_leaves', 15, 127),
        }
        return _cv_score(lambda: lgb.LGBMRegressor(**p, random_state=SEED,
                                                     n_jobs=-1, verbose=-1),
                         'LightGBM')

    # CatBoost
    def obj_cb(trial):
        p = {
            'iterations'      : trial.suggest_int('iterations', 300, 1500),
            'depth'           : trial.suggest_int('depth', 4, 10),
            'learning_rate'   : trial.suggest_float('learning_rate', 0.005, 0.15, log=True),
            'l2_leaf_reg'     : trial.suggest_float('l2_leaf_reg', 0.001, 10.0, log=True),
            'subsample'       : trial.suggest_float('subsample', 0.6, 1.0),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 30),
        }
        return _cv_score(lambda: CatBoostRegressor(**p, random_seed=SEED,
                                                     verbose=0),
                         'CatBoost')

    best_params = {}
    for name, obj in [('xgb', obj_xgb), ('lgb', obj_lgb), ('cb', obj_cb)]:
        study = optuna.create_study(direction='maximize',
                                    sampler=TPESampler(seed=SEED))
        study.optimize(obj, n_trials=n_trials)
        best_params[name] = study.best_params
    return best_params

def optimise_ensemble_weights(oof_stack, y_true):
    """Optimise ensemble weights on inner OOF predictions.
    Called only on outer-training data."""
    def neg_r2(w):
        w = np.abs(w); w /= w.sum()
        return -r2_score(y_true, (oof_stack * w).sum(axis=1))
    n_models = oof_stack.shape[1]
    res = minimize(neg_r2, x0=np.ones(n_models) / n_models, method='Nelder-Mead')
    w = np.abs(res.x); w /= w.sum()
    return w

def run_outer_fold(df_fold, target, feat_list, outer_tr_idx, outer_va_idx):
    """
    Execute one outer fold:
      1. Preprocess (fold-local).
      2. Run inner HPO on outer-training data.
      3. Fit base models on outer-training data.
      4. Optimise ensemble weights on inner OOF of outer-training data.
      5. Predict on outer-validation data.
    Returns (y_true_va, y_pred_ensemble, per_model_preds, weights).
    """
    tr_raw = df_fold.iloc[outer_tr_idx]
    va_raw = df_fold.iloc[outer_va_idx]

    tr_proc, va_proc = preprocess_fold(tr_raw, va_raw)

    Xtr = tr_proc[feat_list].values.astype(np.float32)
    ytr = tr_proc[target].values.astype(np.float32)
    Xva = va_proc[feat_list].values.astype(np.float32)
    yva = va_proc[target].values.astype(np.float32)

    best_p = inner_hpo(Xtr, ytr)

    # Inner OOF for ensemble weight optimisation
    inner_cv = KFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=SEED)
    model_names = ['XGBoost', 'LightGBM', 'CatBoost', 'ExtraTrees']
    inner_oof = np.zeros((len(ytr), len(model_names)))

    for itr, iva in inner_cv.split(Xtr):
        models_i = make_base_models(best_p['xgb'], best_p['lgb'], best_p['cb'])
        for mi, (mname, m) in enumerate(models_i.items()):
            m = fit_model(m, mname, Xtr[itr], ytr[itr], Xtr[iva], ytr[iva])
            inner_oof[iva, mi] = m.predict(Xtr[iva])

    weights = optimise_ensemble_weights(inner_oof, ytr)

    # Fit final base models on full outer-training data
    models_final = make_base_models(best_p['xgb'], best_p['lgb'], best_p['cb'])
    va_preds = {}
    for mi, (mname, m) in enumerate(models_final.items()):
        m = fit_model(m, mname, Xtr, ytr, Xva, yva)
        va_preds[mname] = m.predict(Xva)

    oof_stack_va = np.column_stack([va_preds[n] for n in model_names])
    ens_pred = (oof_stack_va * weights).sum(axis=1)

    return yva, ens_pred, va_preds, weights, best_p

def run_dual_validation(df, target, feat_list):
    """
    Run full dual validation (Standard 5-Fold + GroupKFold) for one target.
    Returns ALL_OOF and ALL_RESULTS dicts for this target.
    """
    groups = df['Concrete_Class'].values
    n = len(df)

    schemes = {
        'standard' : KFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=SEED),
        'group'    : GroupKFold(n_splits=min(N_OUTER_FOLDS, df['Concrete_Class'].nunique())),
    }

    oof_results = {}
    metric_results = {}

    for scheme_name, splitter in schemes.items():
        y_oof   = np.zeros(n)
        per_model_oof = {m: np.zeros(n) for m in
                         ['XGBoost', 'LightGBM', 'CatBoost', 'ExtraTrees']}
        fold_weights = []

        split_kw = {'groups': groups} if scheme_name == 'group' else {}
        for tr_idx, va_idx in splitter.split(df, **split_kw):
            _, ens_pred, va_preds, w, _ = run_outer_fold(
                df, target, feat_list, tr_idx, va_idx)
            y_oof[va_idx]  = ens_pred
            for mname in per_model_oof:
                per_model_oof[mname][va_idx] = va_preds[mname]
            fold_weights.append(w)

        y_true = df[target].values.astype(np.float32)
        results = {'Ensemble': calc_metrics(y_true, y_oof)}
        for mname in per_model_oof:
            results[mname] = calc_metrics(y_true, per_model_oof[mname])

        oof_results[scheme_name]    = {'y': y_true, 'Ensemble': y_oof,
                                        **per_model_oof}
        metric_results[scheme_name] = results

        label = 'Standard 5-Fold' if scheme_name == 'standard' else 'GroupKFold'
        print(f'  {label}:')
        for mname in ['XGBoost', 'LightGBM', 'CatBoost', 'ExtraTrees', 'Ensemble']:
            m = results[mname]
            print(f'    {mname:12s} | R²={m["R2"]:.4f} | RMSE={m["RMSE"]:.3f}')
        mean_w = np.mean(fold_weights, axis=0)
        model_names = ['XGBoost', 'LightGBM', 'CatBoost', 'ExtraTrees']
        print(f'    Mean ensemble weights: ' +
              ', '.join(f'{n}={w:.3f}' for n, w in zip(model_names, mean_w)))

    return oof_results, metric_results

ALL_OOF     = {}
ALL_RESULTS = {}

for tgt, feats in TARGETS_CONFIG.items():
    print(f'\n[S4] TARGET: {tgt}  (n={len(df_raw)})')
    print('=' * 60)
    oof_r, met_r = run_dual_validation(df_raw, tgt, feats)
    ALL_OOF[tgt]     = oof_r
    ALL_RESULTS[tgt] = met_r

print('\n[S4] Dual validation complete.')


# =============================================================================
# S5. SIMPLE BASELINES & FEATURE ABLATION
# =============================================================================

print('\n[S5] Baselines & Feature Ablation')

def run_baseline_cv(df, target, feat_list, model_fn):
    """Run 5-fold CV for a baseline model with fold-local preprocessing."""
    n = len(df)
    y_oof = np.zeros(n)
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for tr_idx, va_idx in cv.split(df):
        tr_proc, va_proc = preprocess_fold(df.iloc[tr_idx], df.iloc[va_idx])
        Xtr = tr_proc[feat_list].values.astype(np.float32)
        ytr = tr_proc[target].values.astype(np.float32)
        Xva = va_proc[feat_list].values.astype(np.float32)
        m = model_fn()
        m.fit(Xtr, ytr)
        y_oof[va_idx] = m.predict(Xva)
    return calc_metrics(df[target].values.astype(np.float32), y_oof)

BASELINE_RESULTS = {}
baselines = {
    'OLS'   : lambda: Pipeline([('sc', StandardScaler()), ('m', LinearRegression())]),
    'Ridge' : lambda: Pipeline([('sc', StandardScaler()), ('m', Ridge(alpha=1.0))]),
    'RF-500': lambda: RandomForestRegressor(n_estimators=500, max_depth=20,
                                              min_samples_leaf=2,
                                              random_state=SEED, n_jobs=-1),
}
print(f'  {"Model":14s} | {"R²":>8s} | {"RMSE":>8s} | {"MAE":>8s}')
for bname, bfn in baselines.items():
    m = run_baseline_cv(df_raw, 'Strength_28d', FEAT_28D, bfn)
    BASELINE_RESULTS[bname] = m
    print(f'  {bname:14s} | {m["R2"]:8.4f} | {m["RMSE"]:8.3f} | {m["MAE"]:8.3f}')

ens_m = ALL_RESULTS['Strength_28d']['standard']['Ensemble']
print(f'  {"Ensemble":14s} | {ens_m["R2"]:8.4f} | {ens_m["RMSE"]:8.3f} | {ens_m["MAE"]:8.3f}')
print(f'  Ensemble improvement over OLS: ΔR² = +{ens_m["R2"] - BASELINE_RESULTS["OLS"]["R2"]:.4f}')

# Feature ablation
ABLATION_CONFIGS = {
    'Full (30 features)'       : FEAT_28D,
    'Without f\'c,7'           : [f for f in FEAT_28D if f != 'Strength_7d'],
    'Without slump features'   : [f for f in FEAT_28D if 'Slump' not in f],
    'Without interactions'     : [f for f in FEAT_28D
                                   if '_x_' not in f and '_sq' not in f
                                   and '_inv' not in f],
    'Raw only (8 features)'    : ['Concrete_Class', 'CA', 'FA', 'Cement_Content',
                                   'WC_Ratio', 'Admixture_Dose', 'Strength_7d', 'Slump_30'],
}

# Fixed CatBoost configuration for ablation comparisons.
CB_ABLATION_PARAMS = dict(iterations=800, depth=6, learning_rate=0.05,
                           l2_leaf_reg=1.0, subsample=0.8, min_data_in_leaf=5)

ABLATION_RESULTS = {}
full_r2 = None
print(f'\n  {"Config":30s} | {"n":>3s} | {"R²":>7s} | {"RMSE":>7s} | {"ΔR²":>7s}')
print(f'  {"-"*65}')
for cfg_name, feats in ABLATION_CONFIGS.items():
    valid_feats = [f for f in feats if f in FEAT_28D]
    n = len(df_raw)
    y_oof = np.zeros(n)
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for tr_idx, va_idx in cv.split(df_raw):
        tr_proc, va_proc = preprocess_fold(df_raw.iloc[tr_idx], df_raw.iloc[va_idx])
        avail = [f for f in valid_feats if f in tr_proc.columns]
        Xtr = tr_proc[avail].values.astype(np.float32)
        ytr = tr_proc['Strength_28d'].values.astype(np.float32)
        Xva = va_proc[avail].values.astype(np.float32)
        m = CatBoostRegressor(**CB_ABLATION_PARAMS, random_seed=SEED, verbose=0)
        m.fit(Xtr, ytr, eval_set=(Xva, va_proc['Strength_28d'].values.astype(np.float32)),
              early_stopping_rounds=50, verbose=0)
        y_oof[va_idx] = m.predict(Xva)
    r2_a  = r2_score(df_raw['Strength_28d'].values, y_oof)
    rmse_a = np.sqrt(mean_squared_error(df_raw['Strength_28d'].values, y_oof))
    delta  = 0.0 if full_r2 is None else r2_a - full_r2
    if full_r2 is None: full_r2 = r2_a
    ABLATION_RESULTS[cfg_name] = {'R2': r2_a, 'RMSE': rmse_a, 'delta': delta}
    print(f'  {cfg_name:30s} | {len(avail):3d} | {r2_a:7.4f} | {rmse_a:7.3f} | {delta:+7.4f}')

no7_delta = ABLATION_RESULTS.get("Without f'c,7", {}).get('delta', float('nan'))
print(f'\n  KEY: Removing f\'c,7 -> Delta R2 = {no7_delta:+.4f}')


# =============================================================================
# S5b. MULTI-TASK PHYSICS-INFORMED NEURAL NETWORK (MTL-PINN)
#
# Architecture:
#   Input(26) -> Dense(128)+LayerNorm+Dropout -> Dense(64)+LayerNorm+Dropout
#             -> Dense(32) -> 4 task-specific heads (Dense(16)->Dense(1))
#
# Loss:
#   L = w₁·MSE(f'c,28) + w₂·MSE(f'c,7) + w₃·MSE(s30) + w₄·MSE(s90) + λ·L_phys
#   L_phys = mean(relu(ŷ_7d − ŷ_28d))     # Abrams monotonicity: f'c,28 ≥ f'c,7
#
# Output: mtl_pinn_final.h5  (Keras HDF5)
#         mtl_pinn_preprocessing.joblib  (scalers + winsor bounds + freq maps)
#         MC-Dropout (T=100) predictive σ for f'c,28
# =============================================================================

print('\n[S5b] Multi-Task Physics-Informed Neural Network (MTL-PINN)')
print('=' * 60)

# Unified MTL feature set: drop target/derived columns to share inputs across heads
FEAT_MTL = [f for f in FEAT_28D if f not in
            {'Strength_7d', 'Slump_30',
             'Slump_Retention', 'Slump_Loss_Rate'}]
TARGETS_MTL  = ['Strength_28d', 'Strength_7d', 'Slump_30', 'Slump_90']
LOSS_WEIGHTS = {'Strength_28d': 1.0, 'Strength_7d': 0.5,
                'Slump_30'    : 0.3, 'Slump_90'   : 0.3}
PHYS_LAMBDA  = 0.10
DROPOUT_RATE = 0.20

print(f'  Features: {len(FEAT_MTL)} | Targets: {len(TARGETS_MTL)} | '
      f'λ_phys={PHYS_LAMBDA} | dropout={DROPOUT_RATE}')

class AbramsViolationLayer(layers.Layer):
    """relu(ŷ_7d − ŷ_28d): positive when f'c,7 > f'c,28 (Abrams violation).
    Trained to predict zeros so the physics constraint enters the loss."""
    def call(self, inputs):
        out_7d, out_28d = inputs
        return keras.ops.relu(out_7d - out_28d)
    def get_config(self):
        return super().get_config()

def build_mtl_pinn(n_features, dropout=DROPOUT_RATE, l2=1e-5):
    """Multi-task PINN: shared backbone + 4 heads + Abrams auxiliary output."""
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

    out_28d = head('Strength_28d')
    out_7d  = head('Strength_7d')
    out_s30 = head('Slump_30')
    out_s90 = head('Slump_90')

    # Abrams monotonicity: auxiliary output trained to predict 0 via MAE
    # so loss = mean(relu(yhat_7d - yhat_28d)), enforcing f'c,28 >= f'c,7
    abrams_viol = AbramsViolationLayer(name='abrams_viol')([out_7d, out_28d])

    model = Model(inputs=inp,
                  outputs=[out_28d, out_7d, out_s30, out_s90, abrams_viol],
                  name='MTL_PINN')
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss={'Strength_28d': 'mse',  'Strength_7d': 'mse',
              'Slump_30'    : 'mse',  'Slump_90'   : 'mse',
              'abrams_viol' : 'mae'},
        loss_weights={'Strength_28d': LOSS_WEIGHTS['Strength_28d'],
                      'Strength_7d' : LOSS_WEIGHTS['Strength_7d'],
                      'Slump_30'    : LOSS_WEIGHTS['Slump_30'],
                      'Slump_90'    : LOSS_WEIGHTS['Slump_90'],
                      'abrams_viol' : PHYS_LAMBDA},
    )
    return model

def fit_mtl_fold(tr_proc, va_proc, feats, targets,
                 epochs=300, batch_size=32, verbose=0):
    """Fit MTL-PINN on one fold with feature & target standardization."""
    sx = StandardScaler().fit(tr_proc[feats].values)
    sy = {t: StandardScaler().fit(tr_proc[t].values.reshape(-1, 1))
          for t in targets}

    Xtr = sx.transform(tr_proc[feats].values).astype(np.float32)
    Xva = sx.transform(va_proc[feats].values).astype(np.float32)
    ytr = {t: sy[t].transform(tr_proc[t].values.reshape(-1, 1)).ravel()
           for t in targets}
    yva = {t: sy[t].transform(va_proc[t].values.reshape(-1, 1)).ravel()
           for t in targets}
    # Physics target: zeros (abrams_viol minimised toward 0)
    ytr['abrams_viol'] = np.zeros(len(Xtr), dtype=np.float32)
    yva['abrams_viol'] = np.zeros(len(Xva), dtype=np.float32)

    K.clear_session()
    tf.random.set_seed(SEED)
    model = build_mtl_pinn(len(feats))

    cb_es = callbacks.EarlyStopping(monitor='val_loss', patience=40,
                                     restore_best_weights=True, verbose=0)
    cb_rl = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                         patience=20, min_lr=1e-5, verbose=0)
    model.fit(Xtr, ytr, validation_data=(Xva, yva),
              epochs=epochs, batch_size=batch_size,
              callbacks=[cb_es, cb_rl], verbose=verbose)

    preds = model.predict(Xva, verbose=0)   # 5 outputs; skip abrams_viol (idx 4)
    out = {}
    for i, t in enumerate(targets):
        out[t] = sy[t].inverse_transform(preds[i].reshape(-1, 1)).ravel()
    return out, model, sx, sy

# --- 5-fold CV evaluation ----------------------------------------------------
n = len(df_raw)
mtl_oof = {t: np.zeros(n) for t in TARGETS_MTL}
cv_mtl  = KFold(n_splits=5, shuffle=True, random_state=SEED)

for fi, (tr_idx, va_idx) in enumerate(cv_mtl.split(df_raw)):
    tr_proc, va_proc = preprocess_fold(df_raw.iloc[tr_idx], df_raw.iloc[va_idx])
    preds, _, _, _ = fit_mtl_fold(tr_proc, va_proc, FEAT_MTL, TARGETS_MTL)
    for t in TARGETS_MTL:
        mtl_oof[t][va_idx] = preds[t]
    print(f'  Fold {fi+1}/5 trained.')

MTL_RESULTS = {}
print(f'\n  {"Target":15s} | {"R²":>7s} | {"RMSE":>7s} | {"MAE":>7s} | {"MAPE%":>7s}')
print(f'  {"-"*55}')
for t in TARGETS_MTL:
    y_true = df_raw[t].values.astype(np.float32)
    m = calc_metrics(y_true, mtl_oof[t])
    MTL_RESULTS[t] = m
    print(f'  {t:15s} | {m["R2"]:7.4f} | {m["RMSE"]:7.3f} | {m["MAE"]:7.3f} | {m["MAPE"]:7.2f}')

pd.DataFrame({**{f'true_{t}'  : df_raw[t].values for t in TARGETS_MTL},
              **{f'pred_{t}'  : mtl_oof[t]       for t in TARGETS_MTL}}
            ).to_csv(os.path.join(OUT_DIR, 'mtl_pinn_oof_predictions.csv'),
                     index=False)

# --- Final model on full dataset (save .h5) ----------------------------------
print('\n  Training final MTL-PINN on full dataset...')
bounds_full_mtl = fit_winsor_bounds(df_raw)
df_full_mtl = apply_winsor_bounds(df_raw, bounds_full_mtl)
df_full_mtl = engineer_features(df_full_mtl)
fb_mtl = df_raw['Cement'].value_counts(normalize=True)
fa_mtl = df_raw['Admixture'].value_counts(normalize=True)
df_full_mtl['Cement_Freq'] = df_raw['Cement'].map(fb_mtl).fillna(0.0).values
df_full_mtl['Admix_Freq']  = df_raw['Admixture'].map(fa_mtl).fillna(0.0).values

sx_final = StandardScaler().fit(df_full_mtl[FEAT_MTL].values)
sy_final = {t: StandardScaler().fit(df_full_mtl[t].values.reshape(-1, 1))
            for t in TARGETS_MTL}
Xf  = sx_final.transform(df_full_mtl[FEAT_MTL].values).astype(np.float32)
yf  = {t: sy_final[t].transform(df_full_mtl[t].values.reshape(-1, 1)).ravel()
       for t in TARGETS_MTL}
yf['abrams_viol'] = np.zeros(len(Xf), dtype=np.float32)

K.clear_session()
tf.random.set_seed(SEED)
mtl_final = build_mtl_pinn(len(FEAT_MTL))
cb_es_f = callbacks.EarlyStopping(monitor='loss', patience=60,
                                   restore_best_weights=True, verbose=0)
cb_rl_f = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5,
                                       patience=30, min_lr=1e-5, verbose=0)
mtl_final.fit(Xf, yf, epochs=600, batch_size=32,
              callbacks=[cb_es_f, cb_rl_f], verbose=0)

MTL_H5_PATH = os.path.join(OUT_DIR, 'mtl_pinn_final.h5')
mtl_final.save(MTL_H5_PATH, save_format='h5')
print(f'  Final model saved: {os.path.basename(MTL_H5_PATH)}')

joblib.dump({'feature_scaler' : sx_final,
             'target_scalers' : sy_final,
             'features'       : FEAT_MTL,
             'targets'        : TARGETS_MTL,
             'winsor_bounds'  : bounds_full_mtl,
             'cement_freq'    : fb_mtl.to_dict(),
             'admix_freq'     : fa_mtl.to_dict(),
             'loss_weights'   : LOSS_WEIGHTS,
             'phys_lambda'    : PHYS_LAMBDA,
             'dropout_rate'   : DROPOUT_RATE},
            os.path.join(OUT_DIR, 'mtl_pinn_preprocessing.joblib'))
print(f'  Preprocessing artifacts saved: mtl_pinn_preprocessing.joblib')

# --- MC Dropout uncertainty (T=100 forward passes, dropout active) -----------
T_MC = 100
print(f'\n  MC Dropout uncertainty (T={T_MC}) for f\'c,28...')
mc_preds = np.zeros((T_MC, n))
for ti in range(T_MC):
    p = mtl_final(Xf, training=True)          # dropout ON at inference
    p_28d = np.array(p[0]).ravel()            # Keras 3: convert KerasTensor safely
    mc_preds[ti] = sy_final['Strength_28d'].inverse_transform(
        p_28d.reshape(-1, 1)).ravel()
mc_mean = mc_preds.mean(axis=0)
mc_std  = mc_preds.std(axis=0)
y_true_28 = df_raw['Strength_28d'].values.astype(np.float32)
pi_lo = mc_mean - 1.96 * mc_std
pi_hi = mc_mean + 1.96 * mc_std
coverage = float(np.mean((y_true_28 >= pi_lo) & (y_true_28 <= pi_hi)) * 100)

print(f'  Mean predictive σ : {mc_std.mean():.3f} MPa')
print(f'  95% PI coverage   : {coverage:.1f}%')

pd.DataFrame({'y_true'  : y_true_28,
              'mc_mean' : mc_mean,
              'mc_std'  : mc_std,
              'pi_lower': pi_lo,
              'pi_upper': pi_hi}).to_csv(
    os.path.join(OUT_DIR, 'mtl_pinn_mc_dropout.csv'), index=False)
print('  MC dropout CSV saved.')

# --- Comparison vs. tree ensemble (f'c,28) -----------------------------------
ens_r2_28 = ALL_RESULTS['Strength_28d']['standard']['Ensemble']['R2']
mtl_r2_28 = MTL_RESULTS['Strength_28d']['R2']
print(f'\n  Comparison on f\'c,28 (5-fold CV):')
print(f'    Tree Ensemble R² : {ens_r2_28:.4f}')
print(f'    MTL-PINN     R² : {mtl_r2_28:.4f}')
print(f'    ΔR² (PINN−Ens)  : {mtl_r2_28 - ens_r2_28:+.4f}')


# =============================================================================
# S6. BOOTSTRAP OOB CONFIDENCE INTERVALS (B=200, CatBoost)
#
# Winsorisation and frequency encoding are fold-local inside each bootstrap.
# =============================================================================

print('\n[S6] Bootstrap OOB CI (B=200, CatBoost)')

def bootstrap_oob(df, target='Strength_28d', feat_list=FEAT_28D,
                  B=200, seed=SEED):
    """B-replicate OOB bootstrap for CatBoost.
    Each replicate uses fold-local preprocessing; winsor bounds and frequency
    encodings are fitted on the bootstrap sample only."""
    rng = np.random.default_rng(seed)
    n = len(df)
    R2_list, RMSE_list, MAE_list = [], [], []

    for b in range(B):
        idx_in  = rng.integers(0, n, size=n)
        oob_idx = np.setdiff1d(np.arange(n), np.unique(idx_in))
        if len(oob_idx) < 20:
            continue

        tr_raw = df.iloc[idx_in]
        va_raw = df.iloc[oob_idx]

        tr_proc, va_proc = preprocess_fold(tr_raw, va_raw)
        avail = [f for f in feat_list if f in tr_proc.columns]

        Xtr = tr_proc[avail].values.astype(np.float32)
        ytr = tr_proc[target].values.astype(np.float32)
        Xva = va_proc[avail].values.astype(np.float32)
        yva = va_proc[target].values.astype(np.float32)

        cb = CatBoostRegressor(**CB_ABLATION_PARAMS, random_seed=seed + b, verbose=0)
        cb.fit(Xtr, ytr, eval_set=(Xva, yva),
               early_stopping_rounds=30, verbose=0)
        yhat = cb.predict(Xva)

        R2_list.append(r2_score(yva, yhat))
        RMSE_list.append(np.sqrt(mean_squared_error(yva, yhat)))
        MAE_list.append(mean_absolute_error(yva, yhat))

    return (np.array(R2_list), np.array(RMSE_list), np.array(MAE_list))

oob_r2, oob_rmse, oob_mae = bootstrap_oob(df_raw)

def ci95(arr):
    return np.percentile(arr, 2.5), np.median(arr), np.percentile(arr, 97.5)

r2_lo,   r2_med,   r2_hi   = ci95(oob_r2)
rmse_lo, rmse_med, rmse_hi = ci95(oob_rmse)
mae_lo,  mae_med,  mae_hi  = ci95(oob_mae)

print(f'  R²:   median={r2_med:.4f}  95% CI=[{r2_lo:.4f}, {r2_hi:.4f}]')
print(f'  RMSE: median={rmse_med:.3f}  95% CI=[{rmse_lo:.3f}, {rmse_hi:.3f}]')
print(f'  MAE:  median={mae_med:.3f}  95% CI=[{mae_lo:.3f}, {mae_hi:.3f}]')

pd.DataFrame({'R2': oob_r2, 'RMSE': oob_rmse, 'MAE': oob_mae}).to_csv(
    os.path.join(OUT_DIR, 'bootstrap_catboost_fc28.csv'), index=False)


# =============================================================================
# S7. TREESHAP EXPLAINABILITY & STABILITY
# =============================================================================

print('\n[S7] TreeSHAP Explainability')

# Fit CatBoost on the full preprocessed dataset for SHAP explanation.
bounds_full = fit_winsor_bounds(df_raw)
df_full_proc = apply_winsor_bounds(df_raw, bounds_full)
df_full_proc = engineer_features(df_full_proc)
fb_full = df_raw['Cement'].value_counts(normalize=True)
fa_full = df_raw['Admixture'].value_counts(normalize=True)
df_full_proc['Cement_Freq'] = df_raw['Cement'].map(fb_full).fillna(0.0).values
df_full_proc['Admix_Freq']  = df_raw['Admixture'].map(fa_full).fillna(0.0).values

X_shap = df_full_proc[FEAT_28D].values.astype(np.float32)
y_shap = df_raw['Strength_28d'].values.astype(np.float32)

cb_full = CatBoostRegressor(**CB_ABLATION_PARAMS, random_seed=SEED, verbose=0)
cb_full.fit(X_shap, y_shap)
explainer_cb = shap.TreeExplainer(cb_full)
shap_values_cb = explainer_cb.shap_values(X_shap)

xgb_full = xgb.XGBRegressor(n_estimators=800, max_depth=6, learning_rate=0.05,
                              random_state=SEED, n_jobs=-1, verbosity=0)
xgb_full.fit(X_shap, y_shap)
explainer_xgb = shap.TreeExplainer(xgb_full)
shap_values_xgb = explainer_xgb.shap_values(X_shap)

mean_shap = np.abs(shap_values_cb).mean(axis=0)
shap_imp = pd.DataFrame({'Feature': FEAT_28D, 'Mean_SHAP': mean_shap}
                        ).sort_values('Mean_SHAP', ascending=False)

print('  Top 10 features by mean |SHAP| (CatBoost):')
for rank, (_, row) in enumerate(shap_imp.head(10).iterrows(), 1):
    print(f'    {rank:2d}. {row["Feature"]:25s} {row["Mean_SHAP"]:.4f} MPa')

# SHAP stability across 5 folds
cv_shap = KFold(n_splits=5, shuffle=True, random_state=SEED)
fold_rankings = []
for tr_idx, va_idx in cv_shap.split(df_raw):
    tr_proc, va_proc = preprocess_fold(df_raw.iloc[tr_idx], df_raw.iloc[va_idx])
    Xtr_s = tr_proc[FEAT_28D].values.astype(np.float32)
    ytr_s = tr_proc['Strength_28d'].values.astype(np.float32)
    Xva_s = va_proc[FEAT_28D].values.astype(np.float32)
    m_s = CatBoostRegressor(**CB_ABLATION_PARAMS, random_seed=SEED, verbose=0)
    m_s.fit(Xtr_s, ytr_s)
    sv = shap.TreeExplainer(m_s).shap_values(Xva_s)
    top5 = [FEAT_28D[j] for j in np.argsort(-np.abs(sv).mean(0))[:5]]
    fold_rankings.append(top5)

rank1_unanimous = len({r[0] for r in fold_rankings}) == 1
tau_vals = [len(set(fold_rankings[i]) & set(fold_rankings[j])) / 5.0
            for i in range(5) for j in range(i + 1, 5)]
print(f'  Rank-1 unanimous: {rank1_unanimous} | Top-5 overlap: {np.mean(tau_vals):.2f}')


# =============================================================================
# S8. SUPPLEMENTARY TABLES
#     S1: Winsorisation sensitivity
#     S2: LOCO (leave-one-class-out) decomposition
# =============================================================================

print('\n[S8] Supplementary Tables')

def quick_cv_r2(df, target, feat_list, cb_params=CB_ABLATION_PARAMS, k=5,
                lo=0.01, hi=0.99):
    """Fast 5-fold CV R² with fold-local preprocessing."""
    n = len(df)
    y_oof = np.zeros(n)
    cv = KFold(n_splits=k, shuffle=True, random_state=SEED)
    for tr_idx, va_idx in cv.split(df):
        tr_proc, va_proc = preprocess_fold(df.iloc[tr_idx], df.iloc[va_idx],
                                           lo=lo, hi=hi)
        avail = [f for f in feat_list if f in tr_proc.columns]
        Xtr = tr_proc[avail].values.astype(np.float32)
        ytr = tr_proc[target].values.astype(np.float32)
        Xva = va_proc[avail].values.astype(np.float32)
        m = CatBoostRegressor(**cb_params, random_seed=SEED, verbose=0)
        m.fit(Xtr, ytr, eval_set=(Xva, va_proc[target].values.astype(np.float32)),
              early_stopping_rounds=50, verbose=0)
        y_oof[va_idx] = m.predict(Xva)
    return r2_score(df[target].values, y_oof)

# Table S1: Winsorisation sensitivity
S1_rows = []
for lo, hi in [(0.005, 0.995), (0.01, 0.99), (0.02, 0.98)]:
    r2_val = quick_cv_r2(df_raw, 'Strength_28d', FEAT_28D, lo=lo, hi=hi)
    S1_rows.append({'lo': lo, 'hi': hi, 'R2_Std5Fold': r2_val})
TableS1 = pd.DataFrame(S1_rows)
TableS1.to_csv(os.path.join(OUT_DIR, 'TableS1_winsorisation_sensitivity.csv'), index=False)
delta_winsor = TableS1['R2_Std5Fold'].max() - TableS1['R2_Std5Fold'].min()
print(f'  Table S1: Winsorisation ΔR² across settings = {delta_winsor:.5f}')

# Table S2: LOCO decomposition
S2_rows = []
for cls in sorted(df_raw['Concrete_Class'].unique()):
    sub = df_raw[df_raw['Concrete_Class'] != cls]
    if sub['Concrete_Class'].nunique() < 2:
        continue
    r2_std = quick_cv_r2(sub, 'Strength_28d', FEAT_28D)
    # GroupKFold on subset
    n_sub = len(sub)
    y_oof_g = np.zeros(n_sub)
    gkf = GroupKFold(n_splits=min(5, sub['Concrete_Class'].nunique()))
    for tr_idx, va_idx in gkf.split(sub, groups=sub['Concrete_Class'].values):
        tr_proc, va_proc = preprocess_fold(sub.iloc[tr_idx], sub.iloc[va_idx])
        avail = [f for f in FEAT_28D if f in tr_proc.columns]
        Xtr = tr_proc[avail].values.astype(np.float32)
        ytr = tr_proc['Strength_28d'].values.astype(np.float32)
        Xva = va_proc[avail].values.astype(np.float32)
        m = CatBoostRegressor(**CB_ABLATION_PARAMS, random_seed=SEED, verbose=0)
        m.fit(Xtr, ytr, eval_set=(Xva, va_proc['Strength_28d'].values.astype(np.float32)),
              early_stopping_rounds=50, verbose=0)
        y_oof_g[va_idx] = m.predict(Xva)
    r2_gkf = r2_score(sub['Strength_28d'].values, y_oof_g)
    S2_rows.append({
        'held_out_class': cls, 'n_train': n_sub,
        'R2_Std5Fold': r2_std, 'R2_GroupKFold': r2_gkf,
        'DeltaR2': r2_std - r2_gkf,
    })
TableS2 = pd.DataFrame(S2_rows)
TableS2.to_csv(os.path.join(OUT_DIR, 'TableS2_LOCO_decomposition.csv'), index=False)
print('  Table S2 (LOCO):')
print(TableS2.to_string(index=False))
print(f'  Mean LOCO ΔR² (range-contraction proxy): {TableS2["DeltaR2"].mean():.4f}')
print(f'  LOCO ΔR² > full GroupKFold ΔR² indicates range-contraction is a')
print(f'  major contributor to the gap; exact % attribution is not claimed.')


# =============================================================================
# S9. STATISTICAL SUPPLEMENT
# =============================================================================

print('\n[S9] Statistical Supplement')

yt = ALL_OOF['Strength_28d']['standard']['y']
yp = ALL_OOF['Strength_28d']['standard']['Ensemble']
residuals = yt - yp

# A. Residual diagnostics
stat_sw, p_sw = shapiro(residuals)
stat_dp, p_dp = normaltest(residuals)
_, p_bp, _, _ = het_breuschpagan(
    residuals, np.column_stack([yp.reshape(-1, 1), np.ones(len(yp))]))
dw = durbin_watson(residuals)
print(f'  Shapiro-Wilk: p={p_sw:.2e} -> {"NORMAL" if p_sw > 0.05 else "NON-NORMAL"}')
print(f'  Breusch-Pagan: p={p_bp:.4f} -> {"HOMOSCEDASTIC" if p_bp > 0.05 else "HETEROSCEDASTIC"}')
print(f'  Durbin-Watson: DW={dw:.4f}')
print(f'  Residuals within ±5 MPa: {(np.abs(residuals) <= 5).mean()*100:.1f}%')

# B. Prediction interval calibration
print('\n  Prediction Interval Calibration:')
res_std = residuals.std()
print(f'  {"Nominal":>8s} | {"z":>5s} | {"±Interval":>12s} | {"Actual":>8s} | {"Cal":>3s}')
for nom, z in [(0.50, 0.674), (0.80, 1.282), (0.90, 1.645), (0.95, 1.960), (0.99, 2.576)]:
    lower = yp - z * res_std
    upper = yp + z * res_std
    cov = ((yt >= lower) & (yt <= upper)).mean()
    cal = 'Y' if abs(cov - nom) < 0.05 else 'N'
    print(f'  {int(nom*100):7d}% | {z:5.3f} | ±{z*res_std:5.2f} MPa   | {cov:7.1%} | {cal}')

# C. Generalization gap significance
ae_std = np.abs(ALL_OOF['Strength_28d']['standard']['y'] -
                ALL_OOF['Strength_28d']['standard']['Ensemble'])
ae_grp = np.abs(ALL_OOF['Strength_28d']['group']['y'] -
                ALL_OOF['Strength_28d']['group']['Ensemble'])
stat_gap, p_gap = wilcoxon(ae_std, ae_grp)
diff = ae_grp - ae_std
cohens_d = diff.mean() / diff.std()
print(f"\n  Gap significance: Wilcoxon p={p_gap:.2e}, Cohen's d={cohens_d:.4f}")

# D. VIF
X_vif = df_full_proc[FEAT_28D].values.astype(np.float64)
X_vif_c = np.column_stack([X_vif, np.ones(len(X_vif))])
vif_vals = []
for i in range(len(FEAT_28D)):
    try:
        v = variance_inflation_factor(X_vif_c, i)
    except Exception:
        v = np.inf
    vif_vals.append(v)
n_high_vif = sum(v > 10 for v in vif_vals)
print(f'  VIF > 10: {n_high_vif}/{len(FEAT_28D)} features (tree models robust to collinearity)')

# E. Influential observations
std_res = residuals / residuals.std()
n_outliers = (np.abs(std_res) > 3.0).sum()
print(f'  Influential obs (|std resid| > 3): {n_outliers}/{len(residuals)} ')
print(f'  ({n_outliers/len(residuals)*100:.1f}% — acceptable for field data)')


# =============================================================================
# S10. ACI 318-19 / EUROCODE 2 CODE COMPLIANCE
# =============================================================================

print('\n[S10] Code Compliance')

def compute_aci_fcr(fck, s):
    """ACI 318-19 Table 26.12.1.1 required average strength f'cr (MPa)."""
    if fck < 21:
        return fck + 7.0
    elif fck <= 35:
        return max(fck + 1.34 * s, fck + 2.33 * s - 3.45)
    else:
        return max(fck + 1.34 * s, 0.90 * fck + 2.33 * s)

y_true_cc = df_raw['Strength_28d'].values.astype(np.float64)
y_pred_std = ALL_OOF['Strength_28d']['standard']['Ensemble']
y_pred_gkf = ALL_OOF['Strength_28d']['group']['Ensemble']

compliance = []
for cls in sorted(df_raw['Concrete_Class'].unique().astype(int)):
    mask = df_raw['Concrete_Class'].astype(int) == cls
    y_c = y_true_cc[mask]; p_s = y_pred_std[mask]; p_g = y_pred_gkf[mask]
    n_c = mask.sum(); fck = float(cls)
    s_m = float(y_c.std()) if n_c > 2 else 5.0
    fcr = compute_aci_fcr(fck, s_m)
    fcm = fck + 8.0
    compliance.append({
        'Class': cls, 'n': n_c, 'f_ck': fck,
        'f_cr_ACI': fcr, 'f_cm_EC2': fcm,
        's_meas': s_m, 'y_mean': float(y_c.mean()), 'y_std': float(y_c.std()),
        'ACI_m': np.mean(y_c >= fcr) * 100,
        'EC2_m': np.mean(y_c >= fcm) * 100,
        'ACI_p': np.mean(p_s >= fcr) * 100,
        'EC2_p': np.mean(p_s >= fcm) * 100,
        'ACI_g': np.mean(p_g >= fcr) * 100,
    })
code_df = pd.DataFrame(compliance)
code_df.to_csv(os.path.join(OUT_DIR, 'code_compliance_results.csv'), index=False)
print(code_df[['Class', 'n', 'f_ck', 'f_cr_ACI', 'ACI_m', 'EC2_m', 'ACI_p']].to_string(index=False))


# =============================================================================
# S11. MULTI-OBJECTIVE PARETO OPTIMISATION (CONSTRAINED)
#
# Optimisation is restricted to true design variables (Cement_Content and
# WC_Ratio). Strength_7d and Slump_30 are not treated as free design variables;
# they are imputed from the observed joint distribution conditioned on
# Cement_Content and WC_Ratio using a simple regression proxy.
# =============================================================================

print('\n[S11] Constrained Pareto Optimisation')

# Fit proxy models for dependent variables (Strength_7d, Slump_30)
# from true design variables (Cement_Content, WC_Ratio, Admixture_Dose)
proxy_feats = ['Cement_Content', 'WC_Ratio', 'Admixture_Dose']
proxy_X = df_full_proc[proxy_feats].values.astype(np.float32)

proxy_f7 = Ridge(alpha=1.0)
proxy_f7.fit(proxy_X, df_raw['Strength_7d'].values)

proxy_slump = Ridge(alpha=1.0)
proxy_slump.fit(proxy_X, df_raw['Slump_30'].values)

# Monte Carlo sampling over TRUE design variables only
N_MC = 10000
rng_mc = np.random.default_rng(SEED)

mc_cement = rng_mc.uniform(
    float(df_raw['Cement_Content'].quantile(0.05)),
    float(df_raw['Cement_Content'].quantile(0.95)), N_MC)
mc_wc = rng_mc.uniform(
    float(df_raw['WC_Ratio'].quantile(0.05)),
    float(df_raw['WC_Ratio'].quantile(0.95)), N_MC)
mc_admix = rng_mc.uniform(
    float(df_raw['Admixture_Dose'].quantile(0.05)),
    float(df_raw['Admixture_Dose'].quantile(0.95)), N_MC)

# Impute dependent variables from proxy models
proxy_input = np.column_stack([mc_cement, mc_wc, mc_admix]).astype(np.float32)
mc_f7    = proxy_f7.predict(proxy_input).clip(
    float(df_raw['Strength_7d'].min()), float(df_raw['Strength_7d'].max()))
mc_slump = proxy_slump.predict(proxy_input).clip(
    float(df_raw['Slump_30'].min()), float(df_raw['Slump_30'].max()))

# Use median values for non-design variables
mc_cls = np.full(N_MC, float(df_raw['Concrete_Class'].median()))
mc_ca  = np.full(N_MC, float(df_raw['CA'].median()))
mc_fa  = np.full(N_MC, float(df_raw['FA'].median()))

# Build feature matrix using the same engineer_features function
mc_df = pd.DataFrame({
    'Concrete_Class': mc_cls, 'CA': mc_ca, 'FA': mc_fa,
    'Cement_Content': mc_cement, 'WC_Ratio': mc_wc,
    'Admixture_Dose': mc_admix, 'Strength_7d': mc_f7, 'Slump_30': mc_slump,
    'Slump_60': np.full(N_MC, float(df_raw['Slump_60'].median())),
    'Slump_90': np.full(N_MC, float(df_raw['Slump_90'].median())),
    'Cement': 'Median', 'Admixture': 'Median',
})
mc_df = engineer_features(mc_df)
mc_df['Cement_Freq'] = float(df_full_proc['Cement_Freq'].median())
mc_df['Admix_Freq']  = float(df_full_proc['Admix_Freq'].median())

pred_s = cb_full.predict(mc_df[FEAT_28D].values.astype(np.float32))

# Physical feasibility constraints
water = mc_cement * mc_wc
feasible = (pred_s > 20) & (water >= 120) & (water <= 220) & (mc_wc >= 0.25)

cement_f   = mc_cement[feasible]
strength_f = pred_s[feasible]
wc_f       = mc_wc[feasible]

# Pareto front: maximise strength, minimise cement
sorted_idx = np.argsort(-strength_f)
pareto_mask = np.zeros(len(cement_f), dtype=bool)
min_cem = np.inf
for i in sorted_idx:
    if cement_f[i] < min_cem:
        pareto_mask[i] = True
        min_cem = cement_f[i]

print(f'  Feasible: {feasible.sum():,} | Pareto: {pareto_mask.sum()}')

pareto_df = pd.DataFrame({
    'Cement': cement_f[pareto_mask], 'WC': wc_f[pareto_mask],
    'fc28': strength_f[pareto_mask],
    'Eff': strength_f[pareto_mask] / cement_f[pareto_mask],
}).sort_values('fc28', ascending=False).reset_index(drop=True)
pareto_df.to_csv(os.path.join(OUT_DIR, 'pareto_optimal_mixes.csv'), index=False)
print('  Top 5 Pareto designs:')
print(pareto_df.head(5).to_string(index=False))


# =============================================================================
# S12. FIGURE GENERATION
#
#   Fig 1 : Architecture of MTL-PINN baseline   (generated separately)
#   Fig 2 : Prediction diagnostics
#   Fig 3 : Generalisation gap
#   Fig 4 : Bootstrap distributions
#   Fig 5 : TreeSHAP attribution
#   Fig 6 : Physical-law diagnostics
#   Fig 7 : Code compliance
#   Fig 8 : Pareto-optimal mix designs
#   Fig 9 : Heatmap comparison
#   Fig S1: Diagnostic plots for secondary targets
# =============================================================================

print('\n[S12] Generating figures...')

NAME_MAP = {
    'Strength_7d': '$f_{c,7}$', 'Cement_sq': 'Cement²',
    'Admix_Freq': 'Admix Freq', 'Concrete_Class_sq': 'Class²',
    'Cement_Freq': 'Cement Freq', 'WC_Ratio_sq': '(W/C)²',
    'WC_Ratio': 'W/C', 'Slump_Retention': 'Slump Ret.',
    'log_Cement': 'log(Cement)', 'Cement_Content': 'Cement',
    'Slump_30': 'Slump 30min', 'Bolomey_Feature': 'Bolomey',
    'Slump_Loss_Rate': 'Slump Loss', 'Class_x_Cement': 'Class×Cement',
    'Cement_x_WC': 'Cement×W/C',
}

# Fig 1 (MTL-PINN architecture schematic) is rendered separately from the
# saved Keras graph (mtl_pinn_architecture.svg) and is not generated here.

# --- Figure 2: Prediction diagnostics ----------------------------------------
yt2 = ALL_OOF['Strength_28d']['standard']['y']
yp2 = ALL_OOF['Strength_28d']['standard']['Ensemble']
res2 = yt2 - yp2
r2_v = r2_score(yt2, yp2); rmse_v = np.sqrt(mean_squared_error(yt2, yp2))
mae_v = mean_absolute_error(yt2, yp2); mu2, sig2 = res2.mean(), res2.std()

fig2, axes2 = plt.subplots(1, 3, figsize=(7.2, 2.5))
ax = axes2[0]; panel_label(ax, 'A')
ax.scatter(yt2, yp2, alpha=0.4, s=10, color='#0072B2', edgecolors='none', rasterized=True)
lims = [min(yt2.min(), yp2.min()) * 0.95, max(yt2.max(), yp2.max()) * 1.05]
ax.plot(lims, lims, 'k--', lw=0.8)
sl, ic, rv, _, _ = stats.linregress(yt2, yp2)
xl = np.linspace(lims[0], lims[1], 100)
ax.plot(xl, sl * xl + ic, color='#D55E00', lw=1.0, label=f'OLS (r={rv:.3f})')
ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect('equal')
ax.set_xlabel("Measured $f'_{c,28}$ (MPa)"); ax.set_ylabel("Predicted $f'_{c,28}$ (MPa)")
ax.text(0.05, 0.95, f'R²={r2_v:.4f}\nRMSE={rmse_v:.2f} MPa\nMAE={mae_v:.2f} MPa',
        transform=ax.transAxes, fontsize=6.5, va='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
ax.legend(fontsize=6, loc='lower right'); ax.grid(True, alpha=0.2); clean_spines(ax)

ax = axes2[1]; panel_label(ax, 'B')
ax.hist(res2, bins=35, color='#A7D100', alpha=0.65, edgecolor='white',
        linewidth=0.3, density=True)
xr = np.linspace(res2.min() - 1, res2.max() + 1, 200)
ax.plot(xr, stats.norm.pdf(xr, mu2, sig2), 'k-', lw=1.0, label=f'N({mu2:.2f},{sig2:.2f}²)')
ax.axvline(0, color='#D55E00', lw=1.0, ls='--')
ax.set_xlabel('Residual (MPa)'); ax.set_ylabel('Probability density')
ax.legend(fontsize=6); ax.grid(axis='y', alpha=0.2); clean_spines(ax)

ax = axes2[2]; panel_label(ax, 'C')
ax.scatter(yp2, res2, alpha=0.4, s=10, color='#009E73', edgecolors='none', rasterized=True)
ax.axhline(0, color='k', lw=0.8, ls='--')
ax.axhline(2 * sig2, color='#E69F00', lw=0.7, ls=':')
ax.axhline(-2 * sig2, color='#E69F00', lw=0.7, ls=':')
si = np.argsort(yp2)
ax.plot(yp2[si], uniform_filter1d(res2[si], size=max(20, len(res2) // 25)),
        color='#D55E00', lw=1.2, label='Smooth trend')
ax.set_xlabel("Predicted $f'_{c,28}$ (MPa)"); ax.set_ylabel('Residual (MPa)')
ax.legend(fontsize=6); ax.grid(True, alpha=0.2); clean_spines(ax)
fig2.tight_layout(pad=0.4); save_fig(fig2, 'fig2_prediction_diagnostics')

# --- Figure 3: Generalisation gap --------------------------------------------
tgt_keys   = ['Strength_28d', 'Strength_7d', 'Slump_30', 'Slump_90']
tgt_labels = ['$f_{c,28}$', '$f_{c,7}$', 'Slump$_{30}$', 'Slump$_{90}$']
r2_std3 = [ALL_RESULTS[t]['standard']['Ensemble']['R2'] for t in tgt_keys]
r2_gkf3 = [ALL_RESULTS[t]['group']['Ensemble']['R2'] for t in tgt_keys]
delta_r2_3 = [s - g for s, g in zip(r2_std3, r2_gkf3)]

def eta_squared(df, cat_col, target_col):
    grand_mean = df[target_col].mean()
    groups = df.groupby(cat_col)[target_col]
    ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for _, g in groups)
    ss_total = sum((df[target_col] - grand_mean)**2)
    return ss_between / ss_total if ss_total > 0 else 0

eta2_vals = [eta_squared(df_raw, 'Concrete_Class', t) for t in tgt_keys]
max_r3 = []
for t in tgt_keys:
    feats = FEAT_28D if t.startswith('Strength') else FEAT_SLUMP
    feats_clean = [f for f in feats if f in df_full_proc.columns and f != t]
    max_r3.append(float(df_full_proc[feats_clean].corrwith(df_raw[t]).abs().max()))

fig3, axes3 = plt.subplots(1, 2, figsize=(7.2, 3.2))
ax = axes3[0]; panel_label(ax, 'A')
xp = np.arange(len(tgt_labels)); bw3 = 0.32
bars1 = ax.bar(xp - bw3/2, r2_std3, bw3, color='#0072B2', alpha=0.85,
               edgecolor='#333', lw=0.4, label='Standard 5-Fold CV')
bars2 = ax.bar(xp + bw3/2, r2_gkf3, bw3, color='#E69F00', alpha=0.85,
               edgecolor='#333', lw=0.4, label='GroupKFold CV')
for bar, val in zip(bars1, r2_std3):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.012,
            f'{val:.3f}', ha='center', va='bottom', fontsize=6.5, fontweight='bold')
for bar, val in zip(bars2, r2_gkf3):
    y_pos = max(val + 0.012, 0.03) if val >= 0 else 0.03
    ax.text(bar.get_x() + bar.get_width()/2, y_pos,
            f'{val:.3f}', ha='center', va='bottom', fontsize=6.5, fontweight='bold',
            color='#CC0000' if val < 0.1 else '#333')
for i, dr in enumerate(delta_r2_3):
    y_top = max(r2_std3[i], r2_gkf3[i]) + 0.06
    ax.annotate(f'ΔR²={dr:.2f}', xy=(i, y_top), fontsize=7, ha='center',
                color='#CC0000' if dr > 0.5 else '#E69F00', fontweight='bold')
ax.axhline(0, color='grey', lw=0.4)
ax.set_xticks(xp); ax.set_xticklabels(tgt_labels, fontsize=10)
ax.set_ylabel('R²'); ax.set_ylim(-0.05, 1.15)
ax.legend(fontsize=7, loc='upper right'); ax.grid(axis='y', alpha=0.3); clean_spines(ax)

ax = axes3[1]; panel_label(ax, 'B')
x3 = np.arange(len(tgt_labels)); w3 = 0.22
ax.bar(x3 - w3, eta2_vals, w3, color='#009E73', alpha=0.8, edgecolor='#333', lw=0.4, label='η² (class effect)')
ax.bar(x3, max_r3, w3, color='#CC79A7', alpha=0.8, edgecolor='#333', lw=0.4, label='Max |r| with target')
ax.bar(x3 + w3, delta_r2_3, w3, color='#D55E00', alpha=0.8, edgecolor='#333', lw=0.4, label='ΔR² (gen. gap)')
ax.set_xticks(x3); ax.set_xticklabels(tgt_labels, fontsize=10)
ax.set_ylabel('Value'); ax.legend(fontsize=6.5, loc='upper left')
ax.grid(axis='y', alpha=0.3); clean_spines(ax)
fig3.tight_layout(pad=0.5); save_fig(fig3, 'fig3_generalisation_gap')

# --- Figure 4: Bootstrap distributions ---------------------------------------
COLORS4 = ['#0072B2', '#009E73', '#CC79A7']
fig4, axes4 = plt.subplots(1, 3, figsize=(7.2, 2.5))
for ax_i, (data, label, unit) in enumerate([
    (oob_r2, 'R²', ''), (oob_rmse, 'RMSE', ' MPa'), (oob_mae, 'MAE', ' MPa')]):
    col = COLORS4[ax_i]; ax = axes4[ax_i]; panel_label(ax, chr(65 + ax_i))
    lo4, med4, hi4 = np.percentile(data, 2.5), np.median(data), np.percentile(data, 97.5)
    ax.hist(data, bins=30, color=col, alpha=0.55, edgecolor='white', lw=0.3, density=True)
    kde = gaussian_kde(data); xk = np.linspace(data.min(), data.max(), 200)
    ax.plot(xk, kde(xk), color='k', lw=1.0)
    ax.axvline(med4, color='#D55E00', lw=1.2, ls='-',
               label=f'Median={med4:.4f}' if ax_i == 0 else f'Median={med4:.3f}')
    ax.axvline(lo4, color='#E69F00', lw=1.0, ls='--',
               label=f'2.5%={lo4:.4f}' if ax_i == 0 else f'2.5%={lo4:.3f}')
    ax.axvline(hi4, color='#E69F00', lw=1.0, ls='--',
               label=f'97.5%={hi4:.4f}' if ax_i == 0 else f'97.5%={hi4:.3f}')
    ax.axvspan(lo4, hi4, alpha=0.12, color=col)
    ax.set_xlabel(f'{label}{unit}'); ax.set_ylabel('Density')
    ax.legend(fontsize=5.5, loc='upper left'); ax.grid(axis='y', alpha=0.2); clean_spines(ax)
fig4.tight_layout(pad=0.4); save_fig(fig4, 'fig4_bootstrap_distributions')

# --- Figure 5: TreeSHAP attribution ------------------------------------------
fig5 = plt.figure(figsize=(7.2, 5.0))
gs5 = gridspec.GridSpec(2, 3, figure=fig5, hspace=0.45, wspace=0.40)
X_df = df_full_proc[FEAT_28D]

imp_order = pd.Series(np.abs(shap_values_cb).mean(0), index=FEAT_28D).sort_values()
top12 = imp_order.tail(12).index.tolist()

ax = fig5.add_subplot(gs5[0, :2]); panel_label(ax, 'A', x=-0.06)
for i, feat in enumerate(top12):
    fidx = FEAT_28D.index(feat)
    sv = shap_values_cb[:, fidx]; fv = X_df[feat].values
    fv_norm = (fv - fv.min()) / (fv.max() - fv.min() + 1e-9)
    jitter = np.random.RandomState(42).uniform(-0.3, 0.3, len(sv))
    sc = ax.scatter(sv, np.full(len(sv), i) + jitter, c=fv_norm,
                    cmap='RdBu_r', alpha=0.45, s=5, vmin=0, vmax=1, rasterized=True)
ax.set_yticks(range(len(top12)))
ax.set_yticklabels([NAME_MAP.get(f, f) for f in top12], fontsize=7.5)
ax.axvline(0, color='k', lw=0.6, ls='--')
ax.set_xlabel('SHAP value (MPa)')
cb5 = plt.colorbar(sc, ax=ax, pad=0.01, shrink=0.85, aspect=25)
cb5.set_label('Feature value (low → high)', fontsize=7); cb5.ax.tick_params(labelsize=6)
clean_spines(ax)

ax = fig5.add_subplot(gs5[0, 2]); panel_label(ax, 'B')
imp_xgb = pd.Series(np.abs(shap_values_xgb).mean(0), index=FEAT_28D).sort_values().tail(12)
colours = plt.cm.viridis(np.linspace(0.2, 0.9, 12))
ax.barh([NAME_MAP.get(f, f) for f in imp_xgb.index], imp_xgb.values,
        color=colours, edgecolor='#333', linewidth=0.3, height=0.7)
ax.set_xlabel('Mean |SHAP| (MPa)'); ax.tick_params(axis='y', labelsize=7); clean_spines(ax)

top3 = imp_order.tail(3).index[::-1].tolist()
for j, feat in enumerate(top3):
    ax = fig5.add_subplot(gs5[1, j]); panel_label(ax, chr(67 + j))
    fidx = FEAT_28D.index(feat)
    sv = shap_values_cb[:, fidx]; fv = X_df[feat].values
    ax.scatter(fv, sv, c=sv, cmap='coolwarm', alpha=0.4, s=6,
               edgecolors='none', rasterized=True)
    z = np.polyfit(fv, sv, 2); xl = np.linspace(fv.min(), fv.max(), 100)
    ax.plot(xl, np.poly1d(z)(xl), 'k-', lw=1.2)
    ax.axhline(0, color='gray', lw=0.6, ls='--')
    ax.set_xlabel(NAME_MAP.get(feat, feat)); ax.set_ylabel('SHAP (MPa)'); clean_spines(ax)

save_fig(fig5, 'fig5_treeshap_attribution')

# --- Figure 6: Physical-law diagnostics --------------------------------------
y_true6 = df_raw['Strength_28d'].values.astype(np.float64)
y_pred6 = ALL_OOF['Strength_28d']['standard']['Ensemble']
residuals6 = y_true6 - y_pred6
classes6 = df_raw['Concrete_Class'].values.astype(int)
wc6 = df_full_proc['WC_Ratio'].values
f7_6 = df_raw['Strength_7d'].values

fig6 = plt.figure(figsize=(7.2, 5.5))
gs6 = gridspec.GridSpec(2, 3, hspace=0.50, wspace=0.42)

ax = fig6.add_subplot(gs6[0, 0]); panel_label(ax, 'A')
ax.scatter(wc6, y_true6, c=classes6, cmap='viridis', s=12, alpha=0.5, edgecolors='none')
valid = (y_true6 > 0) & (wc6 > 0)
sl_a, ic_a, r_a, _, _ = stats.linregress(wc6[valid], np.log(y_true6[valid]))
wc_fit = np.linspace(wc6.min(), wc6.max(), 100)
K1 = np.exp(ic_a); K2 = np.exp(-sl_a)
ax.plot(wc_fit, np.exp(ic_a + sl_a * wc_fit), '--', color='#D55E00', lw=1.5,
        label=f"Abrams': $f_c={K1:.0f}/{K2:.2f}^{{w/c}}$ (r={r_a:.3f})")
ax.set_xlabel('W/C Ratio'); ax.set_ylabel("Measured $f'_{c,28}$ (MPa)")
ax.legend(fontsize=6.5); ax.grid(True, alpha=0.3); clean_spines(ax)

ax = fig6.add_subplot(gs6[0, 1]); panel_label(ax, 'B')
ax.scatter(wc6, y_pred6, c=classes6, cmap='viridis', s=12, alpha=0.5, edgecolors='none')
sl_p, ic_p, r_p, _, _ = stats.linregress(wc6[valid], np.log(np.maximum(y_pred6[valid], 1e-3)))
ax.plot(wc_fit, np.exp(ic_p + sl_p * wc_fit), ':', color='#0072B2', lw=1.2,
        label=f'Model fit (r={r_p:.3f})')
ax.plot(wc_fit, np.exp(ic_a + sl_a * wc_fit), '--', color='#D55E00', lw=1.0,
        label="Abrams' curve")
ax.set_xlabel('W/C Ratio'); ax.set_ylabel("Predicted $f'_{c,28}$ (MPa)")
ax.legend(fontsize=6.5); ax.grid(True, alpha=0.3); clean_spines(ax)

ax = fig6.add_subplot(gs6[0, 2]); panel_label(ax, 'C')
ax.scatter(f7_6, y_true6, c=classes6, cmap='viridis', s=12, alpha=0.5, edgecolors='none')
sl_m, ic_m, r_m, _, _ = stats.linregress(f7_6, y_true6)
x_fit = np.linspace(f7_6.min(), y_true6.max(), 100)
ax.plot(x_fit, sl_m * x_fit + ic_m, '-', color='#D55E00', lw=1.2,
        label=f'$f_{{28}}={sl_m:.2f}f_{{7}}+{ic_m:.1f}$ (r={r_m:.3f})')
ax.plot(x_fit, x_fit, ':', color='grey', lw=0.8, label='1:1 line')
ax.set_xlabel("$f'_{c,7}$ (MPa)"); ax.set_ylabel("$f'_{c,28}$ (MPa)")
ax.legend(fontsize=6.5); ax.grid(True, alpha=0.3); clean_spines(ax)

ax = fig6.add_subplot(gs6[1, 0]); panel_label(ax, 'D')
bi = df_full_proc['Cement_Content'].values / classes6
sc_bi = ax.scatter(bi, y_true6, c=wc6, cmap='RdBu_r', s=12, alpha=0.5, edgecolors='none')
cb6 = fig6.colorbar(sc_bi, ax=ax, shrink=0.8, pad=0.02)
cb6.set_label('W/C', fontsize=7); cb6.ax.tick_params(labelsize=6)
ax.set_xlabel('Binder Intensity (kg/m³ per class)'); ax.set_ylabel("$f'_{c,28}$ (MPa)")
ax.grid(True, alpha=0.3); clean_spines(ax)

ax = fig6.add_subplot(gs6[1, 1]); panel_label(ax, 'E')
classes_sorted = sorted(np.unique(classes6))
bp_data = [residuals6[classes6 == c] for c in classes_sorted]
bp = ax.boxplot(bp_data, labels=[f'M{c}' for c in classes_sorted],
                patch_artist=True, widths=0.6,
                medianprops=dict(color='#D55E00', linewidth=1.2),
                flierprops=dict(marker='o', markersize=2, alpha=0.4))
for p, c in zip(bp['boxes'], plt.cm.viridis(np.linspace(0.2, 0.8, len(classes_sorted)))):
    p.set_facecolor(c); p.set_alpha(0.7)
ax.axhline(0, color='grey', linestyle='--', linewidth=0.7)
ax.set_xlabel('Concrete Class'); ax.set_ylabel('Residual (MPa)')
ax.tick_params(axis='x', rotation=45, labelsize=7)
ax.grid(axis='y', alpha=0.3); clean_spines(ax)

ax = fig6.add_subplot(gs6[1, 2]); panel_label(ax, 'F')
gsr = df_full_proc['Gel_Space_Ratio_28d'].values
ax.scatter(gsr, y_true6, c=classes6, cmap='viridis', s=12, alpha=0.5, edgecolors='none')
valid_gsr = (gsr > 0) & (y_true6 > 0)
sl_pwr, ic_pwr, r_pwr, _, _ = stats.linregress(
    np.log(gsr[valid_gsr]), np.log(y_true6[valid_gsr]))
gsr_fit = np.linspace(gsr.min(), gsr.max(), 100)
ax.plot(gsr_fit, np.exp(ic_pwr) * gsr_fit ** sl_pwr, '--', color='#D55E00', lw=1.5,
        label=f"Powers': $f_c={np.exp(ic_pwr):.0f}X^{{{sl_pwr:.2f}}}$ (r={r_pwr:.3f})")
ax.set_xlabel("Gel-Space Ratio (Powers')"); ax.set_ylabel("$f'_{c,28}$ (MPa)")
ax.legend(fontsize=6.5); ax.grid(True, alpha=0.3); clean_spines(ax)

fig6.tight_layout(pad=0.5); save_fig(fig6, 'fig6_physical_law_diagnostics')

# --- Figure 7: Code compliance -----------------------------------------------
fig7, axes7 = plt.subplots(1, 3, figsize=(7.2, 2.8))
x_cls = np.arange(len(code_df)); bw = 0.30

ax = axes7[0]; panel_label(ax, 'A')
ax.bar(x_cls - bw/2, code_df['f_cr_ACI'], bw, color='#CCC', edgecolor='#666', lw=0.5, label="ACI $f'_{cr}$")
ax.bar(x_cls + bw/2, code_df['y_mean'], bw, color='#0072B2', alpha=0.8, edgecolor='#333', lw=0.5, label='Measured mean')
ax.errorbar(x_cls + bw/2, code_df['y_mean'], yerr=code_df['y_std'], fmt='none', color='#333', lw=0.7, capsize=2)
ax.set_xticks(x_cls); ax.set_xticklabels([f'M{int(c)}' for c in code_df['Class']], fontsize=7)
ax.set_ylabel('Strength (MPa)'); ax.set_xlabel('Concrete Class')
ax.legend(fontsize=6); ax.grid(axis='y', alpha=0.3); clean_spines(ax)

ax = axes7[1]; panel_label(ax, 'B')
ax.bar(x_cls - bw/2, code_df['f_cm_EC2'], bw, color='#CCC', edgecolor='#666', lw=0.5, label='EC2 $f_{cm}$')
ax.bar(x_cls + bw/2, code_df['y_mean'], bw, color='#009E73', alpha=0.8, edgecolor='#333', lw=0.5, label='Measured mean')
ax.errorbar(x_cls + bw/2, code_df['y_mean'], yerr=code_df['y_std'], fmt='none', color='#333', lw=0.7, capsize=2)
ax.set_xticks(x_cls); ax.set_xticklabels([f'M{int(c)}' for c in code_df['Class']], fontsize=7)
ax.set_ylabel('Strength (MPa)'); ax.set_xlabel('Concrete Class')
ax.legend(fontsize=6); ax.grid(axis='y', alpha=0.3); clean_spines(ax)

ax = axes7[2]; panel_label(ax, 'C'); bw2 = 0.18
ax.bar(x_cls - bw2*1.5, code_df['ACI_m'], bw2, color='#0072B2', alpha=0.85, label='ACI (meas.)')
ax.bar(x_cls - bw2*0.5, code_df['ACI_p'], bw2, color='#0072B2', alpha=0.4, hatch='//', edgecolor='#0072B2', label='ACI (pred.)')
ax.bar(x_cls + bw2*0.5, code_df['EC2_m'], bw2, color='#009E73', alpha=0.85, label='EC2 (meas.)')
ax.bar(x_cls + bw2*1.5, code_df['EC2_p'], bw2, color='#009E73', alpha=0.4, hatch='//', edgecolor='#009E73', label='EC2 (pred.)')
ax.axhline(100, color='grey', linestyle=':', lw=0.6)
ax.set_xticks(x_cls); ax.set_xticklabels([f'M{int(c)}' for c in code_df['Class']], fontsize=7)
ax.set_ylabel('Pass Rate (%)'); ax.set_xlabel('Concrete Class'); ax.set_ylim(0, 115)
ax.legend(fontsize=5.5, loc='upper left', ncol=2); ax.grid(axis='y', alpha=0.3); clean_spines(ax)
fig7.tight_layout(pad=0.5); save_fig(fig7, 'fig7_code_compliance')

# --- Figure 8: Pareto-optimal mix designs ------------------------------------
fig8 = plt.figure(figsize=(7.2, 5.5))
gs8 = gridspec.GridSpec(2, 3, hspace=0.50, wspace=0.42)

ax = fig8.add_subplot(gs8[0, 0:2]); panel_label(ax, 'A', x=-0.08)
ax.scatter(cement_f[~pareto_mask], strength_f[~pareto_mask],
           s=1, alpha=0.03, c='#CCC', rasterized=True)
sc8 = ax.scatter(cement_f[pareto_mask], strength_f[pareto_mask],
                 c=wc_f[pareto_mask], cmap='RdYlBu_r', s=30, alpha=0.85,
                 edgecolors='#333', linewidths=0.3, zorder=5)
pidx = np.argsort(cement_f[pareto_mask])
ax.plot(cement_f[pareto_mask][pidx], strength_f[pareto_mask][pidx],
        '-', color='#D55E00', lw=1.5, alpha=0.7)
fig8.colorbar(sc8, ax=ax, shrink=0.8).set_label('W/C Ratio', fontsize=8)
ax.set_xlabel('Cement Content (kg/m³)'); ax.set_ylabel("Predicted $f'_{c,28}$ (MPa)")
ax.grid(True, alpha=0.3); clean_spines(ax)

ax = fig8.add_subplot(gs8[0, 2]); panel_label(ax, 'B')
eff8 = strength_f[pareto_mask] / cement_f[pareto_mask]
sc8b = ax.scatter(cement_f[pareto_mask], eff8, c=strength_f[pareto_mask],
                  cmap='viridis', s=22, alpha=0.8, edgecolors='#333', linewidths=0.3)
fig8.colorbar(sc8b, ax=ax, shrink=0.8).set_label("$f'_{c,28}$ (MPa)", fontsize=7)
ax.set_xlabel('Cement (kg/m³)'); ax.set_ylabel('Efficiency (MPa per kg/m³)')
ax.grid(True, alpha=0.3); clean_spines(ax)

ax = fig8.add_subplot(gs8[1, 0:2]); panel_label(ax, 'C', x=-0.04)
sc8c = ax.scatter(wc_f[pareto_mask], strength_f[pareto_mask],
                  c=cement_f[pareto_mask], cmap='plasma', s=35, alpha=0.85,
                  edgecolors='#333', linewidths=0.3, zorder=5)
fig8.colorbar(sc8c, ax=ax, shrink=0.8).set_label('Cement Content (kg/m³)', fontsize=8)
ax.set_xlabel('W/C Ratio')
ax.set_ylabel("Predicted $f'_{c,28}$ (MPa)")
ax.grid(True, alpha=0.3); clean_spines(ax)

ax = fig8.add_subplot(gs8[1, 2]); panel_label(ax, 'D'); ax.axis('off')
top10 = pareto_df.head(10)
cell_text = [[f'{r["Cement"]:.0f}', f'{r["WC"]:.3f}',
              f'{r["fc28"]:.1f}', f'{r["Eff"]:.4f}']
             for _, r in top10.iterrows()]
table = ax.table(cellText=cell_text,
                 colLabels=['Cem.\n(kg/m³)', 'W/C', "$f'_c$\n(MPa)", 'Eff.'],
                 loc='center', cellLoc='center')
table.auto_set_font_size(False); table.set_fontsize(6)
for j in range(4):
    table[0, j].set_height(0.15); table[0, j].set_facecolor('#E0E0E0')
    table[0, j].set_text_props(fontweight='bold', fontsize=6)
for i in range(1, len(cell_text) + 1):
    for j in range(4): table[i, j].set_height(0.08)
ax.set_title('Top 10 Pareto Designs', fontsize=8, fontweight='bold', pad=2)
fig8.tight_layout(pad=0.5); save_fig(fig8, 'fig8_pareto_optimal_mix_designs')

# --- Figure 9: Heatmap comparison --------------------------------------------
models_show = ['XGBoost', 'LightGBM', 'CatBoost', 'ExtraTrees', 'Ensemble']
tgt_row_labels = [
    '$f_{c,28}$ (Std)', '$f_{c,7}$ (Std)', 'Slump$_{30}$ (Std)', 'Slump$_{90}$ (Std)',
    '$f_{c,28}$ (GKF)', '$f_{c,7}$ (GKF)', 'Slump$_{30}$ (GKF)', 'Slump$_{90}$ (GKF)',
]
heat_data = []
for t in tgt_keys:
    heat_data.append([ALL_RESULTS[t]['standard'].get(m, {}).get('R2', np.nan) for m in models_show])
for t in tgt_keys:
    heat_data.append([ALL_RESULTS[t]['group'].get(m, {}).get('R2', np.nan) for m in models_show])
ha = np.array(heat_data)

fig9, ax9 = plt.subplots(figsize=(5.5, 4.0))
im = ax9.imshow(ha, cmap='RdYlGn', aspect='auto', vmin=-0.3, vmax=1.0)
ax9.set_xticks(range(len(models_show))); ax9.set_xticklabels(models_show, fontsize=8, fontweight='bold')
ax9.set_yticks(range(len(tgt_row_labels))); ax9.set_yticklabels(tgt_row_labels, fontsize=7.5)
for i in range(ha.shape[0]):
    for j in range(ha.shape[1]):
        v = ha[i, j]
        if not np.isnan(v):
            ax9.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=7.5,
                     fontweight='bold' if v > 0.85 else 'normal',
                     color='white' if v < 0.2 else 'black')
ax9.axhline(3.5, color='white', lw=2.5)
cb9 = fig9.colorbar(im, ax=ax9, shrink=0.85, pad=0.02)
cb9.set_label('R²', fontsize=9); cb9.ax.tick_params(labelsize=7)
fig9.tight_layout(pad=0.5); save_fig(fig9, 'fig9_heatmap_comparison')

# --- Figure S1: Diagnostic plots for secondary targets -----------------------
sec_tgts = ['Strength_7d', 'Slump_30', 'Slump_90']
sec_units = ['MPa', 'mm', 'mm']
sec_cols  = ['#2E7D32', '#E65100', '#6A1B9A']
fig_s1, axes_s1 = plt.subplots(3, 3, figsize=(7.2, 7.0))
for row, (tgt, unit, col) in enumerate(zip(sec_tgts, sec_units, sec_cols)):
    yt_s = ALL_OOF[tgt]['standard']['y']
    yp_s = ALL_OOF[tgt]['standard']['Ensemble']
    res_s = yt_s - yp_s
    r2_s = r2_score(yt_s, yp_s); rmse_s = np.sqrt(mean_squared_error(yt_s, yp_s))
    mu_s, sig_s = res_s.mean(), res_s.std()
    ax = axes_s1[row, 0]
    if row == 0: panel_label(ax, 'A')
    ax.scatter(yt_s, yp_s, alpha=0.35, s=8, color=col, edgecolors='none', rasterized=True)
    lims_s = [min(yt_s.min(), yp_s.min()) * 0.95, max(yt_s.max(), yp_s.max()) * 1.05]
    ax.plot(lims_s, lims_s, 'k--', lw=0.7)
    sl_s, ic_s, rv_s, _, _ = stats.linregress(yt_s, yp_s)
    xl_s = np.linspace(*lims_s, 100)
    ax.plot(xl_s, sl_s * xl_s + ic_s, color=col, lw=1.0, label=f'r={rv_s:.3f}')
    ax.set_xlim(lims_s); ax.set_ylim(lims_s)
    ax.set_xlabel(f'Actual ({unit})', fontsize=7); ax.set_ylabel(f'Predicted ({unit})', fontsize=7)
    ax.set_title(f'{tgt} — R²={r2_s:.4f}', fontsize=8, fontweight='bold')
    ax.legend(fontsize=5.5); ax.tick_params(labelsize=6); clean_spines(ax)
    ax = axes_s1[row, 1]
    if row == 0: panel_label(ax, 'B')
    ax.hist(res_s, bins=30, color=col, alpha=0.6, edgecolor='white', lw=0.3, density=True)
    xr_s = np.linspace(res_s.min(), res_s.max(), 200)
    ax.plot(xr_s, stats.norm.pdf(xr_s, mu_s, sig_s), 'k-', lw=0.8)
    ax.axvline(0, color='#D55E00', lw=0.8, ls='--')
    ax.set_xlabel(f'Residual ({unit})', fontsize=7)
    ax.set_title(f'μ={mu_s:.2f}, σ={sig_s:.2f}', fontsize=7)
    ax.tick_params(labelsize=6); clean_spines(ax)
    ax = axes_s1[row, 2]
    if row == 0: panel_label(ax, 'C')
    ax.scatter(yp_s, res_s, alpha=0.35, s=8, color=col, edgecolors='none', rasterized=True)
    ax.axhline(0, color='k', lw=0.7, ls='--')
    ax.axhline(2 * sig_s, color='#E69F00', lw=0.5, ls=':')
    ax.axhline(-2 * sig_s, color='#E69F00', lw=0.5, ls=':')
    ax.set_xlabel(f'Predicted ({unit})', fontsize=7); ax.set_ylabel(f'Residual ({unit})', fontsize=7)
    ax.set_title('Heteroscedasticity check', fontsize=7)
    ax.tick_params(labelsize=6); clean_spines(ax)
fig_s1.tight_layout(pad=0.5); save_fig(fig_s1, 'figS1_diagnostic_plots_secondary_targets')

print('[S12] All figures saved.')


# =============================================================================
# S13. FINAL RESULTS SUMMARY & CSV EXPORT
# =============================================================================

print('\n' + '=' * 78)
print('FINAL RESULTS SUMMARY')
print('=' * 78)

print('\nStandard 5-Fold CV (Ensemble):')
print(f'{"Target":<16s} {"R²":>8s} {"RMSE":>8s} {"MAE":>8s} {"MAPE%":>8s}')
print('-' * 50)
for tgt in tgt_keys:
    m = ALL_RESULTS[tgt]['standard']['Ensemble']
    print(f'{tgt:<16s} {m["R2"]:8.4f} {m["RMSE"]:8.3f} {m["MAE"]:8.3f} {m["MAPE"]:8.2f}')

print('\nGroupKFold CV (Ensemble):')
print(f'{"Target":<16s} {"R²":>8s} {"RMSE":>8s} {"MAE":>8s} {"MAPE%":>8s} {"ΔR²":>7s}')
print('-' * 57)
for tgt in tgt_keys:
    std = ALL_RESULTS[tgt]['standard']['Ensemble']
    grp = ALL_RESULTS[tgt]['group']['Ensemble']
    print(f'{tgt:<16s} {grp["R2"]:8.4f} {grp["RMSE"]:8.3f} {grp["MAE"]:8.3f} '
          f'{grp["MAPE"]:8.2f} {std["R2"]-grp["R2"]:7.4f}')

print(f'\nBootstrap 95% CI (CatBoost, f\'c,28):')
print(f'  R²:   median={r2_med:.4f}  [{r2_lo:.4f}, {r2_hi:.4f}]')
print(f'  RMSE: median={rmse_med:.3f}  [{rmse_lo:.3f}, {rmse_hi:.3f}]')
print(f'  MAE:  median={mae_med:.3f}  [{mae_lo:.3f}, {mae_hi:.3f}]')

print('\nMTL-PINN (5-Fold CV, multi-task joint training):')
print(f'{"Target":<16s} {"R²":>8s} {"RMSE":>8s} {"MAE":>8s} {"MAPE%":>8s}')
print('-' * 50)
for tgt in TARGETS_MTL:
    m = MTL_RESULTS[tgt]
    print(f'{tgt:<16s} {m["R2"]:8.4f} {m["RMSE"]:8.3f} {m["MAE"]:8.3f} {m["MAPE"]:8.2f}')

# Export master results table (includes tree ensemble + MTL-PINN)
rows = []
for tgt in tgt_keys:
    for scheme in ['standard', 'group']:
        m = ALL_RESULTS[tgt][scheme]['Ensemble']
        rows.append({'Target': tgt, 'Scheme': scheme, 'Model': 'TreeEnsemble',
                     'R2': m['R2'], 'RMSE': m['RMSE'],
                     'MAE': m['MAE'], 'MAPE': m['MAPE']})
for tgt in TARGETS_MTL:
    m = MTL_RESULTS[tgt]
    rows.append({'Target': tgt, 'Scheme': 'standard', 'Model': 'MTL_PINN',
                 'R2': m['R2'], 'RMSE': m['RMSE'],
                 'MAE': m['MAE'], 'MAPE': m['MAPE']})
pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, 'master_results_table.csv'), index=False)

print('\n[S13] master_results_table.csv saved.')
print('=' * 78)
print('PIPELINE COMPLETE — ALL OUTPUTS WRITTEN TO:', os.path.abspath(OUT_DIR))
print('=' * 78)
