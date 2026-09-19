"""Remove the early-stopping leak from concrete_strength_pipeline.py.

Three defects, all disclosed in Part A of the response letter:

  D1  fit_model() passes the fold it is about to be scored on as `eval_set`
      with early_stopping_rounds=50. The number of boosting rounds for
      LightGBM and CatBoost was therefore chosen on the evaluation data. The
      same call is reused inside inner_hpo(), so the hyperparameter search was
      scored optimistically too.

  D2  inner_hpo() and the ensemble-weight inner OOF loop both use
      KFold(shuffle=True) regardless of the outer scheme, so under GroupKFold
      the tuning still saw every design class on both sides of the split.

  D3  fit_mtl_fold() passes the outer validation fold as `validation_data`
      with restore_best_weights=True -- selecting the network's weights on the
      data it is scored on -- and relies on seeding alone for reproducibility,
      which TensorFlow does not provide.

Run against a copy:  python patch_pipeline.py in.py out.py
"""
import sys

# (label, old, new, expected occurrences)
PATCHES = [

# ------------------------------------------------------------------ D3 (env)
('determinism environment (must precede the TensorFlow import)',
 '''import os
import warnings
''',
 '''import os
import warnings

# --- Determinism -------------------------------------------------------------
# Seeding alone does NOT make the TensorFlow baseline reproducible: the
# reduction order inside the kernels varies between runs, which moved this
# model's GroupKFold R2 across a range of 18 percentage points at a fixed seed.
# These variables must be set before TensorFlow is imported.
os.environ.setdefault('PYTHONHASHSEED', '42')
os.environ.setdefault('TF_DETERMINISTIC_OPS', '1')
os.environ.setdefault('TF_CUDNN_DETERMINISTIC', '1')
''', 1),

('group-aware split helpers',
 'from sklearn.model_selection import KFold, GroupKFold\n',
 'from sklearn.model_selection import (KFold, GroupKFold, GroupShuffleSplit,\n'
 '                                     train_test_split)\n', 1),

('enable op determinism once TensorFlow is imported',
 '''from tensorflow.keras import backend as K
''',
 '''from tensorflow.keras import backend as K

try:
    tf.config.experimental.enable_op_determinism()
    _TF_DETERMINISM = 'enabled'
except Exception as _e:                      # TensorFlow < 2.9
    _TF_DETERMINISM = f'UNAVAILABLE ({type(_e).__name__}); results will vary'
''', 1),

# ----------------------------------------------------------------------- D1
('fit_model: never show a model the rows it will be scored on',
 '''def fit_model(model, name, Xtr, ytr, Xva, yva):
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
    return model''',
 '''def fit_model(model, name, Xtr, ytr, Xva=None, yva=None):
    """Fit a single model on the training rows only.

    The released version passed (Xva, yva) as `eval_set` with
    early_stopping_rounds=50 for LightGBM and CatBoost. Because that pair is
    the fold the model is then scored on, the number of boosting rounds was
    selected on the evaluation data. No evaluation set is used here at all;
    n_estimators is a tuned quantity from the nested search (range 300-1500),
    which is what early stopping was standing in for.

    Xva/yva are retained in the signature so existing call sites keep working;
    they are deliberately ignored.
    """
    model.fit(Xtr, ytr)
    return model''', 1),

# ----------------------------------------------------------------------- D2
('inner_hpo: group-aware inner splits',
 '''def inner_hpo(X_inner, y_inner, n_trials=N_HPO_TRIALS):
    """Run Optuna HPO for XGBoost, LightGBM, CatBoost on inner CV.
    Returns best_params dict. Called only on outer-training data."""
    inner_cv = KFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=SEED)

    def _cv_score(model_fn, model_name):
        scores = []
        for tr, va in inner_cv.split(X_inner):''',
 '''def make_inner_splits(X, groups=None, inner_mode='kfold'):
    """Inner-fold indices that honour the outer scheme.

    The released version always used KFold(shuffle=True), so under GroupKFold
    the hyperparameter search and the ensemble-weight OOF were still scoring
    candidates on inner folds in which every design class appeared on both
    sides. Group separation is now preserved through tuning as well as
    through evaluation.
    """
    if inner_mode != 'group' or groups is None:
        return list(KFold(n_splits=N_INNER_FOLDS, shuffle=True,
                          random_state=SEED).split(X))
    k = min(N_INNER_FOLDS, len(np.unique(groups)))
    return list(GroupKFold(n_splits=k).split(X, groups=groups))


def inner_hpo(X_inner, y_inner, groups_inner=None, inner_mode='kfold',
              n_trials=N_HPO_TRIALS):
    """Run Optuna HPO for XGBoost, LightGBM, CatBoost on inner CV.
    Returns best_params dict. Called only on outer-training data."""
    inner_splits = make_inner_splits(X_inner, groups_inner, inner_mode)

    def _cv_score(model_fn, model_name):
        scores = []
        for tr, va in inner_splits:''', 1),

('run_outer_fold: thread the outer scheme through to the inner loops',
 '''def run_outer_fold(df_fold, target, feat_list, outer_tr_idx, outer_va_idx):''',
 '''def run_outer_fold(df_fold, target, feat_list, outer_tr_idx, outer_va_idx,
                   inner_mode='kfold', group_col='Concrete_Class'):''', 1),

('run_outer_fold: group-aware HPO',
 '''    best_p = inner_hpo(Xtr, ytr)

    # Inner OOF for ensemble weight optimisation
    inner_cv = KFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=SEED)
    model_names = ['XGBoost', 'LightGBM', 'CatBoost', 'ExtraTrees']
    inner_oof = np.zeros((len(ytr), len(model_names)))

    for itr, iva in inner_cv.split(Xtr):''',
 '''    groups_tr = tr_raw[group_col].values if inner_mode == 'group' else None
    best_p = inner_hpo(Xtr, ytr, groups_tr, inner_mode)

    # Inner OOF for ensemble weight optimisation, under the same grouping
    inner_splits = make_inner_splits(Xtr, groups_tr, inner_mode)
    model_names = ['XGBoost', 'LightGBM', 'CatBoost', 'ExtraTrees']
    inner_oof = np.zeros((len(ytr), len(model_names)))

    for itr, iva in inner_splits:''', 1),

('run_outer_fold: final fit sees no validation rows',
 '''    models_final = make_base_models(best_p['xgb'], best_p['lgb'], best_p['cb'])
    va_preds = {}
    for mi, (mname, m) in enumerate(models_final.items()):
        m = fit_model(m, mname, Xtr, ytr, Xva, yva)
        va_preds[mname] = m.predict(Xva)''',
 '''    models_final = make_base_models(best_p['xgb'], best_p['lgb'], best_p['cb'])
    va_preds = {}
    for mi, (mname, m) in enumerate(models_final.items()):
        m = fit_model(m, mname, Xtr, ytr)      # no eval_set: see fit_model
        va_preds[mname] = m.predict(Xva)''', 1),

('run_dual_validation: tell each fold which scheme it is under',
 '''        split_kw = {'groups': groups} if scheme_name == 'group' else {}
        for tr_idx, va_idx in splitter.split(df, **split_kw):
            _, ens_pred, va_preds, w, _ = run_outer_fold(
                df, target, feat_list, tr_idx, va_idx)''',
 '''        split_kw = {'groups': groups} if scheme_name == 'group' else {}
        inner_mode = 'group' if scheme_name == 'group' else 'kfold'
        for tr_idx, va_idx in splitter.split(df, **split_kw):
            _, ens_pred, va_preds, w, _ = run_outer_fold(
                df, target, feat_list, tr_idx, va_idx, inner_mode=inner_mode)''', 1),

# ----------------------------------------------------------------------- D3
('fit_mtl_fold: carve the early-stopping split out of the training rows',
 '''def fit_mtl_fold(tr_proc, va_proc, feats, targets,
                 epochs=300, batch_size=32, verbose=0):''',
 '''def fit_mtl_fold(tr_proc, va_proc, feats, targets,
                 epochs=300, batch_size=32, verbose=0,
                 groups_tr=None):''', 1),

('fit_mtl_fold: stop on held-out training rows, not on the scored fold',
 '''    cb_es = callbacks.EarlyStopping(monitor='val_loss', patience=40,
                                     restore_best_weights=True, verbose=0)
    cb_rl = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                         patience=20, min_lr=1e-5, verbose=0)
    model.fit(Xtr, ytr, validation_data=(Xva, yva),
              epochs=epochs, batch_size=batch_size,
              callbacks=[cb_es, cb_rl], verbose=verbose)''',
 '''    # The released version passed (Xva, yva) -- the fold this model is scored
    # on -- as validation_data, with restore_best_weights=True. That selects
    # both the stopping epoch and the final weights on the evaluation data.
    # The early-stopping split is carved out of the training rows instead, and
    # respects the grouping when the outer split is group-aware.
    _idx = np.arange(len(Xtr))
    if groups_tr is not None and len(np.unique(groups_tr)) > 1:
        _gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
        _fit_i, _es_i = next(_gss.split(_idx, groups=groups_tr))
    else:
        _fit_i, _es_i = train_test_split(_idx, test_size=0.15,
                                         random_state=SEED)

    cb_es = callbacks.EarlyStopping(monitor='val_loss', patience=40,
                                     restore_best_weights=True, verbose=0)
    cb_rl = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                         patience=20, min_lr=1e-5, verbose=0)
    model.fit(Xtr[_fit_i], {k: v[_fit_i] for k, v in ytr.items()},
              validation_data=(Xtr[_es_i],
                               {k: v[_es_i] for k, v in ytr.items()}),
              epochs=epochs, batch_size=batch_size,
              callbacks=[cb_es, cb_rl], verbose=verbose)''', 1),
]

# --------------------------------------------- D1 continued: diagnostic refits
# Table 5 (jackknife), Table 6 (ablation), Table S1 and the Figure 5 bootstrap
# each refit a fixed-parameter CatBoost with eval_set set to the very rows they
# score. Same defect as D1, at patience 50 and 30.
PATCHES += [

('inner_hpo XGBoost: drop early stopping (no eval_set is supplied any more)',
 '''        return _cv_score(lambda: xgb.XGBRegressor(**p, random_state=SEED,
                                                    n_jobs=-1, verbosity=0,
                                                    early_stopping_rounds=30),
                         'XGBoost')''',
 '''        # early_stopping_rounds is gone: fit_model no longer supplies an
        # eval_set, and XGBoost raises if early stopping is asked for without
        # one. n_estimators is searched over 300-1500 and carries the role.
        return _cv_score(lambda: xgb.XGBRegressor(**p, random_state=SEED,
                                                    n_jobs=-1, verbosity=0),
                         'XGBoost')''', 1),

('Table 6 anchor ablation: no eval_set on the scored fold',
 '''        m.fit(Xtr, ytr, eval_set=(Xva, va_proc['Strength_28d'].values.astype(np.float32)),
              early_stopping_rounds=50, verbose=0)
        y_oof[va_idx] = m.predict(Xva)''',
 '''        m.fit(Xtr, ytr, verbose=0)          # no eval_set: see fit_model
        y_oof[va_idx] = m.predict(Xva)''', 1),

('Table S2 leave-one-class-out: no eval_set on the scored fold',
 '''        m.fit(Xtr, ytr, eval_set=(Xva, va_proc['Strength_28d'].values.astype(np.float32)),
              early_stopping_rounds=50, verbose=0)
        y_oof_g[va_idx] = m.predict(Xva)''',
 '''        m.fit(Xtr, ytr, verbose=0)          # no eval_set: see fit_model
        y_oof_g[va_idx] = m.predict(Xva)''', 1),

('Figure 5 bootstrap: no eval_set on the out-of-bag rows',
 '''        cb.fit(Xtr, ytr, eval_set=(Xva, yva),
               early_stopping_rounds=30, verbose=0)''',
 '''        cb.fit(Xtr, ytr, verbose=0)         # no eval_set: see fit_model''', 1),

('quick_cv_r2 (Table S1 and the LOCO standard arm): no eval_set',
 '''        m.fit(Xtr, ytr, eval_set=(Xva, va_proc[target].values.astype(np.float32)),
              early_stopping_rounds=50, verbose=0)''',
 '''        m.fit(Xtr, ytr, verbose=0)          # no eval_set: see fit_model''', 1),
]


def main():
    src_path, out_path = sys.argv[1], sys.argv[2]
    src = open(src_path, encoding='utf-8').read()
    fails = []

    for label, old, new, count in PATCHES:
        n = src.count(old)
        if n == count:
            src = src.replace(old, new)
            print(f'  ok   {label}')
        elif src.count(new) >= 1:
            print(f'  --   {label} (already applied)')
        else:
            fails.append(f'{label}: found {n}, expected {count}')

    if fails:
        for f in fails:
            print(f'  !! {f}')
        sys.exit(1)

    open(out_path, 'w', encoding='utf-8').write(src)
    print(f'  wrote {out_path} ({len(src)} chars)')


if __name__ == '__main__':
    main()
