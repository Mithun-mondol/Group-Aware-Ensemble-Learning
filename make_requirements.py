"""Write the pinned environment that the Data and Code Availability statement
(and disclosure A.2) promises.

Records the versions actually imported by the analysis rather than a blanket
`pip freeze`, so the file names only what the results depend on. Run it in the
same interpreter the analysis runs in.

    python make_requirements.py
"""
import importlib
import platform
import sys

# (import name, distribution name on PyPI)
PACKAGES = [
    ('numpy', 'numpy'),
    ('pandas', 'pandas'),
    ('scipy', 'scipy'),
    ('sklearn', 'scikit-learn'),
    ('xgboost', 'xgboost'),
    ('lightgbm', 'lightgbm'),
    ('catboost', 'catboost'),
    ('optuna', 'optuna'),
    ('shap', 'shap'),
    ('tensorflow', 'tensorflow'),
    ('keras', 'keras'),
    ('matplotlib', 'matplotlib'),
    ('statsmodels', 'statsmodels'),
    ('joblib', 'joblib'),
    ('openpyxl', 'openpyxl'),
]

lines = [
    '# Pinned environment for the analysis reported in',
    '#   "Group-Aware Ensemble Learning for Concrete Strength Prediction:',
    '#    A Dual-Validation Framework that Quantifies the Generalisation Gap"',
    '#',
    f'# Generated on {platform.platform()}',
    f'# Python {sys.version.split()[0]}',
    '#',
    '# NOTE ON DETERMINISM. The tree-ensemble results are reproducible under',
    '# this environment with random seed 42. The MTL-PINN baseline additionally',
    '# requires keras.utils.set_random_seed() rather than tf.random.set_seed():',
    '# under Keras 3 the latter does not reset the generator Keras draws its',
    '# initializer and dropout randomness from, and the baseline silently',
    '# stopped being reproducible when the environment moved to Keras 3.',
    '# The released pipeline calls the correct function; anyone re-running it',
    '# under Keras 2 is unaffected either way.',
    '',
]

missing = []
for mod, dist in PACKAGES:
    try:
        m = importlib.import_module(mod)
        v = getattr(m, '__version__', None)
        if v is None:
            from importlib.metadata import version as _v
            v = _v(dist)
        lines.append(f'{dist}=={v}')
    except Exception as e:
        missing.append(f'{dist} ({type(e).__name__})')

text = '\n'.join(lines) + '\n'
with open('requirements.txt', 'w', encoding='utf-8') as f:
    f.write(text)

print(text)
if missing:
    print('not importable in this interpreter, so not pinned: ' + ', '.join(missing))
print('written to requirements.txt')
