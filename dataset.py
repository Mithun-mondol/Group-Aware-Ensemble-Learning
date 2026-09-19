"""The 683 modelled records, loaded exactly as the released pipeline does.

Shared by figures_R2.py and supp_tables.py so the figures and the supplementary
tables cannot disagree about which rows are in the dataset.
"""
import os

import pandas as pd

# The raw ledger, for the figures that plot mixture variables rather than
# stored predictions. The four preprocessing steps are the pipeline's own
# (S2): drop rows missing any critical column, remove physically impossible
# strength regression, de-duplicate identical mixes, and winsorise at 1/99 %
# for the columns the pipeline winsorises. Reproduced here so Figure 7 can be
# rebuilt without refitting a model.
XLSX = os.environ.get('R2_XLSX', 'Dhaka-Sylhet_mix_design.xlsx')
WINSOR_COLS = ['Cement_Content', 'WC_Ratio', 'Admixture_Dose', 'CA', 'FA',
               'Strength_7d', 'Slump_30', 'Slump_60', 'Slump_90']
CRITICAL = ['Concrete_Class', 'Cement', 'Admixture', 'Cement_Content',
            'WC_Ratio', 'Admixture_Dose', 'Strength_7d', 'Strength_28d',
            'Slump_30', 'Slump_90']
DUP = ['Cement', 'Admixture', 'CA', 'FA', 'Cement_Content', 'WC_Ratio',
       'Admixture_Dose', 'Strength_7d', 'Strength_28d', 'Slump_30',
       'Slump_60', 'Slump_90']
RENAME = {'Concrete Class': 'Concrete_Class',
          'Cement Content (kg/m3)': 'Cement_Content', 'W/C': 'WC_Ratio',
          'Strength 7-Days': 'Strength_7d', 'Strength 28-Days': 'Strength_28d',
          'Slump (mm) 30 mins': 'Slump_30', 'Slump (mm) 60 mins': 'Slump_60',
          'Slump (mm) 90 mins': 'Slump_90'}


def load_raw():
    """(df_raw, df_proc) -- the 683 modelled rows, unwinsorised and winsorised."""
    d = pd.read_excel(XLSX, header=2).rename(columns=RENAME)
    d = d.rename(columns={c: 'Admixture_Dose' for c in d.columns
                          if str(c).startswith('Admixture Doses')})
    d = d[pd.to_numeric(d['SN'], errors='coerce').notna()].reset_index(drop=True)
    for c in ['CA', 'FA', 'Cement_Content', 'WC_Ratio', 'Admixture_Dose',
              'Strength_7d', 'Strength_28d', 'Slump_30', 'Slump_60',
              'Slump_90', 'Concrete_Class']:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    for c in ['Cement', 'Admixture']:
        d[c] = d[c].astype(str).str.strip().str.title()
    d = d.dropna(subset=CRITICAL).reset_index(drop=True)
    d = d[d.Strength_28d >= d.Strength_7d].reset_index(drop=True)
    d = d.drop_duplicates(subset=DUP, keep='first').reset_index(drop=True)

    p = d.copy()
    for c in WINSOR_COLS:
        p[c] = p[c].clip(d[c].quantile(0.01), d[c].quantile(0.99))
    p['GSR'] = (0.68 * 0.75) / (0.32 * 0.75 + p.WC_Ratio)
    p['Binder_Intensity'] = p.Cement_Content / p.Concrete_Class
    return d, p


def winsorised_sd(df, group='Concrete_Class', col='Strength_28d'):
    """Within-class population SD, which is what the ACI s term uses."""
    return df.groupby(group)[col].std(ddof=0)
