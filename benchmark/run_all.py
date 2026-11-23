"""Run all 12 feature-selection scripts on data/breast_cancer.csv and record what they pick.

Writes benchmark/results.json. For each script: runtime, the features it kept (read from the CSV
it wrote to outputs/), and the 5-fold cross-validated accuracy of a standardised logistic regression
on those features (StratifiedKFold, shuffle, random_state=42; solver tolerance 1e-10 so the result is the
exact optimum and the browser demo, which solves the same problem with Newton's method, agrees to the row).

Usage:  python benchmark/run_all.py            (from anywhere; needs scikit-learn, pandas, deap)
        python benchmark/run_all.py --reuse    (skip the scripts, re-score the CSVs already in outputs/)
Slow scripts (Forward selection, RFE, Ant colony) take a few minutes in total.
"""
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data', 'breast_cancer.csv')
OUT = os.path.join(ROOT, 'outputs')

SCRIPTS = [  # (family, name, path, output csv)
    ('Filter', 'ANOVA F-test (k=10)', 'Filter/ANOVA.py', 'ANOVA_FS.csv'),
    ('Filter', 'Mutual information', 'Filter/InformationGain.py', 'InfoGain_breast_cancer.csv'),
    ('Filter', 'Chi-squared', 'Filter/CST3.py', 'CST_breast_cancer.csv'),
    ('Wrapper', 'Forward selection', 'Wrapper/ForwardSelection.py', 'ForSel_breast_cancer.csv'),
    ('Wrapper', 'Backward elimination (5)', 'Wrapper/BackwardsElimination.py', 'BE_breast_cancer.csv'),
    ('Wrapper', 'RFE', 'Wrapper/RFE.py', 'RFE_FS.csv'),
    ('Embedded', 'Lasso', 'Embedded/LassoRegression.py', 'LassoReg_breast_cancer.csv'),
    ('Embedded', 'Ridge (above-mean |coef|)', 'Embedded/RR.py', 'RR_breast_cancer.csv'),
    ('Embedded', 'Extra-trees importance', 'Embedded/Tree.py', 'Tree_FS.csv'),
    ('Hybrid', 'Chi-squared + simulated annealing', 'Hybrid/ChiSimulatedAnnealing.py', 'ChiSimAnneal_breast_cancer.csv'),
    ('Hybrid', 'Genetic algorithm + random forest', 'Hybrid/GeneticAlgorithmClassifier_FS.py', 'GA_breast_cancer.csv'),
    ('Hybrid', 'Ant colony + logistic regression', 'Hybrid/Ant.py', 'Ant_FS.csv'),
]

df = pd.read_csv(DATA)
X, y = df.iloc[:, :-1], df.iloc[:, -1]
feats = list(X.columns)
cv = StratifiedKFold(5, shuffle=True, random_state=42)
folds = np.zeros(len(df), dtype=int)
for k, (_, te) in enumerate(cv.split(X, y)):
    folds[te] = k


def cv_acc(cols):
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=100000, tol=1e-10))
    return float(cross_val_score(pipe, X[cols], y, cv=cv, scoring='accuracy').mean())


REUSE = '--reuse' in sys.argv
old = json.load(open(os.path.join(ROOT, 'benchmark', 'results.json'))) if REUSE else None
env = dict(os.environ, MPLBACKEND='Agg', PYTHONDONTWRITEBYTECODE='1')
rows = []
for fam, name, path, outcsv in SCRIPTS:
    t0 = time.time()
    if REUSE:
        prev = next(m for m in old['methods'] if m['script'] == path)
        sel = list(pd.read_csv(os.path.join(OUT, outcsv)).columns[:-1])
        rows.append(dict(prev, selected=sel, n_selected=len(sel), cv_acc=cv_acc(sel)))
        print(f"{path:45s} reused n={len(sel):2d} acc={rows[-1]['cv_acc']}", flush=True)
        continue
    arg = DATA if path.startswith(('Filter/ANOVA', 'Embedded/Tree', 'Wrapper/RFE', 'Hybrid/Ant')) else 'breast_cancer.csv'
    p = subprocess.run([sys.executable, '-I', os.path.join(ROOT, path), arg], capture_output=True, text=True, env=env, cwd=ROOT)
    dt = time.time() - t0
    ok = p.returncode == 0
    sel = list(pd.read_csv(os.path.join(OUT, outcsv)).columns[:-1]) if ok else []
    row = dict(family=fam, name=name, script=path, ok=ok, seconds=round(dt, 1), selected=sel, n_selected=len(sel),
               cv_acc=cv_acc(sel) if sel else None, stderr=p.stderr.strip()[-400:])
    rows.append(row)
    print(f"{path:45s} ok={ok} {dt:6.1f}s n={len(sel):2d} acc={row['cv_acc']}", flush=True)

out = dict(dataset='breast_cancer', n_rows=len(df), n_features=len(feats), features=feats,
           target=list(map(int, y)), folds=list(map(int, folds)),
           X=[[float(v) for v in r] for r in X.values], baseline_cv_acc=cv_acc(feats), methods=rows)
json.dump(out, open(os.path.join(ROOT, 'benchmark', 'results.json'), 'w'), indent=1)
print('baseline all features', out['baseline_cv_acc'])
