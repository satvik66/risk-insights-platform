"""Capstone: 'Why did risk rise?' -> score portfolio, counterfactual driver attribution, retrieve policy, grounded answer."""
import joblib, numpy as np, pandas as pd
from common import *; from kb import DOCS; from retrieval import Index, docs_in_order
df = make_loans(); m = joblib.load(f"{OUT}/credit_model.joblib"); X = design(df); df["pd"] = m.predict_proba(X)[:, 1]
a, b = 10, 11; pm = df.groupby(["portfolio", "month"]).pd.mean().unstack()[[a, b]]; pm["chg"] = pm[b] - pm[a]
top = pm.chg.idxmax(); cur, prev = df[(df.portfolio == top) & (df.month == b)], df[(df.portfolio == top) & (df.month == a)]
rng = np.random.default_rng(0); contrib = {}
for f in FEATURES:                                       # swap one feature back to last month's distribution
    d = []
    for _ in range(5):
        Xc = design(cur); Xc[f] = rng.choice(prev[f].values, len(cur)); d.append(m.predict_proba(Xc)[:, 1].mean())
    contrib[f] = cur.pd.mean() - np.mean(d)
drivers = sorted(contrib.items(), key=lambda x: -x[1])[:3]
q = {"utilization": "revolving utilization increase portfolio escalated", "delinq_2y": "delinquency past due 30 days stage 2", "dti": "debt to income limit"}[drivers[0][0]]
hits = Index(DOCS).search(q, "hybrid", 3)
md = (f"**Question:** Why did risk exposure rise in the {top} portfolio in month {b}?\n\n"
      f"Mean predicted default probability in {top} moved from {pm.loc[top, a]:.4f} (month {a}) to {pm.loc[top, b]:.4f} (month {b}), a change of {pm.loc[top,'chg']:+.4f}. "
      f"Counterfactual attribution (reverting one input to last month's distribution) points to: " + "; ".join(f"{k} ({v:+.4f})" for k, v in drivers) + ".\n\n"
      f"Relevant policy [{hits[0]['doc']}]: {hits[0]['text']}\n\nSources: model scores, counterfactual drivers, policy docs {docs_in_order(hits)[:2]}.\n")
open(f"{OUT}/capstone_answer.md", "w").write(md); print(pm.round(4)); print(); print(md)
