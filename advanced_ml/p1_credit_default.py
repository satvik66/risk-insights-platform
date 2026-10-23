"""P1: credit default pipeline: out-of-time split, GBM, AUC/KS/PSI scorecard, permutation importance."""
import json, joblib, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from common import *
df = make_loans()
tr, va, te = df[df.month <= 7], df[df.month.between(8, 9)], df[df.month >= 10]
m = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.07, max_iter=200, random_state=0)
m.fit(design(tr), tr.default)
s = {k: m.predict_proba(design(d))[:, 1] for k, d in dict(train=tr, valid=va, test=te).items()}
card = {"auc_train": roc_auc_score(tr.default, s["train"]), "auc_valid": roc_auc_score(va.default, s["valid"]),
        "auc_test_oot": roc_auc_score(te.default, s["test"]), "ks_test": ks(te.default, s["test"]),
        "score_psi_train_vs_test": psi(s["train"], s["test"]),
        "default_rate_train": tr.default.mean(), "default_rate_test": te.default.mean()}
imp = permutation_importance(m, design(te), te.default, scoring="roc_auc", n_repeats=5, random_state=0)
card["importance"] = dict(sorted(zip(design(te).columns, map(float, imp.importances_mean)), key=lambda x: -x[1]))
json.dump({k: (round(v, 4) if isinstance(v, float) else v) for k, v in card.items()}, open(f"{OUT}/p1_scorecard.json", "w"), indent=1, default=float)
joblib.dump(m, f"{OUT}/credit_model.joblib")
te.assign(pd_score=s["test"]).to_csv(f"{OUT}/p1_scored_test.csv", index=False)
print(json.dumps(card, indent=1, default=lambda x: round(float(x), 4)))
