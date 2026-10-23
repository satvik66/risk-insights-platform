"""P2: fraud detection: supervised + Isolation Forest on imbalanced data, cost-based threshold, ranked alert queue."""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from common import OUT
rng = np.random.default_rng(1); n = 60000
amt = rng.lognormal(3.5, 1.1, n); hour = rng.integers(0, 24, n); mrisk = rng.beta(1.5, 8, n)
vel = rng.poisson(1.0, n); dist = rng.exponential(15, n); newdev = rng.random(n) < 0.08
z = -7.2 + 0.0012*amt + 0.9*(hour < 5) + 4*mrisk + 0.35*vel + 0.012*dist + 1.6*newdev
fraud = rng.random(n) < 1/(1+np.exp(-z))
df = pd.DataFrame(dict(txn_id=np.arange(n), amount=amt, hour=hour, merchant_risk=mrisk, velocity_1h=vel, dist_home=dist, new_device=newdev.astype(int), fraud=fraud.astype(int)))
X = df.drop(columns=["txn_id", "fraud"]); cut = int(n*.7)
Xtr, Xte, ytr, yte = X[:cut], X[cut:], df.fraud[:cut], df.fraud[cut:]
w = np.where(ytr == 1, (ytr == 0).sum()/ytr.sum(), 1.0)                       # cost-sensitive weighting
sup = HistGradientBoostingClassifier(max_depth=4, max_iter=150, random_state=0).fit(Xtr, ytr, sample_weight=w)
iso = IsolationForest(n_estimators=200, contamination=0.01, random_state=0).fit(Xtr)
p_sup = sup.predict_proba(Xte)[:, 1]; a_iso = -iso.score_samples(Xte)
rank = lambda v: pd.Series(v).rank(pct=True).values
blend = 0.8*rank(p_sup) + 0.2*rank(a_iso)
res = {"fraud_rate": float(df.fraud.mean()), "PR_AUC_supervised": average_precision_score(yte, p_sup),
       "PR_AUC_isoforest": average_precision_score(yte, a_iso), "PR_AUC_blend": average_precision_score(yte, blend)}
# business-cost threshold: missed fraud costs the txn amount, each review costs $5
amts = Xte.amount.values; yv = yte.values; best = None
for t in np.linspace(0.05, 0.99, 95):
    flag = p_sup >= t; cost = amts[(~flag) & (yv == 1)].sum() + 5*flag.sum()
    if best is None or cost < best[1]: best = (t, cost, int(flag.sum()))
res.update(best_threshold=best[0], min_total_cost=best[1], alerts_at_threshold=best[2],
           cost_no_model=float(amts[yv == 1].sum()), cost_at_naive_0_5=float(amts[(p_sup < .5) & (yv == 1)].sum() + 5*(p_sup >= .5).sum()))
q = Xte.assign(txn_id=df.txn_id[cut:].values, fraud_score=p_sup, actual=yv).sort_values("fraud_score", ascending=False).head(100)
q.to_csv(f"{OUT}/p2_alert_queue.csv", index=False)
res["precision_at_100"] = float(q.actual.mean())
print({k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()})
