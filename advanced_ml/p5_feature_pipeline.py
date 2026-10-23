"""P5: point-in-time feature pipeline with leakage test and data-quality gates (pandas stand-in for PySpark; Spark version in deploy/)."""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from common import OUT
rng = np.random.default_rng(5); C = 600
ev = pd.DataFrame({"cust": rng.integers(0, C, C*45), "ts": rng.integers(0, 150, C*45)}); ev["amount"] = rng.lognormal(3, .8, len(ev))
snaps = pd.DataFrame([(c, s) for c in range(C) for s in (60, 90, 110)], columns=["cust", "snap"])
def features(ev, snaps, W=30, leaky=False):
    m = snaps.merge(ev, on="cust")
    hi = (m.snap + W) if leaky else m.snap                       # leaky=True peeks into the future
    m = m[(m.ts >= m.snap - W) & (m.ts < hi)]
    g = m.groupby(["cust", "snap"]).amount.agg(txn_cnt_30d="count", txn_sum_30d="sum").reset_index()
    return snaps.merge(g, how="left", on=["cust", "snap"]).fillna({"txn_cnt_30d": 0, "txn_sum_30d": 0})
pit, leak = features(ev, snaps), features(ev, snaps, leaky=True)
fut = snaps.merge(ev, on="cust"); fut = fut[(fut.ts >= fut.snap) & (fut.ts < fut.snap + 30)].groupby(["cust", "snap"]).amount.sum().rename("fut").reset_index()
lab = snaps.merge(fut, how="left", on=["cust", "snap"]).fillna({"fut": 0}); lab["y"] = (lab.fut > lab.fut.median()).astype(int)
# tests
for i in rng.choice(len(pit), 50, replace=False):                 # brute-force recompute
    c, s = snaps.iloc[i]; e = ev[(ev.cust == c) & (ev.ts >= s - 30) & (ev.ts < s)]
    assert pit.txn_cnt_30d[i] == len(e) and abs(pit.txn_sum_30d[i] - e.amount.sum()) < 1e-6
dq = {"no_nulls": bool(pit.notna().all().all()), "unique_key": not pit.duplicated(["cust", "snap"]).any(),
      "non_negative": bool((pit[["txn_cnt_30d", "txn_sum_30d"]] >= 0).all().all()), "row_count_matches": len(pit) == len(snaps)}
assert all(dq.values()), dq
pit.to_csv(f"{OUT}/p5_features_pit.csv", index=False)
print("data quality:", dq, "| brute-force PIT check passed on 50 rows")
print("AUC vs future label  PIT feature:", round(roc_auc_score(lab.y, pit.txn_sum_30d), 3), " leaky feature:", round(roc_auc_score(lab.y, leak.txn_sum_30d), 3))
