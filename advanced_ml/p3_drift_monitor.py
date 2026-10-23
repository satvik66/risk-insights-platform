"""P3: drift & performance monitor: monthly PSI per feature + AUC decay, retrain alerts, chart."""
import joblib, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from common import *
df = make_loans(); m = joblib.load(f"{OUT}/credit_model.joblib"); base = df[df.month <= 3]
rows = []
for mo in range(4, 12):
    cur = df[df.month == mo]; r = {"month": mo, "auc": roc_auc_score(cur.default, m.predict_proba(design(cur))[:, 1])}
    for f in FEATURES: r["psi_" + f] = psi(base[f], cur[f])
    rows.append(r)
mon = pd.DataFrame(rows); base_auc = mon.auc.iloc[0]
mon["alert"] = [("RETRAIN: " + ",".join(c[4:] for c in mon.columns if c.startswith("psi_") and r[c] > 0.2)) if any(r[c] > 0.2 for c in mon.columns if c.startswith("psi_")) or (base_auc - r.auc > 0.03) else "ok" for _, r in mon.iterrows()]
mon.round(4).to_csv(f"{OUT}/p3_drift_report.csv", index=False)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for f in ["utilization", "delinq_2y", "dti"]: ax[0].plot(mon.month, mon["psi_" + f], marker="o", label=f)
ax[0].axhline(0.2, ls="--", c="r"); ax[0].set_title("Feature PSI vs baseline (alert > 0.2)"); ax[0].legend()
ax[1].plot(mon.month, mon.auc, marker="o"); ax[1].set_title("Monthly AUC"); fig.tight_layout(); fig.savefig(f"{OUT}/p3_drift.png", dpi=110)
print(mon[["month", "auc", "psi_utilization", "psi_delinq_2y", "alert"]].round(3).to_string(index=False))
