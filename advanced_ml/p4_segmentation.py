"""P4: customer risk segmentation: KMeans (k by silhouette) vs GMM, profiles mapped to risk tiers."""
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from common import *
df = make_loans(); cols = ["utilization", "delinq_2y", "dti", "credit_age", "inquiries"]
Z = StandardScaler().fit_transform(df[cols]); sub = np.random.default_rng(0).choice(len(df), 5000, replace=False)
sil = {k: silhouette_score(Z[sub], KMeans(k, n_init=5, random_state=0).fit_predict(Z[sub])) for k in range(2, 8)}
k = max(sil, key=sil.get); df["seg"] = KMeans(k, n_init=10, random_state=0).fit_predict(Z)
df["seg_gmm"] = GaussianMixture(k, random_state=0).fit_predict(Z)
prof = df.groupby("seg").agg(n=("loan_id", "size"), default_rate=("default", "mean"), **{c: (c, "mean") for c in cols}).round(3)
prof["tier"] = pd.qcut(prof.default_rate.rank(method="first"), [0, .34, .67, 1], labels=["Low", "Medium", "High"])
prof.to_csv(f"{OUT}/p4_segments.csv"); print("silhouette by k:", {a: round(b, 3) for a, b in sil.items()}, "-> k =", k); print(prof.sort_values("default_rate"))
print("KMeans vs GMM default-rate spread:", round(df.groupby("seg").default.mean().agg(np.ptp), 3), round(df.groupby("seg_gmm").default.mean().agg(np.ptp), 3))
