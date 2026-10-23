import numpy as np, pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
OUT = "outputs"

def make_loans(n=30000, seed=0):
    rng = np.random.default_rng(seed)
    month = rng.integers(0, 12, n)
    income = rng.lognormal(10.8, 0.5, n)
    loan_amt = rng.lognormal(9.5, 0.6, n)
    util = np.clip(rng.beta(2, 3, n) + 0.015 * month, 0, 1)      # utilization drifts up over time
    delinq = rng.poisson(0.3 + 0.03 * month, n)
    dti = np.clip(loan_amt / income * rng.uniform(2, 6, n), 0, 1.5)
    credit_age = rng.gamma(4, 2, n)
    inq = rng.poisson(1.5, n)
    portfolio = rng.choice(["auto", "home", "card"], n, p=[.4, .3, .3])
    z = -4.2 + 2.0*util + 0.5*delinq + 1.5*dti - 0.05*credit_age + 0.12*inq + (portfolio == "card")*0.4
    default = (rng.random(n) < 1/(1+np.exp(-z))).astype(int)
    return pd.DataFrame(dict(loan_id=np.arange(n), month=month, income=income, loan_amt=loan_amt,
        utilization=util, delinq_2y=delinq, dti=dti, credit_age=credit_age, inquiries=inq,
        portfolio=portfolio, default=default))

FEATURES = ["income", "loan_amt", "utilization", "delinq_2y", "dti", "credit_age", "inquiries"]
def design(df):
    X = df[FEATURES].copy()
    for p in ["auto", "home", "card"]: X["pf_" + p] = (df.portfolio == p).astype(int)
    return X

def ks(y, s):
    f, t, _ = roc_curve(y, s); return float(np.max(t - f))

def psi(expected, actual, bins=10):
    cuts = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1))); cuts[0], cuts[-1] = -np.inf, np.inf
    e = np.clip(np.histogram(expected, cuts)[0] / len(expected), 1e-4, None)
    a = np.clip(np.histogram(actual, cuts)[0] / len(actual), 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))
