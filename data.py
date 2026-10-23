"""Deterministic synthetic financial data. Never real clients."""
from pathlib import Path
import numpy as np
import pandas as pd

def generate(out='data', n=3500, seed=42):
    rng=np.random.default_rng(seed)
    p=Path(out); p.mkdir(parents=True,exist_ok=True)
    customer_id=rng.integers(1,max(100,n//5),n)
    income=rng.lognormal(10.7,.48,n).clip(18000,260000)
    utilization=rng.beta(2.2,3.0,n)
    delinquency=rng.poisson(.35,n)
    debt_to_income=rng.beta(2.4,5,n)
    loan_amount=rng.uniform(2500,90000,n)
    segment=rng.choice(['auto','personal','mortgage'],n,p=[.45,.35,.2])
    month=rng.choice(pd.date_range('2025-01-01','2026-06-01',freq='MS'),n)
    z=-3.2+2.8*utilization+.73*delinquency+2.4*debt_to_income-.000005*income
    prob=1/(1+np.exp(-z))
    default=rng.binomial(1,prob)
    loans=pd.DataFrame(dict(loan_id=np.arange(1,n+1),customer_id=customer_id,month=month,portfolio=segment,income=income,utilization=utilization,delinquency_count=delinquency,debt_to_income=debt_to_income,loan_amount=loan_amount,default=default))
    loans.to_csv(p/'loans.csv',index=False)
    m=n*2
    trans=pd.DataFrame(dict(event_id=np.arange(1,m+1),customer_id=rng.integers(1,max(100,n//5),m),amount=rng.lognormal(4,1.35,m).round(2),country=rng.choice(['US','CA','MX'],m),hour=rng.integers(0,24,m),device_change=rng.binomial(1,.09,m),txn_count_24h=rng.poisson(3,m)))
    fraudscore=-5+.0011*trans.amount+.9*trans.device_change+.16*trans.txn_count_24h+.6*((trans.hour<5).astype(int))
    trans['fraud']=rng.binomial(1,1/(1+np.exp(-fraudscore)))
    trans.to_csv(p/'transactions.csv',index=False)
    docs=p/'policies';docs.mkdir(exist_ok=True)
    (docs/'loan_risk.txt').write_text('Loan Risk Policy: Review auto-loan portfolios monthly. Escalate delinquency concentration above approved thresholds. Credit risk decisions must have reason codes, an audit trail and human oversight.\n')
    (docs/'fraud.txt').write_text('Fraud Operations Procedure: Prioritize unusual transaction amounts, changes in devices and high transaction velocity. A model score is an alert, not proof of wrongdoing. Analysts must review alerts before action.\n')
    (docs/'governance.txt').write_text('Model Governance: Monitor drift, calibration and false positive rates. Review demographic fairness with properly authorized data. Do not use the LLM to authorize a credit or fraud decision.\n')
    return loans,trans
