"""FastAPI model serving — synthetic demonstration, not a lending decision engine."""
from pathlib import Path
import pickle
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field
import pandas as pd
app=FastAPI(title='Risk Insights Model API',version='0.1.0')
class CreditInput(BaseModel):
    income:float=Field(gt=0)
    utilization:float=Field(ge=0,le=1)
    delinquency_count:int=Field(ge=0)
    debt_to_income:float=Field(ge=0,le=1)
    loan_amount:float=Field(gt=0)
@app.get('/healthz')
def health():return {'status':'ok'}
@app.get('/readyz')
def ready():
    if not Path('artifacts/credit_model.pkl').exists():raise HTTPException(503,'Train the credit model first')
    return {'status':'ready'}
@app.post('/predict')
def predict(item:CreditInput):
    p=Path('artifacts/credit_model.pkl')
    if not p.exists():raise HTTPException(503,'Model not initialized')
    with p.open('rb') as f:bundle=pickle.load(f)
    row=item.model_dump();row['estimated_monthly_payment_ratio']=row['loan_amount']/(row['income']+1)
    probability=float(bundle['model'].predict_proba(pd.DataFrame([row])[bundle['features']])[0,1])
    return {'synthetic_default_risk_score':round(probability,6),'disclaimer':'Research demo only. Not suitable for real credit decisions.'}
