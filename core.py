"""ML modules for local demonstrators using only synthetic data."""
from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier,IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score,roc_curve,average_precision_score
from sklearn.model_selection import train_test_split

LOAN_NUM=['income','utilization','delinquency_count','debt_to_income','loan_amount']
TXN_NUM=['amount','hour','device_change','txn_count_24h']

def _save(path, obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(path,'wb') as f:pickle.dump(obj,f)

def _load(path):
    with open(path,'rb') as f:return pickle.load(f)

def ks_stat(y,score):
    fpr,tpr,_=roc_curve(y,score)
    return float(np.max(tpr-fpr))

def psi(ref,cur,bins=10):
    ref=np.asarray(ref,dtype=float);cur=np.asarray(cur,dtype=float)
    edges=np.unique(np.quantile(ref,np.linspace(0,1,bins+1)))
    if len(edges)<2:return 0.
    edges[0]=-np.inf;edges[-1]=np.inf
    a=np.histogram(ref,bins=edges)[0]/max(len(ref),1)
    b=np.histogram(cur,bins=edges)[0]/max(len(cur),1)
    a=np.maximum(a,1e-5);b=np.maximum(b,1e-5)
    return float(np.sum((b-a)*np.log(b/a)))

def credit(data='data',out='artifacts'):
    df=pd.read_csv(Path(data)/'loans.csv')
    # Illustrative derived feature; no leakage from outcome label.
    df['estimated_monthly_payment_ratio']=df.loan_amount/(df.income+1)
    cols=LOAN_NUM+['estimated_monthly_payment_ratio']
    train,test=train_test_split(df,test_size=.25,random_state=42,stratify=df.default)
    model=HistGradientBoostingClassifier(max_iter=90,max_leaf_nodes=15,learning_rate=.07,random_state=42)
    model.fit(train[cols],train.default)
    prob=model.predict_proba(test[cols])[:,1]
    metrics={'auc':float(roc_auc_score(test.default,prob)),'ks':ks_stat(test.default,prob),'avg_precision':float(average_precision_score(test.default,prob)),'psi_test_vs_train_utilization':psi(train.utilization,test.utilization),'test_rows':len(test)}
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    _save(out/'credit_model.pkl',{'model':model,'features':cols})
    (out/'credit_metrics.json').write_text(json.dumps(metrics,indent=2))
    scored=test[['loan_id','portfolio','month','default']].copy();scored['risk_score']=prob;scored.to_csv(out/'credit_scores.csv',index=False)
    return metrics

def fraud(data='data',out='artifacts'):
    df=pd.read_csv(Path(data)/'transactions.csv')
    train,test=train_test_split(df,test_size=.25,random_state=42,stratify=df.fraud)
    sup=LogisticRegression(max_iter=2000,class_weight='balanced')
    scaler=StandardScaler();a=scaler.fit_transform(train[TXN_NUM]);b=scaler.transform(test[TXN_NUM])
    sup.fit(a,train.fraud)
    iso=IsolationForest(n_estimators=80,contamination=.05,random_state=42)
    iso.fit(a)
    s=sup.predict_proba(b)[:,1]; anomaly=-iso.decision_function(b)
    a_norm=(anomaly-anomaly.min())/(np.ptp(anomaly)+1e-9)
    combined=.8*s+.2*a_norm
    # Illustrative cost-based decision threshold.
    costs={str(round(t,2)):round(float(12*((combined>=t)&(test.fraud==0)).sum()+500*((combined<t)&(test.fraud==1)).sum()),2) for t in np.arange(.1,.91,.05)}
    threshold=float(min(costs,key=lambda x:costs[x]))
    alert=test[['event_id','customer_id','amount','country','fraud']].copy()
    alert['risk_score']=combined;alert['alert']=(combined>=threshold).astype(int)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    alert.sort_values('risk_score',ascending=False).to_csv(out/'fraud_alerts.csv',index=False)
    _save(out/'fraud_model.pkl',{'model':sup,'scaler':scaler,'isolation':iso,'features':TXN_NUM})
    metrics={'roc_auc':float(roc_auc_score(test.fraud,combined)),'average_precision':float(average_precision_score(test.fraud,combined)),'cost_optimal_threshold':threshold,'illustrative_cost_units':costs[str(round(threshold,2))],'fraud_rate':float(df.fraud.mean())}
    (out/'fraud_metrics.json').write_text(json.dumps(metrics,indent=2))
    return metrics

def drift(data='data',out='artifacts'):
    df=pd.read_csv(Path(data)/'loans.csv').sort_values('month')
    mid=len(df)//2;baseline=df.iloc[:mid];recent=df.iloc[mid:]
    measures={k:psi(baseline[k],recent[k]) for k in LOAN_NUM}
    result={'feature_psi':measures,'review_required':any(v>.2 for v in measures.values()),'threshold':.2,'note':'Threshold is illustrative and must be governed before production use.'}
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/'drift.json').write_text(json.dumps(result,indent=2));return result

def segment(data='data',out='artifacts'):
    df=pd.read_csv(Path(data)/'loans.csv')
    grouped=df.groupby('customer_id',as_index=False).agg(utilization=('utilization','mean'),delinquency_count=('delinquency_count','sum'),debt_to_income=('debt_to_income','mean'),loan_amount=('loan_amount','mean'))
    features=['utilization','delinquency_count','debt_to_income','loan_amount']
    scaler=StandardScaler();X=scaler.fit_transform(grouped[features]);k=KMeans(n_clusters=4,n_init=10,random_state=42)
    grouped['cluster']=k.fit_predict(X)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    grouped.to_csv(out/'segments.csv',index=False)
    profile=grouped.groupby('cluster')[features].mean().round(3).reset_index();profile.to_csv(out/'segment_profiles.csv',index=False)
    return {'customers':len(grouped),'clusters':4}

def features(data='data',out='artifacts'):
    df=pd.read_csv(Path(data)/'loans.csv');df['month']=pd.to_datetime(df.month)
    df=df.sort_values(['customer_id','month','loan_id'])
    # Rolling features strictly from prior observations; shift prevents current-row leakage.
    df['prior_avg_utilization_3']=df.groupby('customer_id')['utilization'].transform(lambda x:x.shift(1).rolling(3,min_periods=1).mean())
    df['prior_delinquency_3']=df.groupby('customer_id')['delinquency_count'].transform(lambda x:x.shift(1).rolling(3,min_periods=1).sum())
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    df[['loan_id','customer_id','month','prior_avg_utilization_3','prior_delinquency_3']].to_csv(out/'feature_store_offline.csv',index=False)
    return {'feature_rows':len(df),'note':'Offline point-in-time demo; not a deployed Feast online store.'}

def score_batch(data='data',out='artifacts'):
    bundle=_load(Path(out)/'credit_model.pkl');df=pd.read_csv(Path(data)/'loans.csv')
    df['estimated_monthly_payment_ratio']=df.loan_amount/(df.income+1)
    df['score']=bundle['model'].predict_proba(df[bundle['features']])[:,1]
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    df[['loan_id','customer_id','portfolio','month','score']].to_csv(out/'all_loan_scores.csv',index=False)
    return {'scored':len(df)}
