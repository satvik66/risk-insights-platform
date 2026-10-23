import json
from pathlib import Path
from risk_platform import data,core,analytics
from fastapi.testclient import TestClient
from risk_platform.api import app

def test_pipeline(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    data.generate(n=450)
    c=core.credit();f=core.fraud();d=core.drift();s=core.segment();feature=core.features();score=core.score_batch()
    assert .5<=c['auc']<=1
    assert .0<=f['roc_auc']<=1
    assert len(d['feature_psi'])==5
    assert s['clusters']==4 and feature['feature_rows']==450 and score['scored']==450
    db=analytics.initialize_sqlite()
    result=analytics.nl2sql('loan counts by portfolio',db)
    assert sum(x['loan_count'] for x in result['rows'])==450
    assert 'error' in analytics.nl2sql('DELETE FROM loans',db)
    assert analytics.rag('fraud velocity')['citations']
    assert analytics.hybrid('loan risk policy')
    assert 'narrative' in analytics.summarize()
    assert analytics.agent('default rate by portfolio')['tool']=='read_only_sql'
    client=TestClient(app)
    assert client.get('/healthz').status_code==200
    assert client.get('/readyz').status_code==200
    assert client.post('/predict',json={'income':70000,'utilization':.55,'delinquency_count':0,'debt_to_income':.31,'loan_amount':25000}).status_code==200

def test_psi():
    assert core.psi([1,2,3,4,5]*10,[1,2,3,4,5]*10)<.001
