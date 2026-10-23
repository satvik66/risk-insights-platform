"""Local explainable retrieval and guarded analytical workflows. No fake LLM invocation."""
from pathlib import Path
import re,sqlite3,json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def _docs(path='data/policies'):
    return [(p.name,p.read_text()) for p in sorted(Path(path).glob('*.txt'))]

def rag(question,path='data/policies',top_k=2):
    docs=_docs(path)
    if not docs:return {'answer':'No policy documents found','citations':[]}
    mat=TfidfVectorizer(stop_words='english').fit_transform([d[1] for d in docs]+[question])
    scores=cosine_similarity(mat[-1],mat[:-1]).ravel()
    inds=np.argsort(scores)[::-1][:top_k]
    citations=[{'document':docs[int(i)][0],'score':round(float(scores[i]),3),'excerpt':docs[int(i)][1].strip()} for i in inds if scores[i]>0]
    return {'answer':'Evidence extracts only (no generative model configured): '+(' '.join(c['excerpt'] for c in citations) if citations else 'No relevant evidence found.'),'citations':citations}

def hybrid(question,path='data/policies',top_k=3):
    docs=_docs(path)
    if not docs:return []
    corpus=[x[1].lower() for x in docs]; words=set(re.findall(r'[a-z]{3,}',question.lower()))
    mat=TfidfVectorizer(stop_words='english').fit_transform(corpus+[question]);dense=cosine_similarity(mat[-1],mat[:-1]).ravel()
    keyword=np.array([len(words & set(re.findall(r'[a-z]{3,}',text))) for text in corpus],dtype=float)
    if keyword.max():keyword/=keyword.max()
    scores=.65*dense+.35*keyword
    return [{'document':docs[int(i)][0],'score':round(float(scores[i]),3),'excerpt':docs[int(i)][1].strip()} for i in np.argsort(scores)[::-1][:top_k]]

def initialize_sqlite(data='data',db='artifacts/analytics.db'):
    Path(db).parent.mkdir(parents=True,exist_ok=True)
    con=sqlite3.connect(db)
    df=pd.read_csv(Path(data)/'loans.csv')
    df[['loan_id','portfolio','month','loan_amount','default','utilization']].to_sql('loans',con,if_exists='replace',index=False)
    con.close()
    return db

# Fixed intent-to-SQL catalog: safer than letting an LLM execute arbitrary SQL.
ALLOWED_QUERIES={
 'loan counts by portfolio':'SELECT portfolio, COUNT(*) AS loan_count FROM loans GROUP BY portfolio ORDER BY portfolio',
 'average utilization by portfolio':'SELECT portfolio, ROUND(AVG(utilization),4) AS avg_utilization FROM loans GROUP BY portfolio ORDER BY portfolio',
 'default rate by portfolio':'SELECT portfolio, ROUND(AVG("default"),4) AS default_rate FROM loans GROUP BY portfolio ORDER BY portfolio',
 'loan exposure by portfolio':'SELECT portfolio, ROUND(SUM(loan_amount),2) AS total_loan_amount FROM loans GROUP BY portfolio ORDER BY portfolio'
}
def nl2sql(question,db='artifacts/analytics.db'):
    key=question.lower().strip().rstrip('?')
    if key not in ALLOWED_QUERIES:return {'error':'Unsupported question. Choose a supported read-only analytical question.','allowed_questions':list(ALLOWED_QUERIES)}
    con=sqlite3.connect('file:'+str(Path(db).resolve())+'?mode=ro',uri=True)
    try:rows=pd.read_sql_query(ALLOWED_QUERIES[key],con).to_dict(orient='records')
    finally:con.close()
    return {'sql':ALLOWED_QUERIES[key],'rows':rows}

def summarize(out='artifacts'):
    p=Path(out);credit=json.loads((p/'credit_metrics.json').read_text());fraud=json.loads((p/'fraud_metrics.json').read_text());drift=json.loads((p/'drift.json').read_text())
    # Source-grounded deterministic narrative; does not hallucinate LLM narrative.
    summary={'credit_auc':credit['auc'],'credit_ks':credit['ks'],'fraud_auc':fraud['roc_auc'],'drift_review_required':drift['review_required'],'narrative':f"Synthetic portfolio: credit holdout AUC {credit['auc']:.3f}, KS {credit['ks']:.3f}; fraud evaluation AUC {fraud['roc_auc']:.3f}. Drift review {'required' if drift['review_required'] else 'not flagged'} against illustrative thresholds. These metrics do not establish production model quality."}
    (p/'risk_summary.json').write_text(json.dumps(summary,indent=2));return summary

def agent(question,data='data',out='artifacts'):
    # Audited, constrained routing demo: no arbitrary code execution, no autonomous financial decisions.
    q=question.lower();log=Path(out)/'agent_audit.jsonl';log.parent.mkdir(parents=True,exist_ok=True)
    if q.strip().rstrip('?') in ALLOWED_QUERIES:
        response=nl2sql(q, str(Path(out)/'analytics.db'));tool='read_only_sql'
    elif any(k in q for k in ['policy','fraud procedure','governance','document']):
        response=rag(question,str(Path(data)/'policies'));tool='policy_retrieval'
    elif any(k in q for k in ['risk summary','model performance','drift']):
        response=summarize(out);tool='metrics_summary'
    else:
        response={'error':'No approved tool for this request; human approval required for new actions.'};tool='blocked'
    with log.open('a') as f:f.write(json.dumps({'question':question,'tool':tool,'response_type':'error' if 'error' in response else 'success'})+'\n')
    return {'tool':tool,'response':response}
