from pathlib import Path
import json
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
D = BASE / "demo_data"
st.set_page_config(page_title="Risk Insights | AI/ML Portfolio", page_icon="📊", layout="wide")
st.markdown("""<style>.block-container{max-width:1150px;padding-top:2rem}.hero{padding:26px 30px;border-radius:18px;background:linear-gradient(115deg,#112c42,#235672);color:white}.hero h1{color:white;margin:0 0 8px;font-size:2.5rem}.hero p{color:#e3edf6;font-size:1.06rem;margin:0}</style>""",unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>Risk Insights Platform</h1><p>Financial risk analytics • Machine learning • Governed data workflows • Explainable insights</p></div>',unsafe_allow_html=True)
st.caption("Independent portfolio prototype • Entirely synthetic financial data • Not a production underwriting or fraud decision service")
@st.cache_data
def read_data():
    return tuple(json.loads((D / n).read_text()) for n in ["credit_metrics.json", "fraud_metrics.json", "drift.json", "risk_summary.json"])
credit,fraud,drift,summary=read_data()
t1,t2,t3,t4,t5=st.tabs(["Executive overview","Credit risk","Fraud detection","Model monitoring","AI & architecture"])
with t1:
 st.markdown("### End-to-end engineering demonstration")
 st.write("A reproducible portfolio combining synthetic financial data, feature preparation, risk modeling, fraud alerts, drift monitoring, a model-serving API, and controlled knowledge retrieval.")
 c1,c2,c3=st.columns(3)
 c1.metric("Credit ROC AUC", f"{credit['auc']:.3f}")
 c2.metric("Fraud ROC AUC", f"{fraud['roc_auc']:.3f}")
 c3.metric("Data source", "Synthetic")
 st.markdown("### Data flow")
 st.code("Synthetic source data → Feature processing → ML training & scoring → Risk monitoring\n                                               ↓\n                               API + policy retrieval", language="text")
 st.info("Illustrative model results on generated data. These metrics do not demonstrate suitability for real-world lending or fraud decisions.")
with t2:
 st.markdown("### Credit default prediction")
 a,b,c=st.columns(3)
 a.metric("ROC AUC", f"{credit['auc']:.3f}")
 b.metric("KS statistic", f"{credit['ks']:.3f}")
 c.metric("Average precision", f"{credit['avg_precision']:.3f}")
 st.write("Features include utilization, delinquency and debt-to-income measures. The codebase includes local training and a FastAPI model serving component.")
 st.markdown("**Engineering focus:** validation, data leakage checks, holdout metrics, and reproducible model artifacts.")
with t3:
 st.markdown("### Ranked synthetic transaction alerts")
 alerts=pd.read_csv(D/'fraud_alerts.csv')
 st.dataframe(alerts.head(100), hide_index=True, use_container_width=True)
 a,b=st.columns(2)
 a.metric("Fraud ROC AUC",f"{fraud['roc_auc']:.3f}")
 b.metric("Average precision",f"{fraud['average_precision']:.3f}")
 st.caption("Synthetic alerts only, not real customer transactions.")
with t4:
 st.markdown("### Feature population stability")
 st.bar_chart(pd.Series(drift['feature_psi'],name="PSI"))
 st.write("**Drift review:**", "Flagged" if drift['review_required'] else "Not flagged using demonstration thresholds")
 st.markdown("### Customer segment profiles")
 st.dataframe(pd.read_csv(D/'segment_profiles.csv'),hide_index=True,use_container_width=True)
 st.caption("Alert thresholds are illustrative and are not institution-approved.")
with t5:
 st.markdown("### Synthetic policy source inspector")
 docs={n:(D/n).read_text() for n in ['loan_risk.txt','fraud.txt','governance.txt']}
 choice=st.selectbox("Policy document",list(docs),format_func=lambda n:n.replace('.txt','').replace('_',' ').title())
 st.text_area("Source text",docs[choice],height=160,disabled=True)
 st.markdown("""### Implementation scope
**Runnable code in the repository:** data generation, risk/fraud models, segmentation, drift calculations, FastAPI inference, text retrieval, approved SQL queries, metrics summaries, and tests.

**Deployment examples:** Docker, CI workflow, Kubernetes/OpenShift manifests, and distributed Spark job.

**Potential extensions — not claimed as deployed:** hosted model inference, embeddings/vector database, production LLM RAG, SHAP reports, managed MLflow registry, and bank-grade approvals and security.

**Guardrails:** only synthetic records; no real customer data; no arbitrary generated SQL; no autonomous credit decisions.
""")
st.divider()
st.caption("Portfolio demonstration of backend development, data engineering, applied machine learning, and production-minded architecture.")
