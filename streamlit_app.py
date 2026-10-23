from pathlib import Path
import json
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
D = BASE / "demo_data"
# Embedded synthetic showcase samples allow cloud deployment even if the demo_data/ folder was omitted.
# Values are snapshots from the checked-in generated demo data, never live financial data.
FALLBACK_JSON = {'credit_metrics.json': {'auc': 0.6685008074395812, 'ks': 0.27350623677469654, 'avg_precision': 0.429380503131696, 'psi_test_vs_train_utilization': 0.0050998369737147164, 'test_rows': 875}, 'fraud_metrics.json': {'roc_auc': 0.6490468568102444, 'average_precision': 0.11724569565309485, 'cost_optimal_threshold': 0.7, 'illustrative_cost_units': 13436.0, 'fraud_rate': 0.018571428571428572}, 'drift.json': {'feature_psi': {'income': 0.007735480276675236, 'utilization': 0.0044682013680245585, 'delinquency_count': 0.00034454610321857306, 'debt_to_income': 0.008051297932260362, 'loan_amount': 0.009080496738971744}, 'review_required': False, 'threshold': 0.2, 'note': 'Threshold is illustrative and must be governed before production use.'}, 'risk_summary.json': {'credit_auc': 0.6685008074395812, 'credit_ks': 0.27350623677469654, 'fraud_auc': 0.6490468568102444, 'drift_review_required': False, 'narrative': 'Synthetic portfolio: credit holdout AUC 0.669, KS 0.274; fraud evaluation AUC 0.649. Drift review not flagged against illustrative thresholds. These metrics do not establish production model quality.'}}
FALLBACK_ALERTS_CSV = 'event_id,customer_id,amount,country,fraud,risk_score,alert\n5425,475,6012.0,CA,1,0.9814317432114418,1\n4646,615,4225.28,CA,0,0.9742932125335224,1\n4833,310,2533.51,CA,0,0.9590661620370928,1\n6295,214,3384.96,CA,1,0.9579773955179944,1\n6748,230,75.05,MX,1,0.9403493695108448,1\n6095,374,1328.15,US,0,0.9216930163021734,1\n2982,603,1669.27,MX,0,0.914715308744835,1\n6795,131,1448.17,MX,1,0.888623372177487,1\n3407,504,2076.03,MX,0,0.8830700119983879,1\n6347,414,977.01,CA,0,0.8760476108571821,1\n5831,469,737.41,US,0,0.8638647204913501,1\n5149,601,287.62,US,0,0.8599818406265609,1\n3546,305,68.86,US,0,0.853516998388395,1\n6584,328,91.1,CA,0,0.843767967258624,1\n2794,601,174.72,MX,0,0.8392612259244512,1\n6552,470,1741.0,MX,0,0.8364792474474639,1\n2113,513,48.27,CA,0,0.8313653469373622,1\n4244,407,584.03,CA,0,0.8287831710742009,1\n5500,173,990.29,MX,0,0.8267154363777413,1\n4956,188,819.36,US,0,0.8231435506422835,1\n934,649,235.87,US,0,0.8025585067128231,1\n3177,56,285.99,CA,0,0.8011418748776534,1\n1559,175,357.21,CA,0,0.8007099283266335,1\n2079,239,49.5,CA,0,0.8000120951176384,1\n120,470,94.56,US,0,0.7977264321951949,1\n6042,322,1270.92,MX,0,0.7975279386194651,1\n140,488,23.64,CA,0,0.7939250050927272,1\n6673,14,901.98,MX,0,0.7922470381249191,1\n6190,602,111.92,US,1,0.7917189777562792,1\n6805,576,313.67,CA,0,0.7895844290262686,1\n6540,120,119.02,US,0,0.7859354499831117,1\n888,492,164.46,MX,0,0.784227032632787,1\n5989,393,45.93,US,0,0.7841343919936008,1\n2591,435,332.03,MX,0,0.7780688092475372,1\n6602,114,296.41,MX,0,0.7746410129290972,1\n5669,637,978.03,MX,0,0.7721253537033667,1\n5702,188,230.72,US,0,0.7716142152112224,1\n2100,699,152.86,MX,0,0.7705043861342815,1\n6007,194,822.6,US,0,0.7686503529902122,1\n3858,630,1194.87,CA,0,0.7683470297675621,1\n5097,604,3.56,CA,1,0.7669782203837227,1\n4669,191,17.12,CA,0,0.7653491658980811,1\n5174,607,345.3,MX,0,0.7651262002456626,1\n3644,390,156.65,MX,0,0.763192382143143,1\n205,385,266.36,CA,0,0.7608153761360003,1\n4343,36,266.11,MX,0,0.758645707100031,1\n1094,318,22.88,CA,0,0.7584212688271621,1\n4754,220,37.93,MX,0,0.7570854899070467,1\n6971,469,220.33,MX,0,0.7538696697665784,1\n2447,88,464.5,CA,0,0.75299182428073,1\n257,694,238.63,MX,0,0.7529183924727777,1\n6032,600,48.1,CA,0,0.7523327918627989,1\n4549,189,88.6,CA,0,0.7514787238350658,1\n4053,25,25.19,US,0,0.7513265263583228,1\n149,269,190.57,US,0,0.7512069245187062,1\n4748,555,272.79,MX,0,0.7485617420259342,1\n1990,350,245.22,US,0,0.7466051493037923,1\n5595,106,476.93,CA,0,0.7460622057111069,1\n6128,559,127.8,CA,0,0.7451664476354886,1\n2516,553,102.56,MX,0,0.7438862064950754,1\n245,404,348.52,CA,0,0.7420916123446182,1\n1633,165,416.04,CA,0,0.7410001638428798,1\n6813,548,128.01,US,0,0.7379453941554543,1\n3763,588,1014.49,US,0,0.7377456015340373,1\n265,60,549.39,CA,0,0.7368537340777499,1\n2393,320,1067.97,CA,0,0.7343223567737207,1\n2493,268,110.48,CA,0,0.7287120465141642,1\n3243,407,1402.42,US,0,0.7268875590812597,1\n6072,200,59.24,MX,0,0.7256051113436159,1\n2437,232,11.2,US,0,0.7216798441894114,1\n420,41,51.23,CA,0,0.7212604328876864,1\n4249,305,23.07,US,0,0.7192192267107409,1\n6893,426,437.25,CA,0,0.7183507410832565,1\n3733,437,9.52,CA,1,0.718189800518772,1\n3579,387,1494.67,MX,0,0.7173123236255703,1\n2190,225,173.15,CA,0,0.7130232034772033,1\n1557,685,91.1,US,0,0.7122383246231372,1\n5055,537,267.02,MX,0,0.7102430706395703,1\n3623,478,7.41,US,0,0.7093364348317488,1\n3163,72,98.85,CA,0,0.7083760207234427,1\n343,345,407.55,MX,0,0.7069441435731789,1\n4618,479,27.12,MX,0,0.7063968765246957,1\n4152,167,94.89,MX,0,0.706220801528282,1\n6832,414,169.37,MX,0,0.7034085083664371,1\n6991,443,226.95,CA,0,0.7004204142987367,1\n3570,537,184.19,MX,0,0.6993093175544501,0\n6225,278,244.57,MX,0,0.6919747061904016,0\n4900,512,1265.61,CA,0,0.6910608346341299,0\n3592,516,142.04,MX,0,0.6895844611777675,0\n3646,689,37.08,US,0,0.6892645409132238,0\n1403,523,35.39,CA,0,0.6888400977761804,0\n1993,196,682.48,MX,0,0.6873638605851835,0\n6316,286,57.48,US,0,0.6872966805558317,0\n4336,620,43.37,MX,0,0.6869774747123821,0\n1735,125,615.85,CA,0,0.6860946235296318,0\n1099,408,32.23,CA,0,0.6860546455958598,0\n1886,657,785.86,CA,0,0.6858246894761596,0\n5235,628,882.67,MX,0,0.6856217074707184,0\n6166,184,188.32,CA,0,0.6848738817683251,0\n3815,92,22.61,CA,0,0.6843245991753926,0\n'
FALLBACK_SEGMENTS_CSV = 'cluster,utilization,delinquency_count,debt_to_income,loan_amount\n0,0.323,1.153,0.315,55514.037\n1,0.486,1.209,0.269,41302.891\n2,0.415,3.94,0.327,46052.577\n3,0.454,1.076,0.424,45925.698\n'
FALLBACK_DOCS = {'loan_risk.txt': 'Loan Risk Policy: Review auto-loan portfolios monthly. Escalate delinquency concentration above approved thresholds. Credit risk decisions must have reason codes, an audit trail and human oversight.\n', 'fraud.txt': 'Fraud Operations Procedure: Prioritize unusual transaction amounts, changes in devices and high transaction velocity. A model score is an alert, not proof of wrongdoing. Analysts must review alerts before action.\n', 'governance.txt': 'Model Governance: Monitor drift, calibration and false positive rates. Review demographic fairness with properly authorized data. Do not use the LLM to authorize a credit or fraud decision.\n'}

def read_demo_text(name):
    path = D / name
    if path.is_file():
        return path.read_text(encoding='utf-8')
    if name in FALLBACK_JSON:
        return json.dumps(FALLBACK_JSON[name])
    if name == 'fraud_alerts.csv':
        return FALLBACK_ALERTS_CSV
    if name == 'segment_profiles.csv':
        return FALLBACK_SEGMENTS_CSV
    if name in FALLBACK_DOCS:
        return FALLBACK_DOCS[name]
    raise FileNotFoundError(f'Unknown demo asset: {name}')

def read_demo_csv(name):
    from io import StringIO
    return pd.read_csv(StringIO(read_demo_text(name)))

st.set_page_config(page_title="Risk Insights | AI/ML Portfolio", page_icon="📊", layout="wide")
st.markdown("""<style>.block-container{max-width:1150px;padding-top:2rem}.hero{padding:26px 30px;border-radius:18px;background:linear-gradient(115deg,#112c42,#235672);color:white}.hero h1{color:white;margin:0 0 8px;font-size:2.5rem}.hero p{color:#e3edf6;font-size:1.06rem;margin:0}</style>""",unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>Risk Insights Platform</h1><p>Financial risk analytics • Machine learning • Governed data workflows • Explainable insights</p></div>',unsafe_allow_html=True)
st.caption("Independent portfolio prototype • Entirely synthetic financial data • Not a production underwriting or fraud decision service")
@st.cache_data
def read_data():
    return tuple(json.loads(read_demo_text(n)) for n in ["credit_metrics.json", "fraud_metrics.json", "drift.json", "risk_summary.json"])
credit,fraud,drift,summary=read_data()
t1,t2,t3,t4,t5,t6=st.tabs(["Executive overview","Credit risk","Fraud detection","Model monitoring","AI & architecture","Financial RAG Assistant"])
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
 alerts=read_demo_csv('fraud_alerts.csv')
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
 st.dataframe(read_demo_csv('segment_profiles.csv'),hide_index=True,use_container_width=True)
 st.caption("Alert thresholds are illustrative and are not institution-approved.")
with t5:
 st.markdown("### Synthetic policy source inspector")
 docs={n:read_demo_text(n) for n in ['loan_risk.txt','fraud.txt','governance.txt']}
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

with t6:
 from risk_platform.rag_ui import render
 render()
