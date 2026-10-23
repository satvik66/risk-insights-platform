"""Optional Streamlit model monitoring/portfolio dashboard."""
from pathlib import Path
import json
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Risk Insights | Synthetic Demo',layout='wide')
st.title('Risk Insights Platform')
st.caption('Educational prototype using synthetic data — not a credit or fraud decision system')
a=Path('artifacts')
if not (a/'credit_metrics.json').exists():
    st.warning('Run: python -m risk_platform.cli all')
    st.stop()
credit=json.loads((a/'credit_metrics.json').read_text())
fraud=json.loads((a/'fraud_metrics.json').read_text())
drift=json.loads((a/'drift.json').read_text())
c1,c2,c3=st.columns(3)
c1.metric('Credit holdout AUC',f"{credit['auc']:.3f}")
c2.metric('Fraud holdout AUC',f"{fraud['roc_auc']:.3f}")
c3.metric('Drift feature review', 'Required' if drift['review_required'] else 'Not flagged')
st.subheader('Population Stability Index — synthetic features')
st.bar_chart(pd.Series(drift['feature_psi']).rename('PSI'))
st.subheader('Ranked transaction alerts (demo)')
st.dataframe(pd.read_csv(a/'fraud_alerts.csv').head(30),hide_index=True)
st.subheader('Customer segmentation profiles')
st.dataframe(pd.read_csv(a/'segment_profiles.csv'),hide_index=True)
st.info('Metrics and alerts are research examples. No validated thresholds, human approvals, or protected-group fairness analysis are provided.')
