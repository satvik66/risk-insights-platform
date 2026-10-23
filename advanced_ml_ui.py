"""Advanced financial ML lab: run reproducible synthetic-data experiments on demand."""
import json
import subprocess
import sys
from pathlib import Path
import streamlit as st

BASE = Path(__file__).resolve().parent / "advanced_ml"
OUT = BASE / "outputs"

def _run(*names):
    OUT.mkdir(exist_ok=True)
    for name in names:
        try:
            proc = subprocess.run([sys.executable, name], cwd=str(BASE), text=True, capture_output=True, timeout=180, check=False)
        except subprocess.TimeoutExpired:
            st.error(f"{name} exceeded the execution limit.")
            return False
        if proc.returncode:
            st.error(f"{name} failed. See limited diagnostic output.")
            st.code((proc.stderr or proc.stdout)[-1200:])
            return False
    return True

def render():
    st.subheader("Advanced Financial ML Lab")
    st.caption("Synthetic ML experiments; no customer data and not production-grade underwriting.")
    st.info("Runs experiments only when you click a button. Metrics are separate from the original dashboard.")
    col1,col2,col3=st.columns(3)
    with col1:
        if st.button("Run temporal credit evaluation",use_container_width=True):
            if _run("p1_credit_default.py"):
                st.success("Credit evaluation complete.")
    with col2:
        if st.button("Analyze portfolio risk change",use_container_width=True):
            if _run("p1_credit_default.py","capstone.py"):
                st.success("Risk-change analysis complete.")
    with col3:
        if st.button("Benchmark retrieval",use_container_width=True):
            if _run("p9_hybrid_search.py"):
                st.success("Retrieval benchmark complete.")
    score=OUT/"p1_scorecard.json"
    if score.exists():
        m=json.loads(score.read_text())
        st.markdown("### Out-of-time credit evaluation")
        a,b,c=st.columns(3)
        a.metric("Train AUC",f"{m['auc_train']:.3f}")
        b.metric("Validation AUC",f"{m['auc_valid']:.3f}")
        c.metric("Out-of-time AUC",f"{m['auc_test_oot']:.3f}")
        st.caption("Train months 0–7, validation 8–9, test 10–11.")
        with st.expander("Model details"):
            st.json(m)
    report=OUT/"capstone_answer.md"
    if report.exists():
        st.markdown("### Portfolio risk-change explanation")
        st.markdown(report.read_text())
        st.caption("Approximate model sensitivity, not a causal analysis.")
    st.markdown("### Search comparison")
    st.write("Benchmark BM25 keyword retrieval, TF-IDF/LSA similarity, and hybrid Reciprocal Rank Fusion on synthetic policies.")
    if st.button("Show retrieval benchmark scores"):
        p=subprocess.run([sys.executable,"p9_hybrid_search.py"],cwd=str(BASE),capture_output=True,text=True,timeout=90)
        st.code(p.stdout if p.returncode==0 else p.stderr[-1200:])
    with st.expander("More imported experiments"):
        st.write("Fraud detection; drift monitoring; segmentation; point-in-time feature tests; extractive RAG; restricted text-to-SQL; risk report verification; hybrid retrieval and rule-based tool routing.")
