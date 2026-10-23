# Risk Insights Platform — AI/ML + MLOps + GenAI portfolio MVP

A local, executable educational portfolio implementing *working prototypes* of 14 connected risk-analytics concepts using **deterministic synthetic financial data only**. It covers credit scoring, fraud anomaly detection, drift, segmentation, point-in-time features, batch scoring, a model API, retrieval-based document Q&A, allowlisted natural-language SQL, hybrid search, source-grounded summaries, and an audited tool router.

**Not a real banking system**: no customer records, no model approvals, no real cloud deployment. Some requested technologies (MLflow, Feast, SHAP, OpenShift, live LLM/vector search, Argo) are documented extension points rather than falsely described as implemented. See [docs/projects.md](docs/projects.md).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m risk_platform.cli all
pytest -q
```

## Run each portfolio project

```bash
python -m risk_platform.cli generate
python -m risk_platform.cli credit
python -m risk_platform.cli fraud
python -m risk_platform.cli drift
python -m risk_platform.cli segments
python -m risk_platform.cli features
python -m risk_platform.cli batch-score
python -m risk_platform.cli rag --question 'What does fraud policy say about velocity?'
python -m risk_platform.cli hybrid --question 'Loan risk governance'
python -m risk_platform.cli sql-init
python -m risk_platform.cli nl2sql --question 'default rate by portfolio'
python -m risk_platform.cli summary
python -m risk_platform.cli agent --question 'What is the fraud procedure?'
```

Run all in one command: `python -m risk_platform.cli all`. This creates local data in `data/`, trained model/metrics/artifacts in `artifacts/`.

## Model serving demo

```bash
uvicorn risk_platform.api:app --reload
# GET http://127.0.0.1:8000/docs
# POST /predict with {"income":70000,"utilization":0.55,"delinquency_count":0,"debt_to_income":0.31,"loan_amount":25000}
```

Container: `docker build -t risk-model-api . && docker run -p 8000:8000 risk-model-api` (run `cli all` beforehand to create the demo model and data). Kubernetes and Spark Operator sample templates are in `deploy/` and require cluster-specific adaptation.

## Optional PySpark job

```bash
pip install pyspark
python -m risk_platform.spark_jobs --input data/loans.csv --output artifacts/spark_features
```

Requires a working Java/Spark environment. The default local demo uses pandas/scikit-learn so it can run without Java or a Spark cluster.

## End-to-end capstone flow

Synthetic loans and transactions → features → credit model / anomaly model → scoring CSVs and risk metrics → governed local SQLite query views → policy document retrieval and hybrid search → grounded risk summary → audited tool-routing assistant.

## Engineering choices

- No external model credentials needed; retrieval returns source text, not generated LLM answers.
- NL-to-SQL is a fixed allowlist, **not arbitrary SQL generation**, and opens the demo database read-only.
- Scores/thresholds and model metrics are illustrative, not validated for real credit decisions.
- Synthetic ground-truth labels are artificially generated; metrics are not transferable to any institution.
- API does **not** authorize financial transactions or credit decisions.
- CI runs tests and the full synthetic pipeline.

See [docs/projects.md](docs/projects.md) for an honest implementation coverage matrix and production build-out steps.

## Optional visual dashboard and MLflow metrics tracking

```bash
pip install streamlit mlflow
streamlit run risk_platform/dashboard.py
python -m risk_platform.tracking
mlflow ui
```

The dashboard displays saved local metrics, synthetic alert records, PSI, and segment profiles. MLflow metric logging is optional; local MLflow execution was not part of the base-environment test.


## Shareable public demo
For a recruiter-friendly Streamlit dashboard, see [DEPLOY_ONLINE.md](DEPLOY_ONLINE.md). The entrypoint is `streamlit_app.py` and all display data is synthetic.
