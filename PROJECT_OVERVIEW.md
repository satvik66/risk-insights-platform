# Risk Insights Platform — Project Overview

## What is this complete project?
Risk Insights Platform is an independent educational AI, machine-learning, and financial data engineering portfolio. It demonstrates how synthetic loan and transaction records can be processed into analytical features, used to evaluate credit and transaction risk models, monitored for distribution drift, and explored through an interactive Streamlit dashboard and a financial policy RAG assistant. It is not a production bank product and is not affiliated with Wells Fargo.

## What can users do?
- Executive overview: view illustrative credit and fraud modeling metrics.
- Credit risk analytics: inspect credit default model evaluation and risk-scoring concepts.
- Fraud analytics: examine scored synthetic transaction alerts; a risk score is not evidence of wrongdoing.
- Model monitoring: examine feature population stability index (PSI), sample drift signals, and customer segmentation.
- Enterprise Financial RAG Assistant: ask questions about synthetic finance, fraud, and model governance policies; inspect retrieved citations. The RAG app uses local lexical BM25-style retrieval and optionally the OpenAI Responses API for source-grounded answers when configured. It accepts text-based PDF, Markdown and TXT uploads.
- Backend demonstration: repository includes FastAPI example prediction endpoints and basic automated tests.

## How is it built?
Synthetic data generator in Python → feature engineering in pandas (optional PySpark example) → scikit-learn credit/fraud/segmentation prototypes → stored demonstration artifacts → Streamlit dashboard. A separate RAG workflow extracts and chunks documents, retrieves relevant text, and optionally sends the excerpts to an OpenAI model to draft a cited response. GitHub Actions, Docker and Kubernetes/OpenShift example manifests illustrate delivery architecture but do not establish a production cloud deployment.

## Key technologies
Python, pandas, SQL/SQLite, scikit-learn, optional PySpark, FastAPI, Streamlit, OpenAI Responses API, local BM25-style retrieval, Docker, GitHub Actions and deployment manifest examples.

## Limits and responsible presentation
All financial records and example policies are synthetic. ML evaluation numbers are illustrative and should not be interpreted as production-level performance or suitability for credit, fraud, or lending decisions. The policy assistant is a prototype without enterprise authorization or confidential-data handling. It does not implement a vector database, dense embedding retrieval, production MLOps registry, or a fully autonomous financial agent. Only use public/synthetic uploads.

## Intended audience and value
A hiring manager can review the code and working dashboard to evaluate the developer's understanding of data pipelines, risk analytics, API design, information retrieval, LLM grounding and system deployment patterns. The strongest takeaway is end-to-end engineering breadth, not the predictive accuracy of the synthetic models.

## My implementation narrative and enterprise AI vision
I built this project to bridge software and data engineering with applied AI: reproducible synthetic financial datasets and feature pipelines feed credit and fraud scoring experiments, validation and drift metrics; these results appear in a Streamlit dashboard alongside a financial-document RAG assistant using local BM25-style retrieval and optional OpenAI Responses API grounded answers. The Advanced ML Lab supports temporal credit validation, portfolio risk-change sensitivity analysis, and BM25/LSA/hybrid retrieval comparisons.

My proposed enterprise direction is evidence-grounded risk and financial-planning copilots, governed policy search, and data operations assistants. Production adoption would require identity and authorization controls, vetted internal data sources, privacy and prompt-injection protection, evaluations, monitoring, and human approval. The repository does not claim that these future integrations are already deployed.
