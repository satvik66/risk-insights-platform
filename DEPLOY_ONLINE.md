# Deploy the Risk Insights interactive portfolio

## Streamlit Community Cloud (recommended)
1. Create a GitHub repository named `risk-insights-platform`.
2. Upload the contents of the repository ZIP, preserving folders (`streamlit_app.py`, `requirements.txt`, `demo_data/`, etc.).
3. Open https://share.streamlit.io/ and sign in using GitHub.
4. Select **Create app** and choose your GitHub repository and branch.
5. Set the main file path to `streamlit_app.py`, then deploy.
6. Open the generated `.streamlit.app` address, verify all tabs, and share that public link with the hiring manager.

The cloud dashboard uses small **precomputed synthetic demo data**, so hosting does not require real customer information, model credentials, or training at startup. This is a dashboard, not a hosted inference API. The repository includes the original model training and FastAPI code separately.

Do not upload `.env` files, real financial data, cloud credentials or proprietary employer code. Public sharing should make clear this is an independent demo, not a Schwab or Wells Fargo system.

## Run locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
