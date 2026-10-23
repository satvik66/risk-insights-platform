# Enterprise Financial RAG Assistant — deployment

This add-on adds a **Financial RAG Assistant** tab to `streamlit_app.py` using three synthetic policy documents.

## Local quickstart

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Without a key it performs **real document retrieval** and returns the exact supporting excerpts. For LLM synthesis, set `OPENAI_API_KEY` in your local environment (never commit a `.env` or API key).

## OpenAI API setup

1. Go to https://platform.openai.com/api-keys and create a project API key. API billing is separate from ChatGPT subscriptions.
2. In Streamlit Community Cloud, select your app > **Manage app** > **Settings** > **Secrets**.
3. Paste the secret TOML value below, replacing the placeholder with your real key (NEVER paste a key into GitHub or this chat):

```toml
OPENAI_API_KEY = "YOUR_API_KEY_HERE"
```

4. Save secrets/reboot the app if necessary. Open **Financial RAG Assistant**.

## Suggested test questions

- What does the credit policy say about escalation?
- What must fraud analysts record in an audit trail?
- What does model governance require before promotion?
- What is the numeric PSI escalation threshold? (the documents do **not** specify a numeric universal threshold; the model should abstain.)

## Design and limitations

- Synthetic demonstration policies only, NOT financial or legal advice.
- Accepts PDF/TXT/MD uploads and parses text locally on the Streamlit backend.
- Uses **local BM25-style keyword retrieval**; no embeddings or vector database yet.
- When an API key is configured, calls `OpenAI().responses.create(...)` **server-side**, sends only retrieved excerpts as context, sets `store=False`, and instructs the model to cite exact passage labels.
- Document passages shown below each answer make the grounding auditable. Citation correctness is prompted and should still be manually checked.
- No multiuser isolation, access control, adversarial retrieval hardening, persistent storage, or production audit logging. Not for confidential bank data or regulated workloads.
- API requests may incur OpenAI usage fees. Hosted mode needs a valid API key and API credits.
- Retrieval-only mode is functional without an API key.
