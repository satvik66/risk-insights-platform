"""Streamlit RAG tab UI for the Risk Insights Platform."""
import os
from pathlib import Path
import streamlit as st
from risk_platform.financial_rag import document_passages, retrieve, answer

BASE = Path(__file__).resolve().parent.parent
SAMPLE = BASE / 'sample_policies'


def _key():
    try:
        return st.secrets.get('OPENAI_API_KEY') or os.getenv('OPENAI_API_KEY')
    except Exception:
        return os.getenv('OPENAI_API_KEY')


def render():
    st.subheader('Enterprise Financial RAG Assistant')
    st.caption('Interactive, source-cited Q&A over synthetic financial policy documents. Educational demo only.')
    st.warning('Upload only public or synthetic content. Uploaded text may be sent to the OpenAI API when AI mode is enabled. Never upload real customer records or confidential policies.')
    source_files = sorted(SAMPLE.glob('*.md'))
    # Include a repository-grounded overview so project-level questions can be answered.
    project_overview = BASE / 'PROJECT_OVERVIEW.md'
    if project_overview.is_file():
        source_files = [project_overview] + source_files
    chosen = st.multiselect('Sample knowledge base', [p.name for p in source_files], default=[p.name for p in source_files])
    uploaded = st.file_uploader('Optional additional document (TXT, MD, text-based PDF; max 2 MB, PDF 30 pages)', type=['txt','md','pdf'], accept_multiple_files=True)
    passages = []
    errors = []
    for p in source_files:
        if p.name in chosen:
            passages.extend(document_passages(p.name, p.read_bytes()))
    for f in uploaded or []:
        try:
            passages.extend(document_passages(f.name, f.getvalue()))
        except Exception as exc:
            errors.append(f'{f.name}: {exc}')
    for err in errors:
        st.error(err)
    key = _key()
    st.info('AI answer generation enabled using OpenAI Responses API.' if key else 'Retrieval-only demo mode. Add OPENAI_API_KEY to Streamlit secrets to enable AI-generated answers.')
    q = st.text_input('Ask about the project or financial policies', placeholder='What is this complete project?')
    st.caption('Try: What is this complete project? · How does the RAG assistant work? · What triggers credit risk escalation?')
    if st.button('Search documents & answer', type='primary', disabled=not q.strip() or not passages):
        evidence = retrieve(q, passages)
        try:
            output, mode = answer(q, evidence, api_key=key)
            st.markdown(output)
            st.caption(f'Mode: {mode} · {len(evidence)} supporting passage(s)')
        except Exception:
            st.error('The OpenAI API request failed. Check your API key, API credits, model availability, and app logs. No key details are displayed.')
        if evidence:
            with st.expander('Inspect retrieved evidence and citation labels', expanded=True):
                for p in evidence:
                    st.markdown(f'**{p.citation}** · relevance {p.score:.2f}')
                    st.write(p.text)
    if not passages:
        st.warning('Select sample documents or upload a valid document to start.')
    with st.expander('How this RAG system works'):
        st.markdown('**Documents → Text extraction → Overlapping chunks → Local BM25-style retrieval → Grounded OpenAI answer → Citations.**\n\nNo key: sources are shown without LLM synthesis. With a key: selected excerpts are sent server-side to the OpenAI Responses API. This demo does not store conversations in a database, provide authorization, or replace professional risk review.')
