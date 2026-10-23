from pathlib import Path
from risk_platform.financial_rag import Passage, answer, document_passages, retrieve


def test_sample_documents_retrieve_relevant_evidence():
    base = Path(__file__).resolve().parents[1] / 'sample_policies'
    docs = []
    for p in base.glob('*.md'):
        docs += document_passages(p.name, p.read_bytes())
    results = retrieve('What does the policy say about PSI model drift?', docs)
    assert results
    assert any('model_governance' in p.source for p in results)
    assert '[' in results[0].citation


def test_unsupported_query_does_not_hallucinate():
    docs = document_passages('policy.md', b'Fraud analysts must review alerts.')
    assert not retrieve('quantum submarine propulsion', docs)
    response, mode = answer('quantum submarine propulsion', [])
    assert mode == 'no evidence'
    assert 'couldn\'t find' in response


def test_no_key_mode_shows_exact_passages():
    docs = [Passage('sample.txt', 2, 1, 'Monthly credit risk reviews are mandatory.', 2.1)]
    response, mode = answer('How often are reviews conducted?', docs)
    assert mode == 'retrieval only'
    assert 'sample.txt' in response and 'Monthly' in response


def test_rejects_unsupported_file_and_size():
    import pytest
    with pytest.raises(ValueError):
        document_passages('payload.exe', b'abc')
    with pytest.raises(ValueError):
        document_passages('big.txt', b'a' * (2_000_001))
