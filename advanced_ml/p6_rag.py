"""P6: RAG over policy docs: chunk -> hybrid retrieval (BM25 + LSA vectors, RRF) -> cited extractive answer, abstention, eval."""
from kb import *; from retrieval import *
def answer(ix, q, k=3):
    hits = ix.search(q, "hybrid", k)
    if max(h["bm25"] for h in hits) == 0 and max(h["cos"] for h in hits) < 0.35:
        return dict(answer="No supported answer in the knowledge base.", sources=[], abstained=True)
    return dict(answer=hits[0]["text"], sources=docs_in_order(hits)[:2], abstained=False)
if __name__ == "__main__":
    ix = Index(DOCS); print("chunks:", len(ix.chunks))
    h1 = h3 = 0
    for q, gold in GOLDEN:
        r = ix.search(q, "hybrid", 6); d = docs_in_order(r); h1 += d[0] == gold; h3 += gold in d[:3]
    print(f"hit@1={h1/len(GOLDEN):.2f} hit@3={h3/len(GOLDEN):.2f} over {len(GOLDEN)} golden questions")
    for q in ["What DTI limit applies for credit card approval?", "how quickly must a breach be reported"] + OUT_OF_SCOPE:
        a = answer(ix, q); print(f"\nQ: {q}\nA: {a['answer']}\nsources: {a['sources']} abstained={a['abstained']}")
