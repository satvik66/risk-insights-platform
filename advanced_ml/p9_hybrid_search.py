"""P9: compare BM25 vs vector (LSA) vs hybrid (RRF) on the golden set."""
from kb import *; from retrieval import *
ix = Index(DOCS); print(f"{'mode':8} hit@1  hit@3  MRR")
for mode in ("bm25", "vector", "hybrid"):
    h1 = h3 = rr = 0
    for q, g in GOLDEN:
        d = docs_in_order(ix.search(q, mode, len(ix.chunks)))
        pos = d.index(g) + 1 if g in d else 99; h1 += pos == 1; h3 += pos <= 3; rr += 1/pos
    n = len(GOLDEN); print(f"{mode:8} {h1/n:.2f}   {h3/n:.2f}   {rr/n:.3f}")
