import re, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
STOP = set("the a an of to in on for and or is are be by with at as that this must any all from it do we how what when who can".split())
def stem(w):
    for s in ("ing", "ed", "es", "s"):
        if w.endswith(s) and len(w) - len(s) >= 3: return w[:-len(s)]
    return w
def tok(t): return [stem(w) for w in re.findall(r"[a-z0-9]+", t.lower()) if w not in STOP]
def chunk(docs, size=2, stride=1):
    out = []
    for d, txt in docs.items():
        s = re.split(r"(?<=[.!?])\s+", txt)
        for i in range(0, max(1, len(s) - size + 1), stride): out.append(dict(doc=d, text=" ".join(s[i:i+size])))
    return out
class BM25:
    def __init__(self, texts, k1=1.5, b=0.75):
        self.t = [tok(x) for x in texts]; self.k1, self.b = k1, b; self.N = len(texts)
        self.avg = np.mean([len(x) for x in self.t]); self.df = {}
        for d in self.t:
            for w in set(d): self.df[w] = self.df.get(w, 0) + 1
    def score(self, q):
        sc = np.zeros(self.N)
        for w in set(tok(q)):
            if w not in self.df: continue
            idf = np.log(1 + (self.N - self.df[w] + .5) / (self.df[w] + .5))
            for i, d in enumerate(self.t):
                f = d.count(w)
                if f: sc[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(d) / self.avg))
        return sc
class LSA:
    def __init__(self, texts, dim=25):
        self.v = TfidfVectorizer(tokenizer=tok, lowercase=False, token_pattern=None); X = self.v.fit_transform(texts)
        self.svd = TruncatedSVD(min(dim, X.shape[1] - 1), random_state=0).fit(X); self.E = self._n(self.svd.transform(X))
    def _n(self, M): return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    def score(self, q): return self.E @ self._n(self.svd.transform(self.v.transform([q])))[0]
def rrf(lists, k=60):
    s = {}
    for L in lists:
        for r, i in enumerate(L): s[i] = s.get(i, 0) + 1 / (k + r + 1)
    return sorted(s, key=lambda i: -s[i])
class Index:
    def __init__(self, docs):
        self.chunks = chunk(docs); T = [c["text"] for c in self.chunks]; self.bm, self.ls = BM25(T), LSA(T)
    def search(self, q, mode="hybrid", k=5):
        b, l = self.bm.score(q), self.ls.score(q)
        order = {"bm25": list(np.argsort(-b)), "vector": list(np.argsort(-l)),
                 "hybrid": rrf([list(np.argsort(-b)), list(np.argsort(-l))])}[mode][:k]
        return [dict(self.chunks[i], bm25=float(b[i]), cos=float(l[i])) for i in order]
def docs_in_order(hits):
    seen = []; [seen.append(h["doc"]) for h in hits if h["doc"] not in seen]; return seen
