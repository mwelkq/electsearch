"""Парне порівняння точності двох RAG-прогонів: bootstrap 95% і точний тест Мак-Немара.
  python tools/rag_compare.py rag_hybrid_lite_v2tw1 rag_hybrid_lite_v3_k5
"""
import json, sys
from math import comb
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
from evaluate import load_alias_map

canon = load_alias_map(str(ROOT / "data/chunks.jsonl")); canon.update(load_alias_map(str(ROOT / "data/chunks_v2.jsonl")))
Q = {q["id"]: q for q in json.load(open(ROOT / "eval/queries.json", encoding="utf-8"))}

def load(tag):
    out = {}
    for l in open(ROOT / f"eval/rag_{tag}.jsonl", encoding="utf-8"):
        r = json.loads(l); q = Q[r["id"]]
        if not q["relevant"] or "error" in r: continue
        rel = {canon.get(u, u) for u in q["relevant"]}; u = r.get("recommended_url")
        out[r["id"]] = 1.0 if u and canon.get(u, u) in rel else 0.0
    return out

a, b = load(sys.argv[1]), load(sys.argv[2]); ids = sorted(set(a) & set(b))
d = np.array([b[i] - a[i] for i in ids]); rng = np.random.default_rng(42)
s = d[rng.integers(0, len(d), (10000, len(d)))].mean(1); lo, hi = np.percentile(s, [2.5, 97.5])
n01 = sum(b[i] > a[i] for i in ids); n10 = sum(b[i] < a[i] for i in ids); n = n01 + n10
p = min(1.0, 2 * sum(comb(n, k) for k in range(min(n01, n10) + 1)) / 2 ** n) if n else 1.0
print(f"{sys.argv[1]}: {np.mean([a[i] for i in ids]):.3f}  ->  {sys.argv[2]}: {np.mean([b[i] for i in ids]):.3f}  (n={len(ids)})")
print(f"різниця {d.mean():+.3f}  bootstrap 95% [{lo:+.3f}; {hi:+.3f}]  Мак-Немар p={p:.4f} ({n01}:{n10})")
print("виправлено:", [i for i in ids if b[i] > a[i]], " зіпсовано:", [i for i in ids if b[i] < a[i]])
