"""
Порівняння методів пошуку: BM25, embeddings, гібрид (RRF).

  pip install sentence-transformers numpy
  python search.py "як вступити на магістратуру"
  python search.py                               # інтерактивно
  python search.py --model intfloat/multilingual-e5-base "..."    # більша, точніша, але повільніша модель

Перший запуск рахує вектори для всіх фрагментів (хвилини/десятки хвилин на CPU) і
зберігає їх по блоках у data/; можна перервати й продовжити. Далі вектори беруться з диска.
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np

from query_expansion import expand_query
from search_bm25 import BM25Search

RRF_K = 60


class EmbedSearch:
    BLOCK = 128  # після кожних 128 фрагментів зберігаємо проміжний результат (можна перервати й продовжити)

    def __init__(self, chunks, model_name="intfloat/multilingual-e5-small", cache_dir="data", max_len=256, index_tag=""):
        self.chunks = chunks
        self.model_name = model_name
        self.is_e5 = "e5" in model_name.lower()
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = max_len   # довші фрагменти обрізаються: суттєво швидше
        # index_tag відрізняє версії індексу (chunks_v2 тощо), щоб не взяти чужі вектори з тією ж кількістю фрагментів
        tag = (index_tag + "_" if index_tag else "") + model_name.replace("/", "_") + f"_{len(chunks)}_{max_len}"
        cache = Path(cache_dir) / f"emb_{tag}.npy"
        if cache.exists():
            self.emb = np.load(cache)
        else:
            texts = [c["title"] + "\n" + c["text"] for c in chunks]
            self.emb = self._encode_with_checkpoints(texts, Path(cache_dir) / f"emb_parts_{tag}")
            np.save(cache, self.emb)

    def _encode_with_checkpoints(self, texts, part_dir):
        part_dir.mkdir(parents=True, exist_ok=True)
        n_blocks = (len(texts) + self.BLOCK - 1) // self.BLOCK
        print(f"[embeddings] {len(texts)} фрагментів, {n_blocks} блоків. Можна перервати (Ctrl+C) і запустити знову: продовжить з місця зупинки.")
        parts = []
        for b in range(n_blocks):
            f = part_dir / f"part_{b:04d}.npy"
            if f.exists():
                parts.append(np.load(f))
                continue
            block = texts[b * self.BLOCK:(b + 1) * self.BLOCK]
            arr = self.encode(block, is_query=False)
            np.save(f, arr)
            parts.append(arr)
            print(f"  блок {b + 1}/{n_blocks} готово", flush=True)
        return np.vstack(parts)

    def encode(self, texts, is_query, show=False):
        if self.is_e5:  # моделі E5 вимагають префікси
            prefix = "query: " if is_query else "passage: "
            texts = [prefix + t for t in texts]
        return self.model.encode(texts, batch_size=16, normalize_embeddings=True,
                                 show_progress_bar=show).astype(np.float32)

    def search(self, query, top_pages=5):
        q = self.encode([query], is_query=True)[0]
        scores = self.emb @ q                      # косинусна подібність (вектори нормовані)
        return rank_pages(scores, self.chunks, top_pages)


def rank_pages(scores, chunks, top_pages):
    best = {}
    for s, c in zip(scores, chunks):
        if c["page_id"] not in best or s > best[c["page_id"]][0]:
            best[c["page_id"]] = (float(s), c)
    ranked = sorted(best.values(), key=lambda x: -x[0])[:top_pages]
    return [{"score": s, "title": c["title"], "context": c.get("context", c["title"]), "url": c["url"],
             "snippet": c["text"][:160].replace("\n", " "), "text": c["text"]} for s, c in ranked]


def rrf(result_lists, top_pages=5, k=RRF_K, weights=None):
    """Reciprocal Rank Fusion: score(d) = sum w_i / (k + rank_i(d)). weights=None -> усі 1."""
    agg, info = {}, {}
    weights = weights or [1.0] * len(result_lists)
    for lst, w in zip(result_lists, weights):
        for rank, r in enumerate(lst, 1):
            agg[r["url"]] = agg.get(r["url"], 0.0) + w / (k + rank)
            info[r["url"]] = r
    out = sorted(agg.items(), key=lambda x: -x[1])[:top_pages]
    return [dict(info[u], score=s) for u, s in out]


class Searcher:
    """Єдиний інтерфейс: method in {'bm25', 'embed', 'hybrid'}."""

    def __init__(self, chunks_path="data/chunks_v2.jsonl", model="intfloat/multilingual-e5-small", max_len=256,
                 title_weight=1.0, bm25_weight=1.0):
        self.bm25 = BM25Search(chunks_path, title_weight=title_weight)
        self.bm25_weight = bm25_weight  # вага BM25 у RRF (embeddings = 1)
        stem = Path(chunks_path).stem
        self.embed = EmbedSearch(self.bm25.chunks, model, max_len=max_len,
                                 index_tag="" if stem == "chunks" else stem)

    def search(self, query, method="hybrid", k=5, expand=False):
        if expand:
            query = expand_query(query)
        if method == "bm25":
            return self.bm25.search(query, k)
        if method == "embed":
            return self.embed.search(query, k)
        pool = 20  # для гібриду беремо ширший топ кожного методу
        return rrf([self.bm25.search(query, pool), self.embed.search(query, pool)], k,
                   weights=[self.bm25_weight, 1.0])


def show_all(s, query, k):
    for name, label in (("bm25", "BM25"), ("embed", "EMBEDDINGS"), ("hybrid", "ГІБРИД (RRF)")):
        print(f"\n=== {label} ===")
        res = s.search(query, name, k)
        for i, r in enumerate(res, 1):
            print(f"  {i}. [{r['score']:.3f}] {r['title']}\n     {r['url']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="*")
    ap.add_argument("--chunks", default="data/chunks_v2.jsonl")
    ap.add_argument("--model", default="intfloat/multilingual-e5-small")
    ap.add_argument("--max-len", type=int, default=256, help="макс. токенів на фрагмент")
    ap.add_argument("-k", type=int, default=3)
    ap.add_argument("--title-weight", type=float, default=1.0, help="вага заголовка в BM25 (0 = вимкнено)")
    args = ap.parse_args()
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    s = Searcher(args.chunks, args.model, args.max_len, args.title_weight)
    if args.query:
        show_all(s, " ".join(args.query), args.k)
        return
    print("Введіть питання (порожній рядок — вихід)")
    while True:
        q = input("\n? ").strip()
        if not q:
            break
        show_all(s, q, args.k)


if __name__ == "__main__":
    main()