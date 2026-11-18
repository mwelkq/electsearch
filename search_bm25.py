"""
Пошук BM25 по фрагментах сторінок (консольний прототип).

  python search_bm25.py                      # інтерактивно
  python search_bm25.py "де розклад занять"  # одне питання
  python search_bm25.py --no-lemma "..."     # без лематизації (для експерименту H4)
"""

import argparse
import json
import re
import sys

import pymorphy3
from rank_bm25 import BM25Okapi

TOKEN_RE = re.compile(r"[a-zа-яіїєґ0-9']+", re.IGNORECASE)
STOP = set("і й та в у на з із до від по за що як це є а але чи не для про при або його її їх їм ми ви вони він вона воно який яка яке які де коли куди хто то так тут там"
           .split())

_morph = pymorphy3.MorphAnalyzer(lang="uk")
_cache = {}


def tokenize(text, lemma=True):
    out = []
    for w in TOKEN_RE.findall(text.lower().replace("’", "'")):
        if w in STOP or len(w) < 2:
            continue
        if lemma:
            if w not in _cache:
                _cache[w] = _morph.parse(w)[0].normal_form
            w = _cache[w]
        out.append(w)
    return out


class BM25Search:
    def __init__(self, chunks_path="data/chunks_v2.jsonl", lemma=True, k1=1.5, b=0.75, title_weight=1.0):
        with open(chunks_path, encoding="utf-8") as f:
            self.chunks = [json.loads(l) for l in f if l.strip()]
        self.lemma = lemma
        # заголовок додаємо до тексту фрагмента: він важливий для навігації
        # (v2: context = шлях розділу + заголовок; у старих індексах його немає -> title)
        corpus = [tokenize(c.get("context", c["title"]) + " " + c["text"], lemma) for c in self.chunks]
        self.bm25 = BM25Okapi(corpus, k1=k1, b=b)
        # Окреме поле «заголовок» (спрощений BM25F): BM25 по коротких рядках
        # «шлях розділу + заголовок», одна на сторінку. Оцінка сторінки =
        # найкращий фрагмент + title_weight * збіг із заголовком. Допомагає
        # сторінкам-хабам («Лабораторії», «Кафедри»), де текст — лише перелік.
        self.title_weight = title_weight
        self.page_ids = sorted({c["page_id"] for c in self.chunks})
        first = {}
        for c in self.chunks:
            first.setdefault(c["page_id"], c)
        self.page_first = first
        self.title_bm25 = BM25Okapi([tokenize(first[p].get("context", first[p]["title"]), lemma) or ["_"]
                                     for p in self.page_ids], k1=k1, b=b)

    def search(self, query, top_pages=5):
        q = tokenize(query, self.lemma)
        scores = self.bm25.get_scores(q)
        best = {}  # page_id -> (score, chunk) : оцінка сторінки = найкращий її фрагмент
        for s, c in zip(scores, self.chunks):
            if s > 0 and (c["page_id"] not in best or s > best[c["page_id"]][0]):
                best[c["page_id"]] = (s, c)
        if self.title_weight:
            for pid, ts in zip(self.page_ids, self.title_bm25.get_scores(q)):
                if ts > 0:
                    s0, c0 = best.get(pid, (0.0, self.page_first[pid]))
                    best[pid] = (s0 + self.title_weight * ts, c0)
        ranked = sorted(best.values(), key=lambda x: -x[0])[:top_pages]
        return [{"score": float(s), "title": c["title"], "url": c["url"],
                 "snippet": c["text"][:160].replace("\n", " "), "text": c["text"]} for s, c in ranked]


def show(results):
    if not results:
        print("  нічого не знайдено")
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r['score']:.2f}] {r['title']}\n     {r['url']}\n     {r['snippet']}…")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="*")
    ap.add_argument("--chunks", default="data/chunks_v2.jsonl")
    ap.add_argument("--no-lemma", action="store_true")
    ap.add_argument("--title-weight", type=float, default=1.0, help="вага поля «заголовок» (0 = звичайний BM25)")
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    engine = BM25Search(args.chunks, lemma=not args.no_lemma, title_weight=args.title_weight)
    if args.query:
        show(engine.search(" ".join(args.query), args.k))
        return
    print("Введіть питання (порожній рядок — вихід)")
    while True:
        q = input("\n? ").strip()
        if not q:
            break
        show(engine.search(q, args.k))


if __name__ == "__main__":
    main()