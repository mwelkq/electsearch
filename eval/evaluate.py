"""
Оцінка методів пошуку на тестовому наборі запитів.

  python eval/evaluate.py                       # усі три методи
  python eval/evaluate.py --methods bm25        # лише BM25 (не потребує моделі)
  python eval/evaluate.py --model intfloat/multilingual-e5-base

Метрики (на рівні сторінок), лише для запитів, на які на сайті є відповідь:
  Hit@k   — частка запитів, де хоча б одна правильна сторінка у топ-k (k=1,3,5)
  MRR     — середнє 1/rank першої правильної сторінки (rank у топ-10; не знайдено = 0)
  Recall@5— частка правильних сторінок, що потрапили у топ-5

Запити-«пастки» (relevant = []) в метрики не входять; для них окремо виводиться
топ-1 і оцінка (щоб було видно, наскільки впевнено метод «знаходить» неіснуюче).
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

K = 10


def load_alias_map(chunks_path):
    """url-аліас -> канонічний url (після відсіву дублікатів у process.py)."""
    canon = {}
    seen = set()
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if c["url"] in seen:
                continue
            seen.add(c["url"])
            canon[c["url"]] = c["url"]
            for a in c.get("aliases", []):
                canon[a] = c["url"]
    return canon


def evaluate(searcher, queries, method, canon, expand=False):
    rows = []
    for q in queries:
        res = searcher.search(q["query"], method, K, expand)
        got = [r["url"] for r in res]
        rel = {canon.get(u, u) for u in q["relevant"]}
        ranks = [i for i, u in enumerate(got, 1) if canon.get(u, u) in rel]
        rows.append({
            "id": q["id"], "category": q["category"], "query": q["query"],
            "n_rel": len(rel), "first_rank": ranks[0] if ranks else None,
            "found_in_top5": len({canon.get(u, u) for u in got[:5]} & rel),
            "top1": got[0] if got else None,
            "top1_score": res[0]["score"] if res else None,
        })
    return rows


def summarize(rows):
    answerable = [r for r in rows if r["n_rel"] > 0]
    n = len(answerable) or 1
    hit = lambda k: sum(1 for r in answerable if r["first_rank"] and r["first_rank"] <= k) / n
    mrr = sum(1 / r["first_rank"] for r in answerable if r["first_rank"]) / n
    rec5 = sum(r["found_in_top5"] / r["n_rel"] for r in answerable) / n
    return {"n": len(answerable), "Hit@1": hit(1), "Hit@3": hit(3), "Hit@5": hit(5), "MRR": mrr, "Recall@5": rec5}


def bootstrap(all_rows, methods, n_boot=10000, seed=42):
    """95% довірчі інтервали (bootstrap по запитах) і парні різниці методів."""
    import numpy as np
    rng = np.random.default_rng(seed)
    idx = [i for i, r in enumerate(all_rows[methods[0]]) if r["n_rel"] > 0]
    def vec(m, metric):
        out = []
        for i in idx:
            fr = all_rows[m][i]["first_rank"]
            out.append((1.0 if fr and fr <= 3 else 0.0) if metric == "Hit@3"
                       else (1.0 if fr and fr <= 1 else 0.0) if metric == "Hit@1"
                       else (1.0 / fr if fr else 0.0))
        return np.array(out)
    samples = rng.integers(0, len(idx), size=(n_boot, len(idx)))
    res = {"ci": {}, "diff": {}}
    for metric in ("Hit@1", "Hit@3", "MRR"):
        v = {m: vec(m, metric) for m in methods}
        for m in methods:
            b = v[m][samples].mean(axis=1)
            res["ci"][f"{m}|{metric}"] = (float(v[m].mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))
        for a in range(len(methods)):
            for c in range(a + 1, len(methods)):
                ma, mc = methods[a], methods[c]
                d = (v[mc] - v[ma])[samples].mean(axis=1)
                lo, hi = np.percentile(d, [2.5, 97.5])
                res["diff"][f"{mc} - {ma}|{metric}"] = (float((v[mc] - v[ma]).mean()), float(lo), float(hi))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default=str(ROOT / "eval" / "queries.json"))
    ap.add_argument("--chunks", default=str(ROOT / "data" / "chunks_v2.jsonl"))
    ap.add_argument("--methods", nargs="+", default=["bm25", "embed", "hybrid"])
    ap.add_argument("--model", default="intfloat/multilingual-e5-small")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--title-weight", type=float, default=1.0, help="вага заголовка в BM25 (0 = як до 18.11)")
    ap.add_argument("--set", default="main", choices=["main", "heldout", "all"], help="який набір запитів оцінювати")
    ap.add_argument("--expand", action="store_true", help="розширювати абревіатури в запитах")
    ap.add_argument("--include-unreviewed", action="store_true", help="разом із непереглянутими запитами (для контрольного набору)")
    ap.add_argument("--only-reviewed", action="store_true", help="брати лише запити з reviewed=true")
    ap.add_argument("--out", default=None, help="файл результатів (за замовчуванням eval/results_<tag>.json)")
    ap.add_argument("--tag", default="e5small256", help="мітка конфігурації для імені файлу")
    args = ap.parse_args()
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    queries = json.load(open(args.queries, encoding="utf-8"))
    if args.set != "all":
        queries = [q for q in queries if q.get("set", "main") == args.set]
    if args.only_reviewed and not args.include_unreviewed:
        queries = [q for q in queries if q.get("reviewed")]
    canon = load_alias_map(args.chunks)

    if args.methods == ["bm25"]:
        from search_bm25 import BM25Search

        class _S:  # мінімальний інтерфейс без завантаження моделі
            def __init__(self): self.b = BM25Search(args.chunks, title_weight=args.title_weight)
            def search(self, q, m, k, expand=False):
                from query_expansion import expand_query
                return self.b.search(expand_query(q) if expand else q, k)
        searcher = _S()
    else:
        from search import Searcher
        searcher = Searcher(args.chunks, args.model, args.max_len, args.title_weight)

    all_rows, summary = {}, {}
    for m in args.methods:
        rows = evaluate(searcher, queries, m, canon, args.expand)
        all_rows[m] = rows
        summary[m] = {"all": summarize(rows)}
        bycat = defaultdict(list)
        for r in rows:
            bycat[r["category"]].append(r)
        for cat, rs in bycat.items():
            if any(r["n_rel"] for r in rs):
                summary[m][cat] = summarize(rs)

    cols = ["Hit@1", "Hit@3", "Hit@5", "MRR", "Recall@5"]
    print(f"\nЗапитів із відповіддю: {summary[args.methods[0]]['all']['n']}\n")
    print("Загальні результати")
    print(f"{'метод':<10}" + "".join(f"{c:>10}" for c in cols))
    for m in args.methods:
        print(f"{m:<10}" + "".join(f"{summary[m]['all'][c]:>10.3f}" for c in cols))

    print("\nHit@3 за категоріями")
    cats = [c for c in summary[args.methods[0]] if c != "all"]
    print(f"{'категорія':<16}{'n':>4}" + "".join(f"{m:>10}" for m in args.methods))
    for c in cats:
        print(f"{c:<16}{summary[args.methods[0]][c]['n']:>4}" + "".join(f"{summary[m][c]['Hit@3']:>10.2f}" for m in args.methods))

    print("\nЗапити, де жоден метод не знайшов правильну сторінку в топ-5:")
    for i, q in enumerate(queries):
        if q["relevant"] and all((all_rows[m][i]["first_rank"] or 99) > 5 for m in args.methods):
            print(f"  {q['id']} [{q['category']}] {q['query']}")

    print("\nПастки (топ-1 і оцінка; чим вища оцінка, тим «впевненіше» метод помиляється):")
    for i, q in enumerate(queries):
        if not q["relevant"]:
            print("  " + q["query"] + " | " + " | ".join(f"{m}: {all_rows[m][i]['top1_score']:.2f}" for m in args.methods))

    boot = bootstrap(all_rows, args.methods)
    print("\n95% довірчі інтервали (bootstrap по запитах, 10 000 повторів)")
    for metric in ("Hit@1", "Hit@3", "MRR"):
        for m in args.methods:
            v, lo, hi = boot["ci"][f"{m}|{metric}"]
            print(f"  {metric:<6} {m:<8} {v:.3f}  [{lo:.3f} ; {hi:.3f}]")
    if len(args.methods) > 1:
        print("\nПарні різниці (позначка * = інтервал не містить 0, тобто різниця статистично помітна)")
        for key, (d, lo, hi) in boot["diff"].items():
            star = "*" if (lo > 0 or hi < 0) else " "
            name, metric = key.split("|")
            print(f"  {star} {metric:<6} {name:<18} {d:+.3f}  [{lo:+.3f} ; {hi:+.3f}]")

    out = args.out or str(ROOT / "eval" / f"results_{args.tag}.json")
    json.dump({"config": {"model": args.model, "max_len": args.max_len, "chunks": args.chunks, "title_weight": args.title_weight, "expand": args.expand}, "summary": summary, "bootstrap": boot, "rows": all_rows},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nДеталі збережено: {out}")


if __name__ == "__main__":
    main()