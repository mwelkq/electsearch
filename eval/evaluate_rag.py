"""
Оцінка повного помічника (пошук + мовна модель) на тестовому наборі.

  python eval/evaluate_rag.py --only-reviewed --method hybrid --tag rag_hybrid
  python eval/evaluate_rag.py --only-reviewed --method embed --model gemini-3.1-flash-lite --tag rag_embed_lite
  python eval/evaluate_rag.py --set heldout --include-unreviewed --tag rag_heldout

Метрики:
  Точність            — частка запитів із відповіддю, де рекомендована сторінка правильна
  Пошук знайшов (@5)  — правильна сторінка була серед п'яти наданих моделі (межа можливого для моделі)
  Точність при знайденому — з тих, де пошук знайшов, у скількох модель вибрала правильну (якість самої LLM)
  Хибна відмова       — відповідь на сайті є, а модель сказала «не знайшла»
  Хибна сторінка      — модель впевнено порекомендувала неправильну сторінку
  Вигаданий URL       — модель вказала URL, якого не було серед знайдених (відкинуто перевіркою)
  Пастки: правильна відмова — для запитів без відповіді модель сказала «не знайшла»
  Затримка            — час відповіді (с), медіана і 90-й перцентиль, лише LLM-крок разом із пошуком

Результати дописуються в eval/rag_<tag>.jsonl: після перерви (ліміт запитів, Ctrl+C) запуск продовжить з місця зупинки.
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from evaluate import load_alias_map  # noqa: E402


def run_one(q, searcher, key, canon, method, model, expand, k, fallbacks):
    import rag
    rel = {canon.get(u, u) for u in q["relevant"]}
    t0 = time.time()
    rec = {"id": q["id"], "category": q["category"], "query": q["query"], "n_rel": len(rel)}
    try:
        res = rag.answer_question(q["query"], searcher, key, method, k, model, expand, fallbacks=fallbacks)
    except Exception as e:  # помилка сервісу: фіксуємо й рухаємось далі
        rec.update({"error": f"{type(e).__name__}: {str(e)[:200]}", "latency": round(time.time() - t0, 2)})
        return rec
    got = [canon.get(u, u) for u in res["retrieved"]]
    r = canon.get(res["recommended_url"], res["recommended_url"]) if res["recommended_url"] else None
    rec.update({
        "latency": round(time.time() - t0, 2),
        "found": res["found"], "recommended_url": res["recommended_url"], "answer": res["answer"],
        "alternatives": res["alternatives"], "hallucinated_url": res["hallucinated_url"],
        "parsed": res["parsed"], "model_used": res["model_used"], "retrieved": res["retrieved"],
        "retrieval_hit": bool(set(got) & rel),
        "correct": bool(res["found"] and r in rel),
        "wrong_page": bool(res["found"] and r not in rel and len(rel) > 0),
        "in_alternatives": bool({canon.get(u, u) for u in res["alternatives"]} & rel),
    })
    return rec


def summarize(rows, model=None):
    ok = [r for r in rows if "error" not in r]
    ans = [r for r in ok if r["n_rel"] > 0]
    trap = [r for r in ok if r["n_rel"] == 0]
    frac = lambda a, b: (a / b) if b else float("nan")
    s = {"n_total": len(rows), "n_errors": len(rows) - len(ok), "n_answerable": len(ans), "n_trap": len(trap)}
    s["accuracy"] = frac(sum(r["correct"] for r in ans), len(ans))
    s["retrieval_hit@5"] = frac(sum(r["retrieval_hit"] for r in ans), len(ans))
    hit = [r for r in ans if r["retrieval_hit"]]
    s["accuracy_given_hit"] = frac(sum(r["correct"] for r in hit), len(hit))
    s["false_refusal"] = frac(sum(not r["found"] for r in ans), len(ans))
    s["wrong_page"] = frac(sum(r["wrong_page"] for r in ans), len(ans))
    s["hallucinated_url"] = frac(sum(r["hallucinated_url"] for r in ok), len(ok))
    s["trap_correct_refusal"] = frac(sum(not r["found"] for r in trap), len(trap))
    lat = [r["latency"] for r in ok]
    s["latency_median"] = float(np.median(lat)) if lat else float("nan")
    s["latency_p90"] = float(np.percentile(lat, 90)) if lat else float("nan")
    s["fallback_used"] = sum(1 for r in ok if model and r.get("model_used") != model)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default=str(ROOT / "eval" / "queries.json"))
    ap.add_argument("--chunks", default=str(ROOT / "data" / "chunks_v2.jsonl"))
    ap.add_argument("--method", default="hybrid", choices=["bm25", "embed", "hybrid"])
    ap.add_argument("--model", default=None, help="модель Gemini (за замовчуванням з rag.py)")
    ap.add_argument("--embed-model", default="intfloat/multilingual-e5-small")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--title-weight", type=float, default=1.0, help="вага заголовка в BM25 (0 = як до 18.11)")
    ap.add_argument("-k", type=int, default=8)  # як у rag.py і /ask (з 19.11)
    ap.add_argument("--no-expand", action="store_true")
    ap.add_argument("--fallback", action="store_true", help="дозволити запасні моделі (за замовчуванням вимкнено)")
    ap.add_argument("--set", default="main", choices=["main", "heldout", "user", "all"])
    ap.add_argument("--only-reviewed", action="store_true")
    ap.add_argument("--include-unreviewed", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="лише перші N запитів (для пробного запуску)")
    ap.add_argument("--sleep", type=float, default=7.0, help="пауза між запитами, с (ліміти безкоштовного тарифу)")
    ap.add_argument("--tag", default="rag")
    args = ap.parse_args()
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    import rag
    from search import Searcher
    key = rag.load_key()
    if not key:
        sys.exit("Немає ключа GEMINI_API_KEY (див. rag.py)")
    model = args.model or rag.DEFAULT_MODEL

    queries = json.load(open(args.queries, encoding="utf-8"))
    if args.set != "all":
        queries = [q for q in queries if q.get("set", "main") == args.set]
    if args.only_reviewed and not args.include_unreviewed:
        queries = [q for q in queries if q.get("reviewed")]
    if args.limit:
        queries = queries[:args.limit]

    canon = load_alias_map(args.chunks)
    searcher = Searcher(args.chunks, args.embed_model, args.max_len, args.title_weight)
    out_path = ROOT / "eval" / f"rag_{args.tag}.jsonl"
    done = {}
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if "error" not in r:          # запити з помилкою перераховуємо
                    done[r["id"]] = r
    print(f"Модель: {model} | пошук: {args.method} | запитів: {len(queries)} | вже готово: {len(done)}")

    fallbacks = rag.FALLBACK_MODELS if args.fallback else ()
    rows = []
    with out_path.open("w", encoding="utf-8") as f:
        for r in done.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for i, q in enumerate(queries, 1):
            if q["id"] in done:
                rows.append(done[q["id"]])
                continue
            r = run_one(q, searcher, key, canon, args.method, model, not args.no_expand, args.k, fallbacks)
            rows.append(r)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            status = "ПОМИЛКА " + r["error"][:60] if "error" in r else ("✓" if r.get("correct") else ("відмова" if not r["found"] else "✗"))
            print(f"[{i}/{len(queries)}] {q['id']} {status}  {r['latency']:.1f}с  {q['query'][:50]}")
            time.sleep(args.sleep)

    s = summarize(rows, model)
    p = lambda v: "—" if v != v else f"{v:.3f}"
    print(f"\nЗапитів: {s['n_total']} (із відповіддю {s['n_answerable']}, пасток {s['n_trap']}), помилок сервісу: {s['n_errors']}")
    print(f"Точність (правильна сторінка):          {p(s['accuracy'])}")
    print(f"Пошук знайшов правильну в топ-{args.k}:       {p(s['retrieval_hit@5'])}")
    print(f"Точність, коли пошук знайшов:           {p(s['accuracy_given_hit'])}")
    print(f"Хибна відмова:                          {p(s['false_refusal'])}")
    print(f"Хибна сторінка:                         {p(s['wrong_page'])}")
    print(f"Вигаданий URL (частка відповідей):      {p(s['hallucinated_url'])}")
    print(f"Пастки: правильна відмова:              {p(s['trap_correct_refusal'])}")
    if s["fallback_used"]:
        print(f"Відповідала запасна модель: {s['fallback_used']} разів")
    print(f"Затримка, с: медіана {s['latency_median']:.1f}, 90-й перцентиль {s['latency_p90']:.1f}")

    by = defaultdict(list)
    for r in rows:
        if "error" not in r and r["n_rel"] > 0:
            by[r["category"]].append(r)
    print("\nТочність за категоріями")
    for c, rs in by.items():
        print(f"  {c:<16}{len(rs):>3}   {sum(r['correct'] for r in rs) / len(rs):.2f}")

    print("\nНеправильні відповіді:")
    for r in rows:
        if "error" in r:
            continue
        if r["n_rel"] > 0 and not r["correct"]:
            why = "пошук не знайшов" if not r["retrieval_hit"] else ("відмова" if not r["found"] else "обрано іншу сторінку")
            print(f"  {r['id']} [{r['category']}] {r['query']}  ← {why}")
    print("\nПастки:")
    for r in rows:
        if "error" not in r and r["n_rel"] == 0:
            print(f"  {'відмовилась ✓' if not r['found'] else 'ВІДПОВІЛА ✗ ' + str(r['recommended_url'])}  | {r['query']}")

    (ROOT / "eval" / f"rag_{args.tag}_summary.json").write_text(
        json.dumps({"config": vars(args) | {"model": model}, "summary": s}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nДеталі: {out_path}")


if __name__ == "__main__":
    main()
