"""
Бекенд вебпомічника: FastAPI поверх пошуку (BM25 / embeddings / гібрид) і RAG.

  pip install fastapi uvicorn
  uvicorn api:app --port 8000          # потім відкрити http://127.0.0.1:8000

Кінцеві точки:
  GET  /health                         стан сервісу
  GET  /search?q=...&method=hybrid&k=5 лише пошук (без LLM)
  POST /ask {"question": "...", "method": "hybrid"}   пошук + LLM, повертає рекомендовану сторінку
  GET  /                               чат-інтерфейс (web/index.html)
"""
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import rag

ROOT = Path(__file__).resolve().parent
CHUNKS = os.environ.get("CHUNKS_PATH", str(ROOT / "data" / "chunks_v2.jsonl"))
EMBED_MODEL = os.environ.get("EMBED_MODEL", "intfloat/multilingual-e5-small")
MAX_LEN = int(os.environ.get("EMBED_MAX_LEN", "256"))
# для вебу головна — швидка стабільна модель; повільніша більша слугує запасною
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
FALLBACKS = tuple(m for m in ("gemini-3.5-flash",) if m != MODEL)
MAX_QUESTION = 300

app = FastAPI(title="Вебпомічник навігації сайтом факультету")
state = {"searcher": None, "key": None}


def index_info():
    """Скільки сторінок в індексі і коли їх скановано (для плашки в інтерфейсі)."""
    if "info" not in state:
        pages = {c["page_id"] for c in get_searcher().bm25.chunks}
        date = None
        pj = Path(CHUNKS).parent / "pages.jsonl"
        if pj.exists():
            latest = ""
            with pj.open(encoding="utf-8") as f:
                for line in f:
                    if '"fetched_at"' in line:
                        latest = max(latest, json.loads(line).get("fetched_at", ""))
            date = latest[:10] or None
        if not date:
            date = datetime.fromtimestamp(Path(CHUNKS).stat().st_mtime, timezone.utc).strftime("%Y-%m-%d")
        y, m, d = date.split("-")
        state["info"] = {"pages": len(pages), "updated": f"{d}.{m}.{y}"}
    return state["info"]


def get_searcher():
    """Індекс вантажимо один раз, при першому зверненні (або при старті)."""
    if state["searcher"] is None:
        from search import Searcher
        state["searcher"] = Searcher(CHUNKS, EMBED_MODEL, MAX_LEN)
    return state["searcher"]


@app.on_event("startup")
def startup():
    state["key"] = rag.load_key()
    if os.environ.get("LAZY_INDEX") != "1":
        get_searcher()


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=MAX_QUESTION)
    method: str = "hybrid"
    k: int = Field(8, ge=1, le=10)  # 8: експ. 8 (19.11), точність RAG 0.942 -> 0.981


def _check_method(method):
    if method not in ("bm25", "embed", "hybrid"):
        raise HTTPException(400, "method має бути bm25, embed або hybrid")


def _page(p):
    return {"title": p["title"], "url": p["url"], "score": round(float(p["score"]), 4),
            "snippet": p["snippet"]}


@app.get("/health")
def health():
    info = index_info() if state["searcher"] is not None else {}
    return {"status": "ok", "index_loaded": state["searcher"] is not None,
            "llm_key": bool(state["key"]), "model": MODEL, **info}


class Feedback(BaseModel):
    question: str = Field(..., max_length=MAX_QUESTION)
    method: str = "hybrid"
    recommended: str | None = Field(None, max_length=500)
    useful: bool


@app.post("/feedback")
def feedback(fb: Feedback):
    """Відгук користувача «Корисно? Так/Ні» — дописуємо в data/feedback.jsonl (для оцінки на реальних запитах)."""
    rec = fb.model_dump() | {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    with (ROOT / "data" / "feedback.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"ok": True}


@app.get("/search")
def search(q: str = Query(..., min_length=2, max_length=MAX_QUESTION),
           method: str = "hybrid", k: int = Query(5, ge=1, le=10)):
    _check_method(method)
    t0 = time.time()
    pages = get_searcher().search(q, method, k, expand=True)
    return {"query": q, "method": method, "results": [_page(p) for p in pages],
            "latency": round(time.time() - t0, 3)}


class _Cached:
    """Щоб не шукати двічі: LLM-крок отримує вже знайдені сторінки."""
    def __init__(self, pages):
        self.pages = pages

    def search(self, *a, **kw):
        return self.pages


@app.post("/ask")
def ask(req: AskRequest):
    _check_method(req.method)
    if not state["key"]:
        raise HTTPException(503, "Не задано GEMINI_API_KEY (файл .env)")
    t0 = time.time()
    pages = get_searcher().search(req.question, req.method, req.k, expand=True)
    titles = {p["url"]: p["title"] for p in pages}
    try:
        res = rag.answer_question(req.question, _Cached(pages), state["key"], req.method, req.k,
                                  MODEL, True, fallbacks=FALLBACKS)
    except Exception:
        # LLM недоступна: віддаємо результати пошуку, щоб помічник лишався корисним
        return {"found": None, "answer": "Мовна модель зараз недоступна. Ось що знайшов пошук за вашим запитом.",
                "recommended": None, "alternatives": [], "results": [_page(p) for p in pages],
                "degraded": True, "latency": round(time.time() - t0, 2)}
    rec = res["recommended_url"]
    return {
        "found": res["found"],
        "answer": res["answer"],
        "recommended": {"url": rec, "title": titles.get(rec, rec)} if rec else None,
        "alternatives": [{"url": u, "title": titles.get(u, u)} for u in res["alternatives"]],
        "results": [_page(p) for p in pages],
        "model_used": res["model_used"],
        "degraded": False,
        "latency": round(time.time() - t0, 2),
    }


@app.get("/logo.png")
def logo():
    return FileResponse(ROOT / "web" / "logo.png")


@app.get("/logo-t.png")
def logo_transparent():
    return FileResponse(ROOT / "web" / "logo-t.png")


@app.get("/")
def index():
    return FileResponse(ROOT / "web" / "index.html")
