"""
Бекенд вебпомічника: FastAPI поверх пошуку (BM25 / embeddings / гібрид) і RAG.

  pip install fastapi uvicorn
  uvicorn api:app --port 8000          # потім відкрити http://127.0.0.1:8000

Кінцеві точки:
  GET  /health                         стан сервісу
  GET  /search?q=...&method=hybrid&k=5 лише пошук (без LLM)
  POST /ask {"question": "...", "method": "hybrid"}   пошук + LLM, повертає рекомендовану сторінку
  GET  /                               чат-інтерфейс (web/index.html)

Налаштування через змінні середовища (див. README.md):
  GEMINI_API_KEY, GEMINI_MODEL, CHUNKS_PATH, EMBED_MODEL, EMBED_MAX_LEN, LAZY_INDEX
  ASK_LIMIT_PER_MIN  (10)  — скільки /ask на хвилину з однієї адреси (0 = без ліміту)
  ALLOWED_ORIGINS    ("")  — CORS: домени через кому, якщо фронтенд на іншому домені
  TRUST_PROXY        ("0") — 1, якщо сервіс стоїть за nginx/проксі (брати IP з X-Forwarded-For)
  LOG_QUERIES        ("1") — писати анонімний журнал запитів queries_log.jsonl (без IP)
  RUNTIME_DIR     (data/)  — тека для feedback.jsonl і queries_log.jsonl (у Docker — том)
"""
import json
import os
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
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
ASK_LIMIT = int(os.environ.get("ASK_LIMIT_PER_MIN", "10"))
TRUST_PROXY = os.environ.get("TRUST_PROXY") == "1"
LOG_QUERIES = os.environ.get("LOG_QUERIES", "1") == "1"
RUNTIME = Path(os.environ.get("RUNTIME_DIR", str(ROOT / "data")))  # куди писати відгуки й журнал
ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]

app = FastAPI(title="Вебпомічник навігації сайтом факультету")
if ORIGINS:  # інтерфейс віддається цим же сервером, тож CORS потрібен лише для іншого домену
    app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["GET", "POST"],
                       allow_headers=["Content-Type"])
state = {"searcher": None, "key": None}
_lock = threading.Lock()
_hits = defaultdict(deque)  # IP -> час останніх запитів /ask (ковзне вікно 60 с)


def _client_ip(request):
    if TRUST_PROXY and request.headers.get("x-forwarded-for"):
        return request.headers["x-forwarded-for"].split(",")[0].strip()
    return request.client.host if request.client else "?"


def rate_limit(request):
    """Простий ліміт у пам'яті: безкоштовна квота Gemini спільна для всіх користувачів,
    тож один користувач (чи бот) не повинен її вичерпати."""
    if ASK_LIMIT <= 0:
        return
    now, ip = time.time(), _client_ip(request)
    with _lock:
        q = _hits[ip]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= ASK_LIMIT:
            raise HTTPException(429, f"Забагато запитів: не більше {ASK_LIMIT} на хвилину. Спробуйте трохи згодом.")
        q.append(now)
        if len(_hits) > 10000:  # не даємо словнику рости безмежно
            for k in [k for k, v in _hits.items() if not v or now - v[-1] > 60]:
                del _hits[k]


def log_query(rec):
    """Анонімний журнал (без IP): реальні запити користувачів — матеріал для оцінки системи."""
    if not LOG_QUERIES:
        return
    rec = rec | {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    with _lock, (RUNTIME / "queries_log.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


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
    with (RUNTIME / "feedback.jsonl").open("a", encoding="utf-8") as f:
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
def ask(req: AskRequest, request: Request):
    _check_method(req.method)
    rate_limit(request)
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
        log_query({"question": req.question, "method": req.method, "k": req.k, "degraded": True,
                   "retrieved": [p["url"] for p in pages], "latency": round(time.time() - t0, 2)})
        return {"found": None, "answer": "Мовна модель зараз недоступна. Ось що знайшов пошук за вашим запитом.",
                "recommended": None, "alternatives": [], "results": [_page(p) for p in pages],
                "degraded": True, "latency": round(time.time() - t0, 2)}
    rec = res["recommended_url"]
    log_query({"question": req.question, "method": req.method, "k": req.k, "degraded": False,
               "found": res["found"], "recommended": rec, "model": res["model_used"],
               "retrieved": [p["url"] for p in pages], "latency": round(time.time() - t0, 2)})
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
