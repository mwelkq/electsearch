"""
Обробка зібраних сторінок -> фрагменти (chunks) для індексації.

  python process.py --pages data/pages.jsonl --out data/chunks.jsonl

Кроки:
 1. відсів сміття (динамічні розклади ?subpage=...);
 2. відсів дублікатів за вмістом (залишається коротша/«канонічна» адреса, решта в aliases);
 3. опис для сторінок-розділів із малим текстом (за назвами підсторінок);
 4. «хлібні крихти» (шлях розділу: «Про факультет › Лабораторії») — поле context,
    яке пошук додає до тексту фрагмента замість голого заголовка (v2, 18.11.2026);
 5. розбиття на фрагменти по абзацах (~CHUNK_CHARS символів, з перекриттям);
 6. запис у JSONL (один рядок = один фрагмент).
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

CHUNK_CHARS = 1000      # цільовий розмір фрагмента, символів
OVERLAP_CHARS = 150     # перекриття між сусідніми фрагментами
HUB_MIN_CHARS = 300     # сторінки коротші за це вважаємо «хабами»
SITE_SUFFIX = re.compile(r"\s*[-–|]\s*Faculty of Electronics.*$")


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


LISTING_RE = re.compile(r"/page/\d+/?$")


def is_junk(r):
    """Службові сторінки: динамічні розклади та списки-пагінація (/news/page/2/ тощо)."""
    return ("subpage=" in r["url"] or "semester=" in r["url"]
            or bool(LISTING_RE.search(r["url"])))


def dedup(rows):
    """Групує сторінки з однаковим текстом. Канонічна = найкоротший шлях."""
    groups = defaultdict(list)
    for r in rows:
        groups[r["text"].strip()].append(r)
    kept, n_removed = [], 0
    for text, rs in groups.items():
        if len(rs) == 1 or len(text) < 50:   # дуже короткі збіги (порожні хаби) не зливаємо
            kept.extend(rs)
            continue
        rs.sort(key=lambda x: (len(urlparse(x["url"]).path), x["url"]))
        main = dict(rs[0])
        main["aliases"] = [x["url"] for x in rs[1:]]
        kept.append(main)
        n_removed += len(rs) - 1
    return kept, n_removed


def add_hub_descriptions(rows):
    """Для сторінок із малим текстом додає список заголовків їхніх підсторінок."""
    by_prefix = defaultdict(list)
    for r in rows:
        path = urlparse(r["url"]).path
        parts = [p for p in path.strip("/").split("/") if p]
        for i in range(1, len(parts)):
            by_prefix["/" + "/".join(parts[:i]) + "/"].append(r)
    n = 0
    for r in rows:
        if len(r["text"]) >= HUB_MIN_CHARS:
            continue
        path = urlparse(r["url"]).path
        kids = [k for k in by_prefix.get(path, []) if k["url"] != r["url"]]
        # лише прямі діти (на 1 рівень глибше), щоб не тягнути сотні профілів
        depth = len([p for p in path.strip("/").split("/") if p])
        kids = [k for k in kids if len([p for p in urlparse(k["url"]).path.strip("/").split("/") if p]) == depth + 1]
        titles = []
        for k in kids[:25]:
            t = SITE_SUFFIX.sub("", k["title"]).strip()
            if t and t not in titles:
                titles.append(t)
        if titles:
            r["hub_text"] = "Розділ містить: " + "; ".join(titles) + "."
            n += 1
    return n


# Назви розділів за першим сегментом адреси. Потрібні, бо для /course/, /employee/ тощо
# окремої сторінки-розділу немає, а заголовок /students/ на сайті — «Студентське самоврядування».
SECTION_NAMES = {
    "about": "Про факультет", "academics": "Навчання", "students": "Студентам",
    "admission": "Вступ", "research": "Наука", "news": "Новини",
    "course": "Дисципліни", "employee": "Працівники", "department": "Кафедри",
    "laboratory": "Лабораторії", "seminar": "Семінари", "seminars": "Семінари",
    "memory": "Пам'ять", "bachelor": "Бакалаврат", "master": "Магістратура",
}


def add_breadcrumbs(rows):
    """context = «Розділ › Підрозділ › Заголовок» за ієрархією адрес."""
    title_by_path = {urlparse(r["url"]).path: SITE_SUFFIX.sub("", r["title"]).strip() for r in rows}
    for r in rows:
        parts = [p for p in urlparse(r["url"]).path.strip("/").split("/") if p]
        crumbs = []
        for i in range(1, len(parts)):
            seg = parts[i - 1]
            name = SECTION_NAMES.get(seg) or title_by_path.get("/" + "/".join(parts[:i]) + "/")
            if name and name not in crumbs:
                crumbs.append(name)
        title = SITE_SUFFIX.sub("", r["title"]).strip()
        r["context"] = " › ".join(crumbs + [title]) if crumbs else title


def clean_text(t):
    t = t.replace("\xa0", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n\s*:\s*\n", ": ", t)        # «Посада\n:\nметодист» -> «Посада: методист»
    t = re.sub(r"\n{2,}", "\n", t)
    return t.strip()


def split_text(text):
    """Розбиття по рядках/абзацах на фрагменти ~CHUNK_CHARS з перекриттям."""
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        while len(p) > CHUNK_CHARS * 1.5:       # дуже довгий абзац ріжемо по реченнях/символах
            cut = p.rfind(". ", 0, CHUNK_CHARS)
            cut = cut + 1 if cut > 200 else CHUNK_CHARS
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(p[:cut].strip())
            p = p[cut:].strip()
        if cur and len(cur) + len(p) + 1 > CHUNK_CHARS:
            chunks.append(cur)
            cur = cur[-OVERLAP_CHARS:].split(" ", 1)[-1] + "\n" + p if OVERLAP_CHARS else p
        else:
            cur = (cur + "\n" + p) if cur else p
    if cur:
        chunks.append(cur)
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", default="data/pages.jsonl")
    ap.add_argument("--out", default="data/chunks.jsonl")
    ap.add_argument("--no-breadcrumbs", action="store_true", help="без шляху розділу (для порівняння)")
    ap.add_argument("--lang", default="uk", choices=["uk", "en", "all"], help="які сторінки обробляти")
    args = ap.parse_args()

    rows = load(args.pages)
    print(f"зчитано сторінок: {len(rows)}")
    rows = [r for r in rows if not is_junk(r)]
    print(f"після відсіву розкладів: {len(rows)}")
    if args.lang != "all":
        rows = [r for r in rows if r["lang"] == args.lang]
        print(f"мова {args.lang}: {len(rows)}")
    rows, removed = dedup(rows)
    print(f"прибрано дублікатів за вмістом: {removed}, лишилось: {len(rows)}")
    for r in rows:
        r["text"] = clean_text(r["text"])
    n_hubs = add_hub_descriptions(rows)
    print(f"хабів з описом із підсторінок: {n_hubs}")
    if not args.no_breadcrumbs:
        add_breadcrumbs(rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_chunks = 0
    with out.open("w", encoding="utf-8") as f:
        for page_id, r in enumerate(rows):
            title = SITE_SUFFIX.sub("", r["title"]).strip()
            body = r["text"]
            if r.get("hub_text"):
                body = (body + "\n" + r["hub_text"]).strip()
            for i, ch in enumerate(split_text(body) or [title]):
                f.write(json.dumps({
                    "chunk_id": f"{page_id}-{i}",
                    "page_id": page_id,
                    "url": r["url"],
                    "title": title,
                    "context": r.get("context", title),
                    "lang": r["lang"],
                    "text": ch,
                    "aliases": r.get("aliases", []),
                }, ensure_ascii=False) + "\n")
                n_chunks += 1
    print(f"записано фрагментів: {n_chunks} -> {out}")


if __name__ == "__main__":
    main()