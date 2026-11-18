"""
Краулер сайту факультету електроніки та комп'ютерних технологій ЛНУ.

Обходить сторінки в межах одного домену (BFS), поважає robots.txt,
має ліміт кількості сторінок і затримку між запитами.
Результат: JSONL-файл, один рядок = одна сторінка.

Приклад запуску:
    python crawler.py --max-pages 10
    python crawler.py --max-pages 300 --delay 1.0 --out data/pages.jsonl
"""

import argparse
import json
import re
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from urllib import robotparser
from urllib.parse import urljoin, urlparse, urlunparse, parse_qsl, urlencode

import requests
from bs4 import BeautifulSoup

START_URL = "https://electronics.lnu.edu.ua/"
ALLOWED_HOST = "electronics.lnu.edu.ua"
USER_AGENT = "KursovaCrawler/0.1 (student research project; LNU)"

# Копія robots.txt сайту (перевірено вручну 29.10.2026)
FALLBACK_ROBOTS = """User-agent: *
Allow: /
Disallow: /wp-admin/
"""

# розширення файлів, які не є HTML-сторінками
SKIP_EXT = (
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".zip", ".rar", ".7z", ".mp3", ".mp4", ".avi", ".css", ".js", ".xml",
)
# службові частини WordPress, які не потрібні
SKIP_PATH_PARTS = ("/wp-admin", "/wp-json", "/wp-content/", "/feed", "/xmlrpc.php", "/wp-login")
SKIP_PATH_RE = re.compile(r"^/(en/)?(sitemap|news/category)(/|$)")   # карта сайту порожня; категорії дублюють новини
NEWS_PAGE_RE = re.compile(r"^/(?:en/)?news/page/(\d+)/?$")
NEWS_ARTICLE_RE = re.compile(r"^/(?:en/)?news/(?!page/|category/)[^/]+/?$")
MAX_NEWS_PAGINATION = 10   # на сайті 260+ сторінок новин; беремо лише перші 10 сторінок архіву
SKIP_QUERY_KEYS = {"subpage", "semester"}
TRACKING_PARAMS = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "fbclid", "gclid"}


def normalize_url(url: str) -> str | None:
    """Прибирає #якір і трекінгові параметри, приводить до https. None, якщо URL не підходить."""
    p = urlparse(url)
    if p.scheme not in ("http", "https"):
        return None
    if p.netloc.lower() != ALLOWED_HOST:
        return None
    path = p.path or "/"
    # WordPress віддає ту саму сторінку з / і без / в кінці: приводимо до одного вигляду
    if not path.endswith("/") and "." not in path.rsplit("/", 1)[-1]:
        path += "/"
    low = path.lower()
    if low.endswith(SKIP_EXT) or any(part in low for part in SKIP_PATH_PARTS):
        return None
    # службові/порожні сторінки та архіви новин, які не несуть корисного вмісту
    if SKIP_PATH_RE.search(path):
        return None
    m = NEWS_PAGE_RE.search(path)
    if m and int(m.group(1)) > MAX_NEWS_PAGINATION:
        return None
    # єдиний вигляд адреси: /about/staff і /about/staff/ це одна сторінка
    if not path.endswith("/") and "." not in path.rsplit("/", 1)[-1]:
        path += "/"
    pairs = [(k, v) for k, v in parse_qsl(p.query) if k not in TRACKING_PARAMS]
    # динамічні сторінки розкладу викладачів (?subpage=lecturer&id=..&semester=..):
    # переважно порожні («Не поставлено жодної пари»), їх сотні, тому не скануємо
    if any(k in SKIP_QUERY_KEYS for k, _ in pairs):
        return None
    query = urlencode(pairs)
    return urlunparse(("https", ALLOWED_HOST, path, "", query, ""))


def is_news_article(url: str) -> bool:
    """/news/якийсь-slug/ або /en/news/slug/, але не /news/page/N/ і не /news/category/..."""
    path = urlparse(url).path
    return bool(NEWS_ARTICLE_RE.match(path))


def extract(html: str, url: str):
    """Повертає (title, основний текст, список посилань)."""
    soup = BeautifulSoup(html, "lxml")

    links = []
    for a in soup.find_all("a", href=True):
        n = normalize_url(urljoin(url, a["href"]))
        if n:
            links.append(n)

    title = soup.title.get_text(strip=True) if soup.title else ""
    h1 = soup.find("h1")
    # для WordPress-тем заголовок сторінки зазвичай у <title> з суфіксом сайту
    title = re.sub(r"\s*[-–|]\s*Факультет електроніки.*$", "", title).strip() or (
        h1.get_text(strip=True) if h1 else ""
    )

    # прибираємо шум
    for tag in soup(["script", "style", "noscript", "nav", "header", "footer", "aside", "form", "iframe"]):
        tag.decompose()

    main = (
        soup.find("main")
        or soup.find("article")
        or soup.select_one(".entry-content, .post-content, #content, .content")
        or soup.body
        or soup
    )
    text = main.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return title, text, links


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pages", type=int, default=10, help="ліміт сторінок (за замовчуванням 10)")
    ap.add_argument("--delay", type=float, default=1.0, help="пауза між запитами, с")
    ap.add_argument("--out", default="data/pages.jsonl")
    ap.add_argument("--skip-en", action="store_true", help="не скачувати англомовну версію /en/")
    ap.add_argument("--max-news", type=int, default=150, help="максимум сторінок-новин (на сайті їх понад 2000)")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    # robots.txt читаємо через requests (той самий мережевий шлях, що й для сторінок).
    # Якщо файл недоступний, RobotFileParser.read() мовчки забороняє ВСЕ,
    # тому в такому разі беремо правила, які ми вручну перевірили в браузері.
    rp = robotparser.RobotFileParser()
    try:
        rr = session.get(urljoin(START_URL, "/robots.txt"), timeout=15)
        if rr.status_code == 200:
            rp.parse(rr.text.splitlines())
        elif rr.status_code in (401, 403):
            rp.parse(FALLBACK_ROBOTS.splitlines())
            print(f"[warn] robots.txt: HTTP {rr.status_code}, використано збережену копію правил")
        else:  # 404 та інші: за стандартом обмежень немає
            rp.parse(["User-agent: *", "Allow: /"])
    except requests.RequestException as e:
        rp.parse(FALLBACK_ROBOTS.splitlines())
        print(f"[warn] не вдалося прочитати robots.txt ({e}); використано збережену копію правил")

    queue = deque([(normalize_url(START_URL), 0)])
    seen = {normalize_url(START_URL)}
    saved = skipped = errors = news_saved = 0

    with out_path.open("w", encoding="utf-8") as f:
        while queue and saved < args.max_pages:
            url, depth = queue.popleft()

            if args.skip_en and urlparse(url).path.startswith("/en/"):
                continue
            if is_news_article(url):
                if news_saved >= args.max_news:
                    continue
            if not rp.can_fetch(USER_AGENT, url):
                print(f"[robots] заборонено: {url}")
                skipped += 1
                continue

            try:
                r = session.get(url, timeout=15)
            except requests.RequestException as e:
                print(f"[err] {url}: {e}")
                errors += 1
                continue

            ctype = r.headers.get("Content-Type", "")
            if r.status_code != 200 or "text/html" not in ctype:
                print(f"[skip] {r.status_code} {ctype[:30]} {url}")
                skipped += 1
                time.sleep(args.delay)
                continue

            r.encoding = r.apparent_encoding if not r.encoding else r.encoding
            title, text, links = extract(r.text, url)

            record = {
                "url": url,
                "title": title,
                "text": text,
                "lang": "en" if urlparse(url).path.startswith("/en/") else "uk",
                "depth": depth,
                "n_links": len(links),
                "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            saved += 1
            if is_news_article(url):
                news_saved += 1
            print(f"[{saved}/{args.max_pages}] d={depth} {len(text):>6} симв. | {title[:60]} | {url}")

            for link in links:
                if link not in seen:
                    seen.add(link)
                    queue.append((link, depth + 1))

            time.sleep(args.delay)

    print(f"\nГотово: збережено {saved}, пропущено {skipped}, помилок {errors}, у черзі лишилось {len(queue)}")
    print(f"Файл: {out_path}")


if __name__ == "__main__":
    main()