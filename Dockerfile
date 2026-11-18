# Вебпомічник навігації сайтом факультету: образ для запуску на сервері.
#   docker build -t electsearch .
#   docker run -p 8000:8000 -e GEMINI_API_KEY=... -v electsearch-data:/app/data/runtime electsearch
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 HF_HOME=/app/.hf

# torch лише для CPU (без CUDA): образ у кілька разів менший, GPU для e5-small не потрібен
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY *.py ./
COPY web/ web/
COPY data/pages.jsonl data/chunks_v2.jsonl data/

# модель e5-small і вектори індексу рахуємо під час збирання (~10 хв на CPU),
# щоб сервер стартував за секунди
RUN python -c "from search import Searcher; Searcher()"

RUN useradd -m app && mkdir -p data/runtime && chown -R app /app/data
USER app
ENV RUNTIME_DIR=/app/data/runtime
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
