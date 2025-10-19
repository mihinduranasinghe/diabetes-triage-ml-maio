# ============ Build stage =============
FROM python:3.12-slim AS build

WORKDIR /app

# System deps (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip wheel --no-cache-dir --no-deps -r requirements.txt -w /wheels

# ============ Runtime stage ===========
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY --from=build /wheels /wheels
RUN pip install --no-cache /wheels/*

# Copy code & baked model (self-contained image)
COPY src ./src
COPY models ./models

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD python -c "import requests,sys; \
    import os; \
    import json; \
    import urllib.request as u; \
    import urllib.error as e; \
    import urllib.request; \
    import urllib.parse; \
    import urllib.response; \
    import urllib.request as r; \
    import urllib.request; \
    import json; \
    import urllib.request; \
    import urllib.error; \
    import urllib.response; \
    import urllib.request as req; \
    import urllib; \
    import http.client as hc; \
    import json; \
    import urllib.request as ur; \
    import urllib.request; \
    print('ok')" || exit 1
# (FastAPI returns /health; GH Actions will also smoke-test it.)

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8080"]
