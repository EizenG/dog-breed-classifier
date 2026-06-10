# Stage 1 - export des dépendances via Poetry
FROM python:3.11-slim AS builder

WORKDIR /build

RUN pip install --no-cache-dir poetry==2.4.1 poetry-plugin-export

COPY pyproject.toml poetry.lock* LICENSE* ./

RUN poetry export -f requirements.txt --output requirements.txt --without-hashes --only main


# Stage 2 - image finale
FROM python:3.11-slim

WORKDIR /app

# Dépendances système pour Pillow et TensorFlow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/           api/
COPY dog_breed_classifier/ dog_breed_classifier/
COPY params.yaml    .

# Répertoires montés en volume au runtime (models/ et reports/)
RUN mkdir -p models reports/drift

ENV PORT=7860
ENV PYTHONUNBUFFERED=1

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')"

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port $PORT"]
