FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir -r src/requirements.txt || \
    (pip uninstall -y mteb pytrec-eval-terrier && \
     pip install --no-cache-dir -r requirements.txt && \
     echo "Warning: mteb and pytrec-eval-terrier were skipped due to compatibility issues")

WORKDIR /app/src

CMD ["python", "main.py"]
