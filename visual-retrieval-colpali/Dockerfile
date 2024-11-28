FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY src/requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt || \
    (pip uninstall -y mteb pytrec-eval-terrier && \
     pip install --no-cache-dir -r requirements.txt && \
     echo "Warning: mteb and pytrec-eval-terrier were skipped due to compatibility issues")

COPY . .

CMD ["python", "src/main.py"]
