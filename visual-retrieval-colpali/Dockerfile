FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    poppler-utils \
    tesseract-ocr \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -L -o vespa-cli.tar.gz https://github.com/vespa-engine/vespa/releases/download/v8.453.24/vespa-cli_8.453.24_linux_amd64.tar.gz && \
    tar xzf vespa-cli.tar.gz && \
    mv vespa-cli_8.453.24_linux_amd64/bin/vespa /usr/local/bin && \
    rm -rf vespa-cli_8.453.24_linux_amd64 vespa-cli.tar.gz

COPY . .

RUN pip install --no-cache-dir -r src/requirements.txt || \
    (pip uninstall -y mteb pytrec-eval-terrier && \
     pip install --no-cache-dir -r requirements.txt && \
     echo "Warning: mteb and pytrec-eval-terrier were skipped due to compatibility issues")

WORKDIR /app/src

ENV PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

CMD ["python", "main.py"]
