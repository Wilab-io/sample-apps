FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sLO https://github.com/tailwindlabs/tailwindcss/releases/latest/download/tailwindcss-linux-x64 \
    && chmod +x tailwindcss-linux-x64 \
    && mv tailwindcss-linux-x64 /usr/local/bin/tailwindcss

COPY src/requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt || \
    (pip uninstall -y mteb pytrec-eval-terrier && \
     pip install --no-cache-dir -r requirements.txt && \
     echo "Warning: mteb and pytrec-eval-terrier were skipped due to compatibility issues")

ENV SHAD4FAST_TAILWINDCSS_PATH=/usr/local/bin/tailwindcss
ENV PATH="/usr/local/bin:${PATH}"

COPY . .

WORKDIR /app/src
RUN rm -f ./tailwindcss && \
    ln -s /usr/local/bin/tailwindcss ./tailwindcss && \
    shad4fast build && \
    rm -f ./tailwindcss

WORKDIR /app
CMD ["python", "src/main.py"]
