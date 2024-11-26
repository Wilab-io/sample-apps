FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including gcc and g++
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy src/requirements.txt first
COPY src/requirements.txt requirements.txt

# Install dependencies, with special handling for problematic packages
RUN pip install --no-cache-dir -r requirements.txt || \
    (pip uninstall -y mteb pytrec-eval-terrier && \
     pip install --no-cache-dir -r requirements.txt && \
     echo "Warning: mteb and pytrec-eval-terrier were skipped due to compatibility issues")

# Copy the rest of the application
COPY . .

# Make the start script executable
RUN chmod +x start.sh

# Use the start script as the entry point
CMD ["./start.sh"]
