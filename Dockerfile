FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Extract data if present (optional build step, usually mounted)
# RUN if [ -d "data" ]; then for f in data/*.zip; do unzip "$f" -d data; done; fi

CMD ["python", "run_experiments.py"]
