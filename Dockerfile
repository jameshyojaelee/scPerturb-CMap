FROM python:3.10-slim

WORKDIR /app

# System deps (optional). Keep minimal for faster builds.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Project files
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY tests ./tests

# Install package with dev/test tools
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -e .[dev]

# Default command prints readiness message
CMD ["python", "-c", "print('scPerturb-CMap container ready')"]
