FROM python:3.10-slim AS builder

WORKDIR /app

# System deps for building wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN pip install --no-cache-dir --upgrade pip build \
 && python -m build

FROM python:3.10-slim AS runtime
WORKDIR /app

COPY --from=builder /app/dist /dist
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir /dist/*.whl

# Default entrypoint to the CLI
ENTRYPOINT ["scperturb-cmap"]
CMD ["diagnose"]
 