# ATLAS Backend — Multi-stage build optimized for ARM64 (Oracle Cloud A1)
# Used for both atlas-api and atlas-worker services
FROM python:3.11-slim AS builder

WORKDIR /build
COPY pyproject.toml ./
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e . && \
    pip install --no-cache-dir psycopg2-binary || true

# ── Runtime Stage ─────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/
COPY pyproject.toml ./

# Create non-root user
RUN groupadd -r atlas && useradd -r -g atlas atlas && \
    mkdir -p /app/data /app/reports && \
    chown -R atlas:atlas /app

USER atlas

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default CMD — overridden by docker-compose per service
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
