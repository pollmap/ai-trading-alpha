# ATLAS Backend — Multi-stage build optimized for ARM64 (Oracle Cloud A1)
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

# Health check — verify the Python process is alive
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import os, signal; os.kill(1, 0)" || exit 1

CMD ["python", "scripts/run_benchmark.py"]
