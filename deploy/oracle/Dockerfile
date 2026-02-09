# ATLAS Backend — Multi-stage build optimized for ARM64 (Oracle Cloud A1)
FROM python:3.11-slim AS builder

WORKDIR /build
COPY pyproject.toml ./
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    pydantic structlog numpy pandas \
    asyncio-mqtt aiohttp \
    openpyxl python-docx reportlab \
    psycopg2-binary redis \
    pyyaml yfinance && \
    pip install --no-cache-dir uuid7 || true

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
COPY api/ ./api/
COPY pyproject.toml ./

# Create non-root user
RUN groupadd -r atlas && useradd -r -g atlas atlas && \
    mkdir -p /app/data /app/reports && \
    chown -R atlas:atlas /app

USER atlas

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["python", "-m", "scripts.run_benchmark"]
