# =============================================================================
# Stage 1: Builder — install Python deps, pre-download models & Playwright
# =============================================================================
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_NO_DEV=1

WORKDIR /app

# Layer caching: install deps before copying source
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project

# Copy source and data
COPY src/ src/
COPY data/ data/

# Install project + all deps into .venv
RUN uv sync --locked

# Pre-download HuggingFace cross-encoder ONNX model
ENV HF_HOME=/app/.cache/huggingface
RUN uv run python -c "from src.rerank import _snapshot_dir; _snapshot_dir()"

# Install Playwright and Chromium browser
RUN uv pip install playwright && uv run python -m playwright install chromium

# =============================================================================
# Stage 2: Runtime — minimal slim image with system deps
# =============================================================================
FROM python:3.12-slim-bookworm AS runtime

# System dependencies: Tesseract (ara+eng+fra), Playwright browser libs, tini
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-ara \
    tesseract-ocr-fra \
    libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 \
    libdbus-1-3 libxkbcommon0 libatspi2.0-0 libxcomposite1 libxdamage1 \
    libxfixes3 libxrandr2 libgbm1 libpango-1.0-0 libcairo2 libasound2 \
    libwayland-client0 \
    fonts-noto \
    tini \
    && rm -rf /var/lib/apt/lists/*

# Environment variables
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata
ENV HF_HOME=/app/.cache/huggingface
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PLAYWRIGHT_BROWSERS_PATH=/app/.cache/ms-playwright

# Create non-root user
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

# Create writable store and cache directories BEFORE copying files
RUN mkdir -p /app/store /app/.cache/huggingface /app/.cache/ms-playwright && \
    chown -R appuser:appgroup /app

# Copy from builder stage
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/data /app/data
COPY --from=builder /app/.cache/huggingface /app/.cache/huggingface
COPY --from=builder /root/.cache/ms-playwright /app/.cache/ms-playwright

# Make the venv binaries available
ENV PATH=/app/.venv/bin:$PATH

# Switch to non-root user
USER appuser
WORKDIR /home/appuser/app

ENTRYPOINT ["tini", "--"]
CMD ["python", "-m", "src.main"]
