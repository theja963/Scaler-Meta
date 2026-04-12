# Multi-stage build using openenv-base

ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

ARG BUILD_MODE=in-repo
ARG ENV_NAME=scaler_meta

# Copy everything (your repo root)
COPY . /app/env

WORKDIR /app/env

# Ensure uv exists
RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-install-project --no-editable; \
    else \
        uv sync --no-install-project --no-editable; \
    fi

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-editable; \
    else \
        uv sync --no-editable; \
    fi

# ---------- FINAL STAGE ----------
FROM ${BASE_IMAGE}

WORKDIR /app

# Copy virtual env
COPY --from=builder /app/env/.venv /app/.venv

# Copy project
COPY --from=builder /app/env /app/env

ENV PATH="/app/.venv/bin:$PATH"

# 🔥 IMPORTANT: root now holds models + env
ENV PYTHONPATH="/app/env:$PYTHONPATH"

# Health check (unchanged)
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 🔥 UPDATED ENTRYPOINT
# Now points to app_env instead of server.app
CMD ["sh", "-c", "cd /app/env && uvicorn Modified.app_env:app --host 0.0.0.0 --port 8000"]