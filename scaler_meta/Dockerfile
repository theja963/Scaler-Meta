ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE}

WORKDIR /app

# Install git (needed for deps)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

# Install uv if not present
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    mv /root/.local/bin/uvx /usr/local/bin/uvx

# Copy project
COPY . /app

# Set working dir to your env root
WORKDIR /app

# Install dependencies using uv

ENV UV_LINK_MODE=copy

RUN pip install git+https://github.com/meta-pytorch/openenv.git

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync
# RUN python -c "import openenv; print('openenv OK')"


# Ensure Python can find your modules
ENV PYTHONPATH="/app:$PYTHONPATH"

# HF Spaces requirement (if used)
EXPOSE 7860

# Run app
CMD ["uvicorn", "Modified.app_env:app", "--host", "0.0.0.0", "--port", "7860"]