# CPU-first image; GPU users should extend FROM nvidia/cuda:12.4.1-runtime-ubuntu24.04
# and install torch with the matching CUDA wheel.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# uv for fast installs
RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

# Resolve + install; --extra dev is stripped for production image
RUN uv sync --no-dev

ENV PATH="/app/.venv/bin:${PATH}"

ENTRYPOINT ["rl-arcade"]
CMD ["--help"]
