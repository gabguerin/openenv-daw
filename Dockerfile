# Simulator image (pure Python, multi-arch). Build: docker build -t openenv-daw .
# Works from the package directory (monorepo, public mirror, `openenv push`).
# Dependencies come from requirements.lock (exported from the monorepo uv.lock).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
COPY --from=ghcr.io/astral-sh/uv:0.10.2 /uv /usr/local/bin/uv

WORKDIR /app
RUN uv venv /app/.venv
ENV VIRTUAL_ENV=/app/.venv PATH="/app/.venv/bin:$PATH"
COPY requirements.lock ./
RUN uv pip install --no-cache -r requirements.lock
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN uv pip install --no-cache --no-deps .

ENV MAX_CONCURRENT_ENVS=64 PORT=8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health', timeout=2)" || exit 1

EXPOSE 8000
CMD ["uvicorn", "openenv_daw.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
