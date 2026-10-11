# Stage 1: dependencies only. This layer depends on pyproject.toml alone, so a
# code change does not rebuild (or re-download on the server) the large
# dependency layer.
FROM python:3.12-slim AS deps

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
COPY pyproject.toml .
# Install the declared dependencies (not the project itself). Bytecode is
# compiled at runtime instead of shipped, and bundled test suites and pip are
# dropped: together ~60 MB less to download per deploy.
RUN python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml', 'rb'))['project']['dependencies']))" > requirements.txt \
 && pip install --no-compile -r requirements.txt \
 && pip uninstall -y pip \
 && find /opt/venv -depth -type d -name tests -path '*/site-packages/*' -exec rm -rf {} + \
 && find /opt/venv -depth -type d -name __pycache__ -exec rm -rf {} +

# Stage 2: runtime image = dependency layer + application source.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/home/appuser/app

COPY --from=deps /opt/venv /opt/venv

RUN useradd --create-home appuser \
 && install -d -m 0755 -o appuser -g appuser /home/appuser/.tradingagents
USER appuser
WORKDIR /home/appuser/app

COPY --chown=appuser:appuser . .

ARG PORT=8503
# Also available at runtime (CMD and HEALTHCHECK run after the build); a PORT
# from the deployment environment still overrides it.
ENV PORT=$PORT
EXPOSE $PORT

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/_stcore/health', timeout=4)"

CMD streamlit run app/streamlit_app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true
