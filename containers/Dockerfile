# WhiteSearch — reproducible science environment (Linux)
# Build from repository root:
#   docker build -f containers/Dockerfile -t whitesearch:latest .

FROM python:3.11-slim-bookworm AS base

LABEL org.opencontainers.image.title="WhiteSearch"
LABEL org.opencontainers.image.description="White hole candidate signal search engine"
LABEL org.opencontainers.image.source="https://github.com/dragonheart8787/whitehole"
LABEL org.opencontainers.image.licenses="MIT"
LABEL version="0.1.1"

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    gfortran \
    libfftw3-dev \
    libhdf5-dev \
    libgsl-dev \
    libopenblas-dev \
    liblapack-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/whitesearch

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY configs/ ./configs/
COPY workflows/ ./workflows/

RUN pip install --upgrade pip setuptools wheel \
    && pip install -e ".[science,gw]" \
    && pip install pytest>=8.0

# Optional heavy extras (best-effort; image still usable if they fail)
RUN pip install "pycbc>=2.3" || echo "WARN: PyCBC not installed"
RUN pip install "einsteinpy>=0.4" "ehtim>=1.2" || echo "WARN: imaging extras not installed"

COPY containers/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

RUN useradd -m -u 1000 -s /bin/bash whitesearch \
    && chown -R whitesearch:whitesearch /opt/whitesearch

USER whitesearch
WORKDIR /workspace

VOLUME ["/workspace"]

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["--help"]
