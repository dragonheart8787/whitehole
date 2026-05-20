# WhiteSearch Docker image — reproducible science environment
# Base: Python 3.11 slim on Debian Bookworm
# Includes: numpy, scipy, astropy, bilby, dynesty, gwpy, pycbc (where available),
#            einsteinpy, ehtim, snakemake, mlflow, dvc

FROM python:3.11-slim-bookworm AS base

LABEL maintainer="WhiteSearch Team"
LABEL description="White hole candidate signal search engine"
LABEL version="0.1.0"

# ── System dependencies ────────────────────────────────────────────────────────
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
    wget \
    && rm -rf /var/lib/apt/lists/*

# ── Python base packages ───────────────────────────────────────────────────────
WORKDIR /opt/whitesearch

COPY pyproject.toml README.md ./

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install core dependencies first (cacheable layer)
RUN pip install --no-cache-dir \
    numpy>=1.26 \
    scipy>=1.12 \
    matplotlib>=3.8 \
    astropy>=6.0 \
    astroquery>=0.4.7 \
    pydantic>=2.6 \
    h5py>=3.10 \
    pandas>=2.2 \
    PyYAML>=6.0 \
    click>=8.1 \
    tqdm>=4.66 \
    scikit-learn>=1.4 \
    corner>=2.2

# ── Bayesian inference ─────────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    bilby>=2.2 \
    dynesty>=2.1

# ── GW tools (Linux-only) ──────────────────────────────────────────────────────
RUN pip install --no-cache-dir gwpy>=3.0 || echo "GWPy install failed — continuing"
RUN pip install --no-cache-dir pycbc>=2.3 || echo "PyCBC install failed — continuing"

# ── EHT / imaging tools ────────────────────────────────────────────────────────
RUN pip install --no-cache-dir einsteinpy>=0.4 || echo "EinsteinPy install failed — continuing"
RUN pip install --no-cache-dir ehtim>=1.2 || echo "ehtim install failed — continuing"

# ── Workflow and tracking ──────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    mlflow>=2.11 \
    dvc>=3.48 \
    snakemake>=8.0 \
    joblib>=1.3

# ── Copy source and install ────────────────────────────────────────────────────
COPY src/ ./src/
COPY configs/ ./configs/
COPY workflows/ ./workflows/
COPY tests/ ./tests/

RUN pip install --no-cache-dir -e "."

# ── Non-root user ──────────────────────────────────────────────────────────────
RUN useradd -m -u 1000 whitesearch
USER whitesearch

WORKDIR /workspace

# ── Entrypoint ─────────────────────────────────────────────────────────────────
ENTRYPOINT ["whitesearch"]
CMD ["--help"]

# ── Metadata labels ────────────────────────────────────────────────────────────
LABEL org.opencontainers.image.source="https://github.com/whitesearch/whitesearch"
LABEL org.opencontainers.image.licenses="MIT"
