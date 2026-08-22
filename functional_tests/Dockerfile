# Hermetic functional-test image for AI-Houkai.
#
# Builds a clean install of the package (cli + dev extras), pre-downloads the
# embedding model so the tests run with no network, then runs the unit suite
# followed by the black-box functional suite against the real console scripts.
#
#   docker build -f functional_tests/Dockerfile -t ai-houkai-functional .
#   docker run --rm ai-houkai-functional
#
# Build context must be the repo root (note the `.` above) so the package source
# is available to pip.

FROM python:3.12-slim

# libgomp1 is required by torch's CPU runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install CPU-only torch first so sentence-transformers doesn't pull the much
# larger default (CUDA) build. This layer caches across source changes.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install the package with cli (typer/rich → houkai) and dev (pytest) extras.
# `dev` is what still carries sentence-transformers: it was moved out of the
# core dependencies into the `local` extra, so a bare `pip install .` no longer
# provides an embedder and these tests would fail at the first recall.
COPY . .
RUN pip install ".[cli,dev]"

# Warm the model cache at build time so the test run needs no network.
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2')"

# Enforce hermeticity at run time: no outbound calls to HF or telemetry.
ENV TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    ANONYMIZED_TELEMETRY=False

# Unit tests first (fast fail), then the functional suite against the installed
# `houkai` / `ai-houkai-serve` entry points.
CMD ["sh", "-c", "pytest -q tests && pytest -v functional_tests"]
