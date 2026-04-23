# Customizable build arguments
ARG CUDA_VER=13.0.0
ARG MODEL_ID=Qwen/Qwen3-0.6B

# -----------------------------------------------------------------------------
# Runtime image
# -----------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VER}-devel-ubuntu22.04
ARG CUDA_VER
ARG MODEL_ID

ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/root/.cache/huggingface
ENV MODEL_PATH=/models/${MODEL_ID}

# Install system dependencies
# CRITICAL FIX: Forcefully purge any stale apt lists from the Docker cache 
# before updating to prevent Debian/Ubuntu package conflicts.
RUN rm -rf /var/lib/apt/lists/* && \
    apt-get clean && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory and copy requirements first
WORKDIR /workspace
COPY requirements.txt .

# Install all Python dependencies from requirements.txt
# --index-strategy unsafe-best-match resolves packaging dependency conflicts
RUN CUDA_SHORT=$(echo "${CUDA_VER}" | cut -d. -f1,2 | tr -d '.') && \
    uv pip install --system --no-cache \
    --index-strategy unsafe-best-match \
    -r requirements.txt \
    --extra-index-url "https://download.pytorch.org/whl/cu${CUDA_SHORT}"

# Download the full model snapshot directly into the runtime environment
RUN python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='${MODEL_ID}', local_dir='/models/${MODEL_ID}', local_dir_use_symlinks=False)"

# Critical: Set offline mode AFTER the download to prevent network calls during runtime
ENV HF_HUB_OFFLINE=1

# Copy the serverless handler
COPY handler.py .

# Execute the handler script directly with unbuffered output
CMD ["python3", "-u", "handler.py"]