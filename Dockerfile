# Customizable build arguments
ARG CUDA_VER=13.0.0
ARG VLLM_VER=0.19.0

# Base image with CUDA support
FROM nvidia/cuda:${CUDA_VER}-devel-ubuntu22.04

# Re-declare ARGs after FROM to use them in this stage
ARG CUDA_VER
ARG VLLM_VER

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install vLLM with flashinfer using uv
# Format CUDA_VER as cu<major><minor> for the PyTorch extra index URL
# Example: 13.0.0 -> cu130
RUN CUDA_SHORT=$(echo "${CUDA_VER}" | cut -d. -f1,2 | tr -d '.') && \
    uv pip install --system --no-cache "vllm[flashinfer]==${VLLM_VER}" \
    --extra-index-url "https://download.pytorch.org/whl/cu${CUDA_SHORT}"

# Set working directory
WORKDIR /workspace

# Default command
CMD ["/bin/bash"]
