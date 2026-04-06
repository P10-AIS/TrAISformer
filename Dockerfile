# 1. Use Ubuntu 22.04 as a stable, cross-platform base
FROM ubuntu:22.04

# 2. Set environment variables
ENV CONDA_DIR=/opt/conda
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# --- ADD THESE FOR GPU SUPPORT ---
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
# ---------------------------------

# 3. Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    ca-certificates \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 4. Install Micromamba (Auto-detecting Architecture)
RUN set -x && \
    ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then PLATFORM="linux-64"; \
    elif [ "$ARCH" = "aarch64" ]; then PLATFORM="linux-aarch64"; \
    else echo "Unsupported architecture: $ARCH" && exit 1; fi && \
    wget -qO- "https://micro.mamba.pm/api/micromamba/${PLATFORM}/latest" | tar -xj -C /usr/bin/ --strip-components=1 bin/micromamba

# 5. Build the environment
COPY environment.yaml .
RUN micromamba install -y -n base -f environment.yaml && \
    micromamba clean -afy

# 6. Final Setup
WORKDIR /workspaces
CMD ["bash"]