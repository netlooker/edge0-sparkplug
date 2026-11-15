# ==============================================================================
# Stage 1: Builder (CUDA 13.0.1 Devel on ARM64 for NVIDIA GB10 / sm_121)
# ==============================================================================
FROM nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    curl \
    ca-certificates \
    python3 \
    python3-venv \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Pin uv to 0.11.6 and force system /usr/bin/python3
COPY --from=ghcr.io/astral-sh/uv:0.11.6 /uv /uvx /bin/
ENV UV_PYTHON_DOWNLOADS=never

# Build at the exact path (/opt/edge0) used by the runtime stage
ARG EDGE0_REPO=https://github.com/Edge0-AI/Edge0.git
ARG EDGE0_REF=7b60076641ccd4adeec8b9a41874a24c477fec02

WORKDIR /opt/edge0
RUN git init && \
    git remote add origin "${EDGE0_REPO}" && \
    git fetch --depth 1 origin "${EDGE0_REF}" && \
    git checkout FETCH_HEAD

# 1. Unpack pinned llama.cpp (ggml backend) required by edge0-strata
WORKDIR /opt/edge0/edge0-strata
RUN mkdir -p third_party/llama.cpp && \
    curl -fsSL https://github.com/ggml-org/llama.cpp/archive/3cf03257f219afbe7334045ff7c6a06ac68c627d.tar.gz \
    | tar -xz --strip-components=1 -C third_party/llama.cpp

# 2. Build edge0-strata for NVIDIA GB10 (Compute Capability 12.1 -> 121)
ENV LIBRARY_PATH="/usr/local/cuda/lib64/stubs:${LIBRARY_PATH}"

RUN cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_CUDA_ARCHITECTURES=121 \
    -DSTRATA_FORCE_ARM64=ON \
    -DSTRATA_ENABLE_CUDA=ON \
    -DSTRATA_GGML_DIR="$PWD/third_party/llama.cpp" && \
    cmake --build build --target strata

# Collect strata binary and any shared libraries from anywhere in the build tree
RUN mkdir -p /opt/edge0/dist-strata-build && \
    cp -a build/strata /opt/edge0/dist-strata-build/ && \
    find build -name '*.so*' -not -path '*/CMakeFiles/*' \
         -exec cp -a {} /opt/edge0/dist-strata-build/ \;

# 3. Create the edge0-strata Python server environment (.venv)
RUN uv venv --python /usr/bin/python3 .venv && \
    uv pip install --python .venv/bin/python -r requirements.txt

# 4. Build edge0-infer CLI & proxy server with static linking enforced
WORKDIR /opt/edge0/edge0-infer
RUN cmake --preset cpu -DBUILD_SHARED_LIBS=OFF && \
    cmake --build --preset cpu

# 5. Strip heavy C++/CUDA build trees & source files so runtime copy is clean & complete
RUN rm -rf /opt/edge0/edge0-strata/build \
           /opt/edge0/edge0-strata/third_party \
           /opt/edge0/edge0-strata/src \
           /opt/edge0/edge0-strata/metal \
           /opt/edge0/edge0-strata/tests && \
    mv /opt/edge0/dist-strata-build /opt/edge0/edge0-strata/build

# ==============================================================================
# Stage 2: Slim Runtime (CUDA 13.0.1 Runtime + Compiled Engine)
# ==============================================================================
FROM nvcr.io/nvidia/cuda:13.0.1-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    libgomp1 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/edge0

# Copy complete cleaned edge0-strata directory and edge0-infer binary to identical paths
COPY --from=builder /opt/edge0/edge0-strata /opt/edge0/edge0-strata
COPY --from=builder /opt/edge0/edge0-infer/build/cpu/bin/edge0-infer /opt/edge0/edge0-infer/build/cpu/bin/edge0-infer

ENV EDGE0_STRATA_DIR=/opt/edge0/edge0-strata
ENV LD_LIBRARY_PATH="/opt/edge0/edge0-strata/build:${LD_LIBRARY_PATH}"
ENV PATH="/opt/edge0/edge0-infer/build/cpu/bin:${PATH}"

ENTRYPOINT ["edge0-infer"]
