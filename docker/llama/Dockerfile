FROM debian:13

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y \
    git \
    build-essential \
    cmake \
    curl \
    libvulkan-dev \
    vulkan-tools \
    mesa-vulkan-drivers \
    mesa-common-dev \
    glslc \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/ggerganov/llama.cpp /llama.cpp

WORKDIR /llama.cpp

RUN cmake -B build \
    -DGGML_VULKAN=ON \
    -DGGML_NATIVE=ON \
#    -DGGML_OPENMP=ON \
    -DCMAKE_BUILD_TYPE=Release

RUN cmake --build build -j

EXPOSE 8080

RUN ls -l /llama.cpp/build/bin/llama-server
RUN chmod +x /llama.cpp/build/bin/llama-server
RUN ldd /llama.cpp/build/bin/llama-server || true
