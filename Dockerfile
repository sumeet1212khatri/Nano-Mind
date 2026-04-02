# ============================================================
# NanoMind — HuggingFace Space Dockerfile
# Base: Ubuntu 22.04 | AVX2 + OpenMP C++ build
# Port: 7860 (HF default)
# ============================================================

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_REPO_ID=NOT-OMEGA/NanoMind
ENV N_ENGINES=1
ENV OMP_NUM_THREADS=2

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    g++ \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3    /usr/bin/python

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY inference.cpp  .
COPY main.py        .
COPY index.html     .
COPY model.bin      .
COPY tokenizer.bin  .

RUN g++ -O3 -march=native -fopenmp -ffast-math -std=c++17 \
        -o inference inference.cpp -lm && \
    chmod +x inference && \
    echo "✅ inference binary compiled"

RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

EXPOSE 7860
CMD ["python", "main.py"]
