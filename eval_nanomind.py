# ============================================================
# Stage 1: Builder — compiles C++ AVX2 engine with batch prefill
# ============================================================
FROM ubuntu:22.04 AS builder

RUN apt-get update && apt-get install -y g++ libgomp1

WORKDIR /build
COPY inference.cpp .

# -mavx2 -mfma: AVX2 + FMA (dot products, matmul)
# -funroll-loops: loop unrolling for inner matmul loops
# -flto: link-time optimization (inlines matmul_vec_serial into OMP regions)
# -fno-math-errno: skip errno checks in math (safe for inference)
RUN g++ -O3 -mavx2 -mfma -fopenmp \
        -ffast-math -funroll-loops -flto \
        -fno-math-errno \
        -std=c++17 \
        -o inference inference.cpp -lm && \
    echo "✅ inference binary compiled" && \
    ls -lh inference

# ============================================================
# Stage 2: Production runtime
# ============================================================
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV HF_REPO_ID=NOT-OMEGA/NanoMind

# 3 engines × 1 OMP thread = best CPU utilization on 2-vCPU HF Spaces
# 3 engines handle 3 concurrent requests without any queue wait
# OMP=1 prevents thread contention between engines
ENV N_ENGINES=3
ENV OMP_NUM_THREADS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libstdc++6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy compiled binary from builder
COPY --from=builder /build/inference .

# Application files
COPY main.py index.html ./

# Model weights (bundled — avoids HF download delay on cold start)
COPY model.bin tokenizer.bin ./

RUN chmod +x inference && \
    useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

EXPOSE 7860
CMD ["python", "main.py"]
