FROM python:3.10-slim

WORKDIR /app
COPY . /app

RUN apt-get update && apt-get install -y g++ libomp-dev curl && rm -rf /var/lib/apt/lists/*

RUN g++ -O3 -march=native -fopenmp -mfma -std=c++17 inference.cpp -o inference -lm
RUN chmod +x inference

RUN pip install --no-cache-dir -r requirements.txt


RUN curl -L -o /app/model.bin "https://huggingface.co/spaces/NOT-OMEGA/Inference/resolve/main/model.bin" && \
    curl -L -o /app/tokenizer.bin "https://huggingface.co/spaces/NOT-OMEGA/Inference/resolve/main/tokenizer.bin"

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
