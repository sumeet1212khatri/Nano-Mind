# 🧠 NanoMind

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![C++17](https://img.shields.io/badge/Standard-C++17-orange.svg)]()
[![HuggingFace Spaces](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-yellow)](https://huggingface.co/spaces/NOT-OMEGA/NanoMind)

**NanoMind** is a custom, end-to-end language model system built entirely from scratch. It features a 152M parameter GPT-2 style architecture trained on GPT-4 quality instruction data, served through a highly optimized, hand-written C++17 inference engine.

[**Launch Live Demo on HuggingFace Spaces →**](https://huggingface.co/spaces/NOT-OMEGA/NanoMind)

---

## ✨ Key Features

* **High-Performance Custom C++ Engine:** Built from the ground up without relying on external inference frameworks like `llama.cpp`. 
* **Hardware Optimized:** Leverages **AVX2 SIMD** instructions (8 floats/instruction) and **OpenMP** parallelism across attention heads and matmul rows.
* **Aggressive Memory Management:** Pre-allocated working buffers ensure zero stack VLAs and zero per-request heap allocations.
* **Session-Persistent KV-Cache:** Stateful daemon process with LRU eviction (up to 20 concurrent sessions) for significantly reduced latency on follow-up prompts.
* **Modern API Backend:** FastAPI serving Server-Sent Events (SSE) for real-time token streaming with a seamless UI integration.

---

## 🚀 Quickstart

The fastest way to get NanoMind running locally is via Docker. Model weights (`model.bin`) and vocabularies are automatically fetched from HuggingFace on the first run.

```bash
git clone [https://github.com/NOT-OMEGA/NanoMind](https://github.com/NOT-OMEGA/NanoMind)
cd NanoMind

# Build and run the container
docker build -t nanomind .
docker run -p 7860:7860 nanomind
```

## 📂 Project Structure


```

NanoMind/
├── inference.cpp        # The core C++ AVX2/OpenMP engine
├── main.py              # FastAPI async server & process manager
├── index.html           # Frontend UI with real-time benchmarks
├── Dockerfile           # Ubuntu-based build env for C++ & Python
├── requirements.txt     # Python dependencies
├── eval_nanomind.py     # 50-question factual evaluation harness
├── test_nanomind.py     # Pytest suite (tokenizer, binary headers)
└── kaggle/
    └── train_slm.ipynb  # End-to-end training notebook

```


## 🛠️ Manual Build (Linux with AVX2)

```

# 1. Compile the C++ inference engine
g++ -O3 -march=native -fopenmp -ffast-math -std=c++17 -o inference inference.cpp -lm

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Download pre-trained weights
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('NOT-OMEGA/NanoMind', 'model.bin')"

# 4. Start the server
python main.py

```


## 📄 License

This project is licensed under the MIT License.
