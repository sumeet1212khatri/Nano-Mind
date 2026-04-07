# 🧠 NanoMind

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![C++17](https://img.shields.io/badge/Standard-C++17-orange.svg)]()
[![HuggingFace Spaces](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-yellow)](https://huggingface.co/spaces/NOT-OMEGA/NanoMind)

**NanoMind** is a custom, end-to-end language model system built entirely from scratch. It features a 152M parameter GPT-2 style architecture trained on GPT-4 quality instruction data, served through a highly optimized, hand-written C++17 inference engine.

[**Launch Live Demo on HuggingFace Spaces →**](https://huggingface.co/spaces/NOT-OMEGA/NanoMind)

---
## Benchmarks

<img width="247" height="199" alt="image" src="https://github.com/user-attachments/assets/e87eda99-48b1-4eb1-99ea-52b0cd462aeb" />


## SYSTEM DESIGN

[HHD Design](https://app.eraser.io/workspace/RMsX2vhTVpB6uaZgnBqV?origin=share)

<img width="1648" height="841" alt="image" src="https://github.com/user-attachments/assets/396dac9c-4256-48d7-808b-8fc4718df27a" />



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


## Load Test Results 
```
=================================================================
  NanoMind Load Test v2.0
  Target: https://not-omega-nanomind.hf.space
=================================================================

🏥 Health Check
  status         : ok
  engines_ready  : 2
  active_sessions: 10
  inflight       : 0
  ram_mb         : 1737.1
  uptime_s       : 244.9

🔬 Concurrency Sweep: 1 → 2 → 4 → 8 → 16

=================================================================
  NanoMind Load Test  concurrency=1  requests=8  [sweep c=1]
=================================================================
  Wall time        : 28.9s
  QPS              : 0.28
  System tok/s     : 18.4  (532 tokens total)
  Success          : 8/8  (100.0%)
  Rejected (429)   : 0  (0.0%)
  Timeouts         : 0
  Other errors     : 0

  ┌──────────────────┬─────────┬─────────┬─────────┬─────────┐
  │ Metric            │     p50  │     p95  │     p99  │     avg  │
  ├──────────────────┼─────────┼─────────┼─────────┼─────────┤
  │ TTFT             │  1652ms  │  2053ms  │  2053ms  │  1706ms  │
  │ E2E latency      │  3950ms  │  4342ms  │  4342ms  │  3612ms  │
  │ Queue wait       │       —  │       —  │       —  │       —  │
  │ TPS / req        │      35  │      37  │      37  │      35  │
  └──────────────────┴─────────┴─────────┴─────────┴─────────┘

  ⚠️   TTFT p95 = 2053ms (ok)
  ✅  Avg TPS = 35 tok/s
=================================================================

=================================================================
  NanoMind Load Test  concurrency=2  requests=8  [sweep c=2]
=================================================================
  Wall time        : 22.4s
  QPS              : 0.36
  System tok/s     : 21.1  (472 tokens total)
  Success          : 8/8  (100.0%)
  Rejected (429)   : 0  (0.0%)
  Timeouts         : 0
  Other errors     : 0

  ┌──────────────────┬─────────┬─────────┬─────────┬─────────┐
  │ Metric            │     p50  │     p95  │     p99  │     avg  │
  ├──────────────────┼─────────┼─────────┼─────────┼─────────┤
  │ TTFT             │  2521ms  │  3497ms  │  3497ms  │  2576ms  │
  │ E2E latency      │  5982ms  │  6266ms  │  6266ms  │  5332ms  │
  │ Queue wait       │   428ms  │   428ms  │   428ms  │   390ms  │
  │ TPS / req        │      21  │      36  │      36  │      21  │
  └──────────────────┴─────────┴─────────┴─────────┴─────────┘

  ⚠️   TTFT p95 = 3497ms (ok)
  ⚠️   Avg TPS = 21 tok/s
=================================================================

=================================================================
  NanoMind Load Test  concurrency=4  requests=16  [sweep c=4]
=================================================================
  Wall time        : 46.3s
  QPS              : 0.35
  System tok/s     : 20.8  (965 tokens total)
  Success          : 16/16  (100.0%)
  Rejected (429)   : 0  (0.0%)
  Timeouts         : 0
  Other errors     : 0

  ┌──────────────────┬─────────┬─────────┬─────────┬─────────┐
  │ Metric            │     p50  │     p95  │     p99  │     avg  │
  ├──────────────────┼─────────┼─────────┼─────────┼─────────┤
  │ TTFT             │  8002ms  │ 10411ms  │ 10411ms  │  7155ms  │
  │ E2E latency      │ 11366ms  │ 15324ms  │ 15324ms  │ 10958ms  │
  │ Queue wait       │  5705ms  │  6848ms  │  6848ms  │  5031ms  │
  │ TPS / req        │      16  │      37  │      37  │      17  │
  └──────────────────┴─────────┴─────────┴─────────┴─────────┘

  ❌  TTFT p95 = 10411ms (bad — serialization or overload)
  ⚠️   Avg TPS = 17 tok/s
=================================================================

=================================================================
  NanoMind Load Test  concurrency=8  requests=32  [sweep c=8]
=================================================================
  Wall time        : 99.3s
  QPS              : 0.32
  System tok/s     : 22.1  (2195 tokens total)
  Success          : 32/32  (100.0%)
  Rejected (429)   : 0  (0.0%)
  Timeouts         : 0
  Other errors     : 0

  ┌──────────────────┬─────────┬─────────┬─────────┬─────────┐
  │ Metric            │     p50  │     p95  │     p99  │     avg  │
  ├──────────────────┼─────────┼─────────┼─────────┼─────────┤
  │ TTFT             │ 17204ms  │ 25913ms  │ 27389ms  │ 18011ms  │
  │ E2E latency      │ 22388ms  │ 30782ms  │ 30929ms  │ 22194ms  │
  │ Queue wait       │ 14424ms  │ 23024ms  │ 24297ms  │ 16110ms  │
  │ TPS / req        │      17  │      18  │      31  │      17  │
  └──────────────────┴─────────┴─────────┴─────────┴─────────┘

  ❌  TTFT p95 = 25913ms (bad — serialization or overload)
  ⚠️   Avg TPS = 17 tok/s
=================================================================

=================================================================
  NanoMind Load Test  concurrency=16  requests=64  [sweep c=16]
=================================================================
  Wall time        : 135.0s
  QPS              : 0.47
  System tok/s     : 11.9  (1607 tokens total)
  Success          : 64/64  (100.0%)
  Rejected (429)   : 0  (0.0%)
  Timeouts         : 0
  Other errors     : 0

  ┌──────────────────┬─────────┬─────────┬─────────┬─────────┐
  │ Metric            │     p50  │     p95  │     p99  │     avg  │
  ├──────────────────┼─────────┼─────────┼─────────┼─────────┤
  │ TTFT             │ 25896ms  │ 30816ms  │ 30934ms  │ 23456ms  │
  │ E2E latency      │ 30897ms  │ 35948ms  │ 36170ms  │ 29332ms  │
  │ Queue wait       │ 25525ms  │ 29097ms  │ 29802ms  │ 23290ms  │
  │ TPS / req        │      14  │      17  │      17  │      10  │
  └──────────────────┴─────────┴─────────┴─────────┴─────────┘

  ❌  TTFT p95 = 30816ms (bad — serialization or overload)
  ❌  Avg TPS = 10 tok/s
=================================================================

  📈 Scaling Summary (TTFT p95)
   Concurrency │  TTFT p95 │  TPS avg │    QPS │  SysTPS
  ─────────────┼───────────┼──────────┼────────┼────────
             1 │     2053ms │    35.2   │    0.3 │    18.4
             2 │     3497ms │    21.1   │    0.4 │    21.1
             4 │    10411ms │    17.3   │    0.3 │    20.8
             8 │    25913ms │    16.9   │    0.3 │    22.1
            16 │    30816ms │    10.1   │    0.5 │    11.9

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
