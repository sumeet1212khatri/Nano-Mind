"""
load_test.py — NanoMind Production Load Test v2.0
==================================================
Google-level stress testing: latency distribution, throughput,
concurrency scaling, cache speedup, rejection behavior, and
stability under sustained load.

Metrics collected (resume-ready):
  - TTFT p50 / p95 / p99         (time-to-first-token)
  - E2E latency p50 / p95 / p99  (full response)
  - Queue wait p50 / p95 / p99   (scheduler overhead)
  - tok/s p50 / p95 / p99        (throughput per request)
  - Total system tok/s            (aggregate throughput)
  - QPS                           (requests per second)
  - 429 rejection rate            (backpressure behavior)
  - KV cache speedup              (warm vs cold TTFT ratio)
  - Error rate                    (stability signal)

Usage:
  pip install aiohttp

  # Full Google-level benchmark (recommended)
  python load_test.py --api https://not-omega-nanomind.hf.space --full

  # Quick concurrency sweep
  python load_test.py --api https://not-omega-nanomind.hf.space --sweep

  # Sustained load (stability test)
  python load_test.py --api https://not-omega-nanomind.hf.space --sustained --duration 60

  # Single concurrency level
  python load_test.py --api https://not-omega-nanomind.hf.space --concurrency 8 --requests 40

  # Local server
  python load_test.py --full
"""

import argparse
import asyncio
import json
import random
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

try:
    import aiohttp
except ImportError:
    print("❌ Missing: pip install aiohttp")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────
# Prompts — categorized by input token length
# ─────────────────────────────────────────────────────────────

PROMPTS = {
    "short": [
        "What is the capital of Japan?",
        "What is 2 + 2?",
        "Who wrote Romeo and Juliet?",
        "What is H2O?",
        "How many days in a week?",
        "What is the speed of light?",
        "What planet is closest to the Sun?",
        "What is the chemical symbol for gold?",
    ],
    "medium": [
        "Explain machine learning in simple terms.",
        "What are 3 benefits of regular exercise?",
        "Write a Python function that reverses a string.",
        "Describe the water cycle briefly.",
        "What is a neural network and how does it learn?",
        "Explain what an API is and give an example.",
        "What is the difference between RAM and storage?",
        "How does HTTPS encryption work?",
    ],
    "long": [
        (
            "Explain the transformer architecture in NLP, covering attention "
            "mechanisms, positional encoding, and how BERT differs from GPT."
        ),
        (
            "Give a detailed explanation of how the internet works, from DNS "
            "resolution to HTTP request handling to rendering in a browser."
        ),
        (
            "Describe the key differences between supervised, unsupervised, and "
            "reinforcement learning with concrete examples for each."
        ),
        (
            "Explain continuous batching in LLM inference systems, why it matters "
            "for throughput, and what tradeoffs it introduces on CPU vs GPU."
        ),
    ],
}

ALL_PROMPTS = PROMPTS["short"] + PROMPTS["medium"] + PROMPTS["long"]


# ─────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────

@dataclass
class Result:
    status:        int
    prompt_len:    str        = "medium"
    ttft_ms:       float      = 0.0
    e2e_ms:        float      = 0.0
    tps:           float      = 0.0
    queue_wait_ms: float      = 0.0
    tokens_out:    int        = 0
    error:         str        = ""
    session_id:    str        = ""


# ─────────────────────────────────────────────────────────────
# Core request function
# ─────────────────────────────────────────────────────────────

async def do_request(
    session: aiohttp.ClientSession,
    api_url: str,
    prompt: str,
    prompt_len: str = "medium",
    max_new: int = 80,
    temperature: float = 0.7,
    top_k: int = 40,
    session_id: Optional[str] = None,
    timeout: float = 90.0,
) -> Result:
    sid = session_id or str(uuid.uuid4())
    payload = {
        "message":        prompt,
        "session_id":     sid,
        "max_new_tokens": max_new,
        "temperature":    temperature,
        "top_k":          top_k,
    }
    t0   = time.perf_counter()
    ttft = None
    toks = 0
    queue_wait = 0.0
    final_tps  = 0.0

    try:
        async with session.post(
            f"{api_url}/chat",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            if resp.status == 429:
                return Result(status=429, prompt_len=prompt_len,
                              error="rejected_429", session_id=sid)
            if resp.status == 503:
                return Result(status=503, prompt_len=prompt_len,
                              error="engine_loading", session_id=sid)
            if resp.status != 200:
                return Result(status=resp.status, prompt_len=prompt_len,
                              error=f"http_{resp.status}", session_id=sid)

            async for raw_line in resp.content:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                ctype = chunk.get("type")
                if ctype == "token":
                    if ttft is None:
                        ttft = (time.perf_counter() - t0) * 1000
                    toks += 1
                elif ctype == "done":
                    final_tps  = chunk.get("tps", 0.0)
                    queue_wait = chunk.get("queue_wait_ms", 0.0)

            e2e = (time.perf_counter() - t0) * 1000
            return Result(
                status=200,
                prompt_len=prompt_len,
                ttft_ms=round(ttft or 0, 1),
                e2e_ms=round(e2e, 1),
                tps=round(final_tps, 1),
                queue_wait_ms=round(queue_wait, 1),
                tokens_out=toks,
                session_id=sid,
            )

    except asyncio.TimeoutError:
        return Result(status=408, prompt_len=prompt_len,
                      error="timeout", session_id=sid)
    except aiohttp.ClientConnectorError:
        return Result(status=0, prompt_len=prompt_len,
                      error="connection_refused", session_id=sid)
    except Exception as exc:
        return Result(status=0, prompt_len=prompt_len,
                      error=str(exc)[:80], session_id=sid)


# ─────────────────────────────────────────────────────────────
# Statistics helpers
# ─────────────────────────────────────────────────────────────

def pct(data: list, p: int) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    i = min(len(s) - 1, int(p / 100 * len(s)))
    return round(s[i], 1)

def mean(data: list) -> float:
    return round(sum(data) / len(data), 1) if data else 0.0

def pct_row(label: str, data: list, unit: str = "ms") -> str:
    if not data:
        return f"  │ {label:<16} │ {'—':>7}  │ {'—':>7}  │ {'—':>7}  │ {'—':>7}  │"
    return (
        f"  │ {label:<16} │ {f'{pct(data,50):.0f}{unit}':>7}  │"
        f" {f'{pct(data,95):.0f}{unit}':>7}  │"
        f" {f'{pct(data,99):.0f}{unit}':>7}  │"
        f" {f'{mean(data):.0f}{unit}':>7}  │"
    )


# ─────────────────────────────────────────────────────────────
# Print summary
# ─────────────────────────────────────────────────────────────

def print_summary(
    results: list,
    concurrency: int,
    elapsed_s: float,
    label: str = "",
) -> dict:
    ok   = [r for r in results if r.status == 200]
    rej  = [r for r in results if r.status == 429]
    tmo  = [r for r in results if r.status == 408]
    err  = [r for r in results if r.status not in (200, 429, 408)]
    n    = len(results)

    ttfts  = [r.ttft_ms       for r in ok if r.ttft_ms > 0]
    e2es   = [r.e2e_ms        for r in ok if r.e2e_ms  > 0]
    tps_l  = [r.tps           for r in ok if r.tps     > 0]
    qwts   = [r.queue_wait_ms for r in ok if r.queue_wait_ms > 0]
    t_out  = sum(r.tokens_out for r in ok)

    qps        = round(n / max(elapsed_s, 0.1), 2)
    sys_tps    = round(t_out / max(elapsed_s, 0.1), 1)
    success_rt = round(100 * len(ok) / max(n, 1), 1)
    rej_rt     = round(100 * len(rej) / max(n, 1), 1)
    err_rt     = round(100 * (len(tmo) + len(err)) / max(n, 1), 1)

    W = 65
    hdr = f"  concurrency={concurrency}  requests={n}"
    if label:
        hdr += f"  [{label}]"

    print()
    print("=" * W)
    print(f"  NanoMind Load Test{hdr}")
    print("=" * W)
    print(f"  Wall time        : {elapsed_s:.1f}s")
    print(f"  QPS              : {qps}")
    print(f"  System tok/s     : {sys_tps}  ({t_out} tokens total)")
    print(f"  Success          : {len(ok)}/{n}  ({success_rt}%)")
    print(f"  Rejected (429)   : {len(rej)}  ({rej_rt}%)")
    print(f"  Timeouts         : {len(tmo)}")
    print(f"  Other errors     : {len(err)}")
    print()
    print(f"  ┌{'─'*18}┬{'─'*9}┬{'─'*9}┬{'─'*9}┬{'─'*9}┐")
    print(f"  │ {'Metric':<16}  │ {'p50':>7}  │ {'p95':>7}  │ {'p99':>7}  │ {'avg':>7}  │")
    print(f"  ├{'─'*18}┼{'─'*9}┼{'─'*9}┼{'─'*9}┼{'─'*9}┤")
    print(pct_row("TTFT",       ttfts))
    print(pct_row("E2E latency", e2es))
    print(pct_row("Queue wait",  qwts))
    print(pct_row("TPS / req",   tps_l, unit=""))
    print(f"  └{'─'*18}┴{'─'*9}┴{'─'*9}┴{'─'*9}┴{'─'*9}┘")

    # Verdict
    print()
    p95_ttft = pct(ttfts, 95)
    p95_e2e  = pct(e2es, 95)
    avg_tps  = mean(tps_l)
    if not ok:
        print("  ⚠️  No successful requests — check server is running.")
    else:
        verdict = []
        verdict.append(
            f"  {'✅' if p95_ttft < 1500 else '⚠️ ' if p95_ttft < 4000 else '❌'}"
            f"  TTFT p95 = {p95_ttft:.0f}ms"
            + (" (good)" if p95_ttft < 1500 else
               " (ok)" if p95_ttft < 4000 else
               " (bad — serialization or overload)")
        )
        verdict.append(
            f"  {'✅' if avg_tps > 25 else '⚠️ ' if avg_tps > 15 else '❌'}"
            f"  Avg TPS = {avg_tps:.0f} tok/s"
        )
        if rej_rt > 20:
            verdict.append(f"  ⚠️   {rej_rt:.0f}% rejected — raise MAX_INFLIGHT or add engines")
        elif rej_rt > 0:
            verdict.append(f"  ℹ️   {rej_rt:.0f}% rejected — backpressure working correctly")
        if err_rt > 5:
            verdict.append(f"  ❌  {err_rt:.0f}% error rate — investigate stability")
        for v in verdict:
            print(v)

    print("=" * W)

    return {
        "concurrency":  concurrency,
        "n":            n,
        "ok":           len(ok),
        "qps":          qps,
        "sys_tps":      sys_tps,
        "ttft_p50":     pct(ttfts, 50),
        "ttft_p95":     pct(ttfts, 95),
        "ttft_p99":     pct(ttfts, 99),
        "e2e_p50":      pct(e2es,  50),
        "e2e_p95":      pct(e2es,  95),
        "tps_avg":      avg_tps,
        "tps_p50":      pct(tps_l, 50),
        "rej_pct":      rej_rt,
        "success_pct":  success_rt,
        "t_total":      t_out,
    }


# ─────────────────────────────────────────────────────────────
# Test runners
# ─────────────────────────────────────────────────────────────

async def run_batch(
    api_url: str,
    concurrency: int,
    total_requests: int,
    max_new: int = 80,
    prompt_mix: str = "all",
    reuse_session: bool = False,
    seed: int = 42,
) -> list:
    """Run `total_requests` requests with `concurrency` in-flight at once."""
    rng = random.Random(seed)
    if prompt_mix == "short":
        pool = PROMPTS["short"]
    elif prompt_mix == "medium":
        pool = PROMPTS["medium"]
    elif prompt_mix == "long":
        pool = PROMPTS["long"]
    else:
        pool = ALL_PROMPTS

    shared_sid = str(uuid.uuid4()) if reuse_session else None
    prompts    = [(rng.choice(pool), prompt_mix) for _ in range(total_requests)]
    results    = []
    sem        = asyncio.Semaphore(concurrency)

    async def bounded(prompt: str, cat: str):
        async with sem:
            async with aiohttp.ClientSession() as s:
                r = await do_request(
                    s, api_url, prompt, cat, max_new,
                    session_id=shared_sid,
                )
                results.append(r)

    await asyncio.gather(*[asyncio.create_task(bounded(p, c)) for p, c in prompts])
    return results


async def run_kv_cache_test(api_url: str) -> dict:
    """
    Measure KV cache speedup:
      Turn 1 (cold): full system prompt + first message
      Turn 2 (warm): same session, follow-up question
    Returns speedup ratio.
    """
    print("\n  📊 KV Cache Speedup Test...")
    sid = str(uuid.uuid4())

    cold_times = []
    warm_times = []

    for _ in range(5):
        sid = str(uuid.uuid4())
        async with aiohttp.ClientSession() as s:
            # Cold
            r = await do_request(
                s, api_url, "What is machine learning?",
                max_new=60, temperature=0.1, session_id=sid
            )
            if r.status == 200:
                cold_times.append(r.ttft_ms)

            # Warm (same session, follow-up)
            r2 = await do_request(
                s, api_url, "Give me one more example.",
                max_new=60, temperature=0.1, session_id=sid
            )
            if r2.status == 200:
                warm_times.append(r2.ttft_ms)

    cold_avg = mean(cold_times)
    warm_avg = mean(warm_times)
    speedup  = round(cold_avg / warm_avg, 2) if warm_avg > 0 else 0

    print(f"  Cold TTFT avg : {cold_avg:.0f}ms  (n={len(cold_times)})")
    print(f"  Warm TTFT avg : {warm_avg:.0f}ms  (n={len(warm_times)})")
    print(f"  KV speedup    : {speedup}x {'✅' if speedup > 1.2 else '—'}")

    return {"cold_ttft_ms": cold_avg, "warm_ttft_ms": warm_avg, "speedup_x": speedup}


async def run_health_check(api_url: str) -> bool:
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(
                f"{api_url}/health",
                timeout=aiohttp.ClientTimeout(total=15)
            ) as r:
                d = await r.json()
                ready = d.get("engines_ready", 0)
                print(f"  status         : {d.get('status', '?')}")
                print(f"  engines_ready  : {ready}")
                print(f"  active_sessions: {d.get('active_sessions', '?')}")
                print(f"  inflight       : {d.get('inflight_requests', '?')}")
                print(f"  ram_mb         : {d.get('process_ram_mb', '?')}")
                print(f"  uptime_s       : {d.get('uptime_seconds', '?')}")
                return ready > 0
    except Exception as e:
        print(f"  ❌ Cannot reach {api_url}: {e}")
        return False


async def run_sustained(api_url: str, duration_s: int, concurrency: int, max_new: int):
    """
    Fire requests continuously for `duration_s` seconds.
    Measures stability: does TPS degrade? Do errors increase over time?
    """
    print(f"\n  ⏱  Sustained load: {duration_s}s @ concurrency={concurrency}")
    results  = []
    t_start  = time.perf_counter()
    rng      = random.Random(99)
    sem      = asyncio.Semaphore(concurrency)
    stop     = asyncio.Event()

    async def worker():
        while not stop.is_set():
            prompt = rng.choice(ALL_PROMPTS)
            async with sem:
                if stop.is_set():
                    return
                async with aiohttp.ClientSession() as s:
                    r = await do_request(s, api_url, prompt, max_new=max_new)
                    results.append((time.perf_counter() - t_start, r))

    tasks = [asyncio.create_task(worker()) for _ in range(concurrency * 2)]
    await asyncio.sleep(duration_s)
    stop.set()
    await asyncio.gather(*tasks, return_exceptions=True)

    elapsed = time.perf_counter() - t_start

    # Split into 3 thirds to check for degradation
    third  = elapsed / 3
    t1 = [r for ts, r in results if ts < third]
    t2 = [r for ts, r in results if third <= ts < 2*third]
    t3 = [r for ts, r in results if ts >= 2*third]

    def ok_tps(batch):
        ok = [r for r in batch if r.status == 200 and r.tps > 0]
        return mean([r.tps for r in ok])

    all_r = [r for _, r in results]
    print(f"  Total requests : {len(all_r)}")
    print(f"  T1 avg TPS     : {ok_tps(t1):.1f}  (first {duration_s//3}s)")
    print(f"  T2 avg TPS     : {ok_tps(t2):.1f}  (middle {duration_s//3}s)")
    print(f"  T3 avg TPS     : {ok_tps(t3):.1f}  (last {duration_s//3}s)")
    degradation = "✅ stable" if abs(ok_tps(t1) - ok_tps(t3)) < 5 else "⚠️  degrading"
    print(f"  Stability      : {degradation}")

    return all_r


# ─────────────────────────────────────────────────────────────
# Sweep
# ─────────────────────────────────────────────────────────────

async def run_sweep(api_url: str, max_new: int):
    print("\n🔬 Concurrency Sweep: 1 → 2 → 4 → 8 → 16")
    sweep_results = []
    for c in (1, 2, 4, 8, 16):
        n  = max(c * 4, 8)
        t0 = time.perf_counter()
        r  = await run_batch(api_url, c, n, max_new)
        sr = print_summary(r, c, time.perf_counter() - t0, label=f"sweep c={c}")
        sweep_results.append(sr)
        await asyncio.sleep(2)

    # Scaling table
    print("\n  📈 Scaling Summary (TTFT p95)")
    print(f"  {'Concurrency':>12} │ {'TTFT p95':>9} │ {'TPS avg':>8} │ {'QPS':>6} │ {'SysTPS':>7}")
    print(f"  {'─'*12}─┼─{'─'*9}─┼─{'─'*8}─┼─{'─'*6}─┼─{'─'*7}")
    for sr in sweep_results:
        print(
            f"  {sr['concurrency']:>12} │ {sr['ttft_p95']:>8.0f}ms │"
            f" {sr['tps_avg']:>7.1f}   │ {sr['qps']:>6.1f} │ {sr['sys_tps']:>7.1f}"
        )


# ─────────────────────────────────────────────────────────────
# Full Google-level benchmark
# ─────────────────────────────────────────────────────────────

async def run_full(api_url: str, max_new: int):
    """
    Full benchmark suite — produces all numbers needed for resume.
    """
    all_stats = {}

    # 1. Baseline single request
    print("\n[1/6] Baseline — Single request (warmup)")
    r = await run_batch(api_url, 1, 5, max_new, seed=1)
    print_summary(r, 1, sum(x.e2e_ms for x in r if x.status==200)/1000, label="baseline")

    # 2. Concurrency scaling
    print("\n[2/6] Concurrency scaling sweep")
    await run_sweep(api_url, max_new)

    # 3. KV cache speedup
    print("\n[3/6] KV cache speedup")
    kv = await run_kv_cache_test(api_url)
    all_stats["kv_cache"] = kv

    # 4. Input length sensitivity (short vs long prompts)
    print("\n[4/6] Input length sensitivity")
    for mix, label in [("short", "short prompts"), ("long", "long prompts")]:
        t0 = time.perf_counter()
        r  = await run_batch(api_url, 4, 12, max_new, prompt_mix=mix)
        print_summary(r, 4, time.perf_counter()-t0, label=label)

    # 5. Backpressure test (hammer with 32 concurrent)
    print("\n[5/6] Backpressure / overload test (concurrency=32)")
    t0 = time.perf_counter()
    r  = await run_batch(api_url, 32, 48, max_new)
    print_summary(r, 32, time.perf_counter()-t0, label="overload")

    # 6. Sustained stability (30s)
    print("\n[6/6] Sustained stability (30s)")
    sr = await run_sustained(api_url, 30, 4, max_new)
    t0 = 30.0
    print_summary(sr, 4, t0, label="sustained-30s")

    # Final resume numbers
    print("\n" + "=" * 65)
    print("  📋 RESUME-READY NUMBERS")
    print("=" * 65)
    kv = all_stats.get("kv_cache", {})
    print(f"  KV cache speedup  : {kv.get('speedup_x', '?')}x")
    print(f"  Cold TTFT         : {kv.get('cold_ttft_ms', '?'):.0f}ms")
    print(f"  Warm TTFT         : {kv.get('warm_ttft_ms', '?'):.0f}ms")
    print()
    print("  Run --sweep for full TTFT/TPS scaling table.")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(
        description="NanoMind Load Test v2.0 — Google-level benchmark"
    )
    parser.add_argument("--api",         default="http://localhost:7860",
                        help="API base URL")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="Concurrent requests")
    parser.add_argument("--requests",    type=int, default=32,
                        help="Total requests")
    parser.add_argument("--max-new",     type=int, default=80,
                        help="Max new tokens per request")
    parser.add_argument("--sweep",       action="store_true",
                        help="Concurrency sweep 1→2→4→8→16")
    parser.add_argument("--full",        action="store_true",
                        help="Full Google-level benchmark suite")
    parser.add_argument("--sustained",   action="store_true",
                        help="Sustained load stability test")
    parser.add_argument("--duration",    type=int, default=60,
                        help="Sustained test duration in seconds")
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print(f"  NanoMind Load Test v2.0")
    print(f"  Target: {args.api}")
    print(f"{'='*65}")

    # Health check
    print("\n🏥 Health Check")
    ready = await run_health_check(args.api)
    if not ready:
        print("\n❌ Server not ready. Aborting.")
        print("   If using HF Spaces, wait 60s for engine startup.")
        return

    if args.full:
        await run_full(args.api, args.max_new)

    elif args.sweep:
        await run_sweep(args.api, args.max_new)

    elif args.sustained:
        r  = await run_sustained(args.api, args.duration, args.concurrency, args.max_new)
        print_summary(r, args.concurrency, float(args.duration), label="sustained")

    else:
        t0 = time.perf_counter()
        r  = await run_batch(
            args.api, args.concurrency, args.requests, args.max_new
        )
        print_summary(r, args.concurrency, time.perf_counter()-t0)


if __name__ == "__main__":
    asyncio.run(main())
