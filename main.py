"""
main.py — NanoMind Production Server v2.0
=========================================
Changes from v1.5:
  ✅ Bounded inflight counter → HTTP 429 backpressure
  ✅ Per-request tracing: queue_wait_ms, ttft_ms, e2e_ms
  ✅ Rolling P50/P95/P99 latency percentiles (LatencyTracker)
  ✅ Request timeout (configurable, default 30s)
  ✅ "scheduled" event from pool.generate() measures true queue wait
  ✅ /metrics returns full latency distribution
  ✅ /health exposes inflight count and engine busy state
  ✅ done payload includes queue_wait_ms + ttft_ms for UI display
  ✅ error_rate_pct in metrics
"""

import asyncio
import collections
import json
import os
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional

import psutil
import tiktoken
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from huggingface_hub import hf_hub_download

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
INFERENCE_EXE = BASE_DIR / "inference"
MODEL_BIN     = BASE_DIR / "model.bin"
TOKENIZER_BIN = BASE_DIR / "tokenizer.bin"
HF_REPO_ID    = os.environ.get("HF_REPO_ID", "NOT-OMEGA/NanoMind")

SYSTEM_TOKEN = "System:"
USER_TOKEN   = "User:"
ASST_TOKEN   = "Assistant:"
SEP          = "\n"

BLOCK_SIZE         = 1024
MAX_GEN_CEILING    = 256
SAFETY_MARGIN      = 24
MAX_SESSION_TOKENS = BLOCK_SIZE - MAX_GEN_CEILING - SAFETY_MARGIN
MAX_INPUT_CHARS    = 2000

N_ENGINES        = int(os.environ.get("N_ENGINES",       "1"))
OMP_PER_ENGINE   = int(os.environ.get("OMP_NUM_THREADS", "2"))
SESSION_TTL_S    = int(os.environ.get("SESSION_TTL",     "1800"))
# ── NEW: tuneable via env ────────────────────────────────────
MAX_INFLIGHT     = int(os.environ.get("MAX_INFLIGHT",    "32"))   # backpressure ceiling
REQUEST_TIMEOUT  = float(os.environ.get("REQUEST_TIMEOUT", "30")) # seconds
LATENCY_WINDOW   = 500                                             # rolling window samples

enc            = tiktoken.get_encoding("gpt2")
STOP_TOKEN_IDS = [50256]
STOP_STRINGS   = ["User:", "System:"]


# ─────────────────────────────────────────────────────────────
# LatencyTracker — rolling P50/P95/P99
# ─────────────────────────────────────────────────────────────
class LatencyTracker:
    """
    Maintains a fixed-size rolling window of the last N samples
    for each of: queue_wait, TTFT, end-to-end latency, tok/s.
    Thread-safe under asyncio (single event loop).
    """

    def __init__(self, maxlen: int = LATENCY_WINDOW):
        self._queue_wait = collections.deque(maxlen=maxlen)
        self._ttft       = collections.deque(maxlen=maxlen)
        self._e2e        = collections.deque(maxlen=maxlen)
        self._tps        = collections.deque(maxlen=maxlen)

    def record(
        self,
        *,
        queue_wait_ms: float,
        ttft_ms: float,
        e2e_ms: float,
        tps: float,
    ) -> None:
        self._queue_wait.append(queue_wait_ms)
        self._ttft.append(ttft_ms)
        self._e2e.append(e2e_ms)
        self._tps.append(tps)

    @staticmethod
    def _pct(d: collections.deque, ps=(50, 95, 99)) -> dict:
        if not d:
            return {f"p{p}": 0.0 for p in ps}
        s = sorted(d)
        n = len(s)
        return {f"p{p}": round(s[min(n - 1, int(p / 100 * n))], 1) for p in ps}

    def summary(self) -> dict:
        return {
            "samples":       len(self._e2e),
            "queue_wait_ms": self._pct(self._queue_wait),
            "ttft_ms":       self._pct(self._ttft),
            "e2e_ms":        self._pct(self._e2e),
            "tps":           self._pct(self._tps),
        }


# ─────────────────────────────────────────────────────────────
# InferenceEngine
# ─────────────────────────────────────────────────────────────
class InferenceEngine:
    def __init__(self, eid: int):
        self.eid    = eid
        self._proc  = None
        self._ready = False

    async def start(self):
        if not INFERENCE_EXE.exists():
            raise RuntimeError(f"inference binary not found: {INFERENCE_EXE}")
        if not MODEL_BIN.exists():
            raise RuntimeError(f"model.bin not found: {MODEL_BIN}")
        print(f"[engine-{self.eid}] launching...")
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(OMP_PER_ENGINE)
        self._proc = await asyncio.create_subprocess_exec(
            str(INFERENCE_EXE),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(BASE_DIR),
            env=env,
        )
        try:
            async with asyncio.timeout(120):
                while True:
                    raw = await self._proc.stdout.readline()
                    if not raw:
                        err = await self._proc.stderr.read()
                        raise RuntimeError(
                            f"[engine-{self.eid}] died before READY. "
                            f"stderr: {err.decode(errors='replace')[:500]}"
                        )
                    line = raw.decode().strip()
                    if line:
                        print(f"[engine-{self.eid}] {line}")
                    if line == "READY":
                        self._ready = True
                        print(f"[engine-{self.eid}] ready  pid={self._proc.pid}")
                        break
                    if line.startswith("ERROR"):
                        raise RuntimeError(f"[engine-{self.eid}] startup error: {line}")
        except TimeoutError:
            self._proc.kill()
            raise RuntimeError(f"[engine-{self.eid}] timed out waiting for READY")

    async def stop(self):
        self._ready = False
        if self._proc:
            try:
                self._proc.stdin.write(b"QUIT\n")
                await self._proc.stdin.drain()
                await asyncio.wait_for(self._proc.wait(), timeout=3.0)
            except Exception:
                self._proc.kill()
            self._proc = None

    async def reset_session(self, session_id: str):
        if not self._ready or not self._proc:
            return
        self._proc.stdin.write(f"RESET|{session_id}\n".encode())
        await self._proc.stdin.drain()
        try:
            async with asyncio.timeout(5):
                while True:
                    raw = await self._proc.stdout.readline()
                    if not raw or raw.decode().strip() == "RESET_OK":
                        break
        except TimeoutError:
            pass

    async def generate(
        self, session_id, new_token_ids, max_new, temperature, top_k
    ) -> AsyncIterator[dict]:
        if not self._ready or not self._proc:
            yield {"type": "error", "message": f"engine-{self.eid} not ready"}
            return
        tokens_csv = ",".join(map(str, new_token_ids))
        stop_csv   = ",".join(map(str, STOP_TOKEN_IDS))
        cmd = (
            f"REQUEST|{session_id}|{tokens_csv}"
            f"|{max_new}|{temperature}|{top_k}|{stop_csv}\n"
        )
        self._proc.stdin.write(cmd.encode())
        await self._proc.stdin.drain()

        byte_buffer = bytearray()
        try:
            while True:
                raw = await self._proc.stdout.readline()
                if not raw:
                    self._ready = False
                    yield {"type": "error", "message": "engine process died unexpectedly"}
                    return
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue

                if line.startswith("TOKEN"):
                    parts    = line.split()
                    token_id = int(parts[1])
                    elapsed  = float(parts[2])
                    byte_buffer.extend(enc.decode_bytes([token_id]))
                    try:
                        decoded = byte_buffer.decode("utf-8")
                        byte_buffer.clear()
                    except UnicodeDecodeError:
                        continue
                    if decoded:
                        yield {
                            "type":       "token",
                            "id":         token_id,
                            "text":       decoded,
                            "elapsed_ms": elapsed,
                        }

                elif line.startswith("DONE"):
                    if byte_buffer:
                        leftover = byte_buffer.decode("utf-8", errors="replace")
                        byte_buffer.clear()
                        if leftover:
                            yield {"type": "token", "id": -1, "text": leftover, "elapsed_ms": 0.0}
                    parts    = line.split()
                    total_t  = int(parts[1])
                    total_ms = float(parts[2])
                    tps = round(total_t / (total_ms / 1000.0), 2) if total_ms > 0 else 0.0
                    yield {
                        "type":         "done",
                        "total_tokens": total_t,
                        "total_ms":     total_ms,
                        "tps":          tps,
                    }
                    return

                elif line.startswith("ERROR"):
                    yield {"type": "error", "message": line}
                    return

        except asyncio.CancelledError:
            # Drain engine output so it's not stuck mid-stream
            try:
                async with asyncio.timeout(5):
                    while True:
                        raw = await self._proc.stdout.readline()
                        if not raw:
                            break
                        if raw.decode().strip().startswith(("DONE", "ERROR")):
                            break
            except Exception:
                pass
            raise

    @property
    def pid(self):
        return self._proc.pid if self._proc else None


# ─────────────────────────────────────────────────────────────
# EnginePool
# ─────────────────────────────────────────────────────────────
class EnginePool:
    """
    Manages N inference engine processes.
    Key change: generate() now emits a "scheduled" event containing
    the time the request spent waiting for an engine lock — this is
    the true queue-wait latency visible to callers.
    """

    def __init__(self, n: int):
        self.n             = n
        self.engines       = [InferenceEngine(i) for i in range(n)]
        self._locks        = []
        self._session_map  = {}
        self._engine_load  = []
        self._map_lock     = None

    async def start(self):
        self._map_lock    = asyncio.Lock()
        self._locks       = [asyncio.Lock() for _ in range(self.n)]
        self._engine_load = [0] * self.n
        await asyncio.gather(*(e.start() for e in self.engines))
        print(f"[pool] {self.n} engine(s) ready  (OMP_NUM_THREADS={OMP_PER_ENGINE} each)")

    async def stop(self):
        await asyncio.gather(*(e.stop() for e in self.engines), return_exceptions=True)

    async def _assign(self, session_id: str) -> int:
        async with self._map_lock:
            if session_id not in self._session_map:
                idx = min(range(self.n), key=lambda i: self._engine_load[i])
                self._session_map[session_id] = idx
                self._engine_load[idx] += 1
            return self._session_map[session_id]

    async def _drop(self, session_id: str):
        async with self._map_lock:
            if session_id in self._session_map:
                idx = self._session_map.pop(session_id)
                self._engine_load[idx] = max(0, self._engine_load[idx] - 1)

    async def generate(
        self, session_id, new_token_ids, max_new, temp, top_k
    ) -> AsyncIterator[dict]:
        idx              = await self._assign(session_id)
        lock_wait_start  = time.perf_counter()

        async with self._locks[idx]:
            # ── Emit scheduling info so callers can record true queue-wait ──
            queue_wait_ms = (time.perf_counter() - lock_wait_start) * 1000
            yield {
                "type":          "scheduled",
                "queue_wait_ms": round(queue_wait_ms, 1),
                "engine_id":     idx,
            }

            async for chunk in self.engines[idx].generate(
                session_id, new_token_ids, max_new, temp, top_k
            ):
                yield chunk

    async def reset_session(self, session_id: str):
        if not self._map_lock:
            return
        async with self._map_lock:
            idx = self._session_map.get(session_id)
        if idx is not None:
            async with self._locks[idx]:
                await self.engines[idx].reset_session(session_id)
        await self._drop(session_id)

    def get_all_pids(self):
        return [e.pid for e in self.engines if e.pid]

    def status(self):
        return [
            {
                "engine_id": i,
                "pid":       self.engines[i].pid,
                "sessions":  self._engine_load[i],
                "busy":      self._locks[i].locked() if self._locks else False,
                "ready":     self.engines[i]._ready,
            }
            for i in range(self.n)
        ]


# ─────────────────────────────────────────────────────────────
# Global singletons
# ─────────────────────────────────────────────────────────────
pool            = EnginePool(N_ENGINES)
latency_tracker = LatencyTracker()


# ─────────────────────────────────────────────────────────────
# Session store
# ─────────────────────────────────────────────────────────────
class SessionData:
    __slots__ = ("system_prompt", "history", "tokens_in_engine", "last_active")

    def __init__(self, system_prompt: str):
        self.system_prompt    = system_prompt
        self.history          = []
        self.tokens_in_engine = 0
        self.last_active      = time.monotonic()

    def touch(self):
        self.last_active = time.monotonic()

    def append_user(self, content: str):
        self.history.append({"role": "user", "content": content})

    def append_assistant(self, content: str):
        self.history.append({"role": "assistant", "content": content})

    def new_turn_tokens(self, user_msg: str) -> list:
        if self.tokens_in_engine == 0:
            text = (
                f"{SYSTEM_TOKEN} {self.system_prompt}{SEP}"
                f"{USER_TOKEN} {user_msg}{SEP}{ASST_TOKEN} "
            )
        else:
            text = f"{USER_TOKEN} {user_msg}{SEP}{ASST_TOKEN} "
        return enc.encode_ordinary(text)

    def rebuild_with_sliding_window(self, user_msg: str, token_budget: int) -> list:
        prefix = enc.encode_ordinary(f"{SYSTEM_TOKEN} {self.system_prompt}{SEP}")
        suffix = enc.encode_ordinary(f"{USER_TOKEN} {user_msg}{SEP}{ASST_TOKEN} ")
        budget = token_budget - len(prefix) - len(suffix)
        selected = []
        for turn in reversed(self.history):
            role  = USER_TOKEN if turn["role"] == "user" else ASST_TOKEN
            chunk = enc.encode_ordinary(f"{role} {turn['content']}{SEP}")
            if len(chunk) > budget:
                break
            selected.insert(0, chunk)
            budget -= len(chunk)
        tokens = prefix[:]
        for chunk in selected:
            tokens += chunk
        tokens += suffix
        return tokens


sessions      = {}
_metrics_lock = None
metrics = {
    "total_requests":   0,
    "total_tokens":     0,
    "total_ms":         0.0,
    "errors":           0,
    "rejected_429":     0,   # ← NEW
    "timed_out":        0,   # ← NEW
    "start_time":       time.time(),
    "sessions_evicted": 0,
}

# Inflight counter (backpressure)
_inflight_count = 0
_inflight_lock  = None


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def get_total_ram_mb() -> float:
    try:
        total = psutil.Process(os.getpid()).memory_info().rss
        for pid in pool.get_all_pids():
            try:
                total += psutil.Process(pid).memory_info().rss
            except psutil.NoSuchProcess:
                pass
        return round(total / 1_000_000, 1)
    except Exception:
        return 0.0


async def session_gc_loop():
    while True:
        await asyncio.sleep(300)
        if not _metrics_lock:
            continue
        now     = time.monotonic()
        expired = [
            sid for sid, s in list(sessions.items())
            if now - s.last_active > SESSION_TTL_S
        ]
        for sid in expired:
            sessions.pop(sid, None)
            await pool.reset_session(sid)
            async with _metrics_lock:
                metrics["sessions_evicted"] += 1
        if expired:
            print(f"[GC] Evicted {len(expired)} idle sessions")


def _download_file(repo_id: str, filename: str, dest_dir: Path) -> None:
    dest = dest_dir / filename
    print(f"[HF] Downloading {filename} from {repo_id}...")
    cached = Path(hf_hub_download(repo_id=repo_id, filename=filename))
    if not cached.exists():
        raise FileNotFoundError(f"hf_hub_download path missing: {cached}")
    shutil.copy2(cached, dest)
    print(f"[HF] {filename} ready ({dest.stat().st_size // 1_000_000} MB)")


async def _startup_background():
    loop = asyncio.get_running_loop()
    for fname in ("model.bin", "tokenizer.bin"):
        dest = BASE_DIR / fname
        if dest.exists():
            print(f"[startup] {fname} present — skip download")
        else:
            try:
                await loop.run_in_executor(None, _download_file, HF_REPO_ID, fname, BASE_DIR)
            except Exception as e:
                print(f"[ERROR] {fname} download failed: {e}")
                return
    print(f"[startup] All files OK — starting {N_ENGINES} engine(s)...")
    try:
        await pool.start()
    except Exception as e:
        print(f"[ERROR] Pool start failed: {e}")


# ─────────────────────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _metrics_lock, _inflight_lock
    _metrics_lock  = asyncio.Lock()
    _inflight_lock = asyncio.Lock()
    asyncio.create_task(_startup_background())
    asyncio.create_task(session_gc_loop())
    yield
    await pool.stop()


app = FastAPI(title="NanoMind", version="2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message:        str
    session_id:     str   = Field(default_factory=lambda: str(uuid.uuid4()))
    system_prompt:  str   = "You are a helpful, thoughtful, and articulate AI assistant."
    max_new_tokens: int   = Field(default=200, ge=1,    le=MAX_GEN_CEILING)
    temperature:    float = Field(default=0.7,  ge=0.01, le=2.0)
    top_k:          int   = Field(default=40,   ge=1,    le=200)


class ResetRequest(BaseModel):
    session_id: str


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────
@app.get("/")
async def serve_ui(request: Request):
    if request.query_params.get("logs") == "container":
        return {"status": "ok"}
    return FileResponse(BASE_DIR / "index.html")


@app.get("/health")
async def health(response: Response):
    mem          = psutil.virtual_memory()
    ready_count  = sum(1 for e in pool.engines if e._ready)
    is_ready     = ready_count > 0
    engines_busy = sum(1 for l in pool._locks if l and l.locked())
    if not is_ready:
        response.status_code = 503
    return {
        "status":              "ok" if is_ready else "starting",
        "engines_ready":       ready_count,
        "engines_total":       N_ENGINES,
        "engines_busy":        engines_busy,             # ← NEW
        "inflight_requests":   _inflight_count,          # ← NEW
        "max_inflight":        MAX_INFLIGHT,             # ← NEW
        "active_sessions":     len(sessions),
        "sessions_evicted":    metrics["sessions_evicted"],
        "process_ram_mb":      get_total_ram_mb(),
        "system_ram_used_pct": mem.percent,
        "uptime_seconds":      round(time.time() - metrics["start_time"], 1),
    }


@app.get("/pool/status")
async def pool_status():
    return {
        "n_engines":      N_ENGINES,
        "omp_per_engine": OMP_PER_ENGINE,
        "max_inflight":   MAX_INFLIGHT,                  # ← NEW
        "inflight":       _inflight_count,               # ← NEW
        "engines":        pool.status(),
        "sessions":       len(sessions),
    }


@app.post("/chat")
async def chat(req: ChatRequest):
    global _inflight_count

    # ── Guard: engine not ready ───────────────────────────────
    if not any(e._ready for e in pool.engines):
        raise HTTPException(503, "Engine loading — retry in ~30s")

    # ── Guard: input too long ─────────────────────────────────
    if len(req.message) > MAX_INPUT_CHARS:
        raise HTTPException(
            400,
            f"Message too long ({len(req.message)} chars, max {MAX_INPUT_CHARS}).",
        )

    # ── Backpressure: reject if server is saturated ───────────
    # This prevents unbounded queue growth and gives clients
    # a clear signal to back off (HTTP 429 + Retry-After header).
    async with _inflight_lock:
        if _inflight_count >= MAX_INFLIGHT:
            async with _metrics_lock:
                metrics["rejected_429"] += 1
            raise HTTPException(
                status_code=429,
                detail={
                    "error":       "overloaded",
                    "inflight":    _inflight_count,
                    "max":         MAX_INFLIGHT,
                    "retry_after": "2",
                    "message":     "Server busy — too many concurrent requests. Retry in ~2s.",
                },
                headers={"Retry-After": "2"},
            )
        _inflight_count += 1

    enqueue_time = time.perf_counter()

    # ── Session setup ─────────────────────────────────────────
    sess = sessions.get(req.session_id)
    if sess is None:
        sess = SessionData(req.system_prompt)
        sessions[req.session_id] = sess
    sess.touch()

    new_tokens = sess.new_turn_tokens(req.message)
    if sess.tokens_in_engine + len(new_tokens) + req.max_new_tokens > MAX_SESSION_TOKENS:
        await pool.reset_session(req.session_id)
        sess.tokens_in_engine = 0
        new_tokens = sess.rebuild_with_sliding_window(
            req.message, MAX_SESSION_TOKENS - req.max_new_tokens
        )
    sess.append_user(req.message)

    async with _metrics_lock:
        metrics["total_requests"] += 1

    # ── Streaming response ────────────────────────────────────
    async def event_stream():
        global _inflight_count

        parts         = []
        t0            = time.perf_counter()
        ttft_ms: Optional[float] = None
        queue_wait_ms = 0.0
        stopped_early = False
        final_chunk   = None
        tok_generated = 0

        try:
            async with asyncio.timeout(REQUEST_TIMEOUT):
                async for chunk in pool.generate(
                    req.session_id, new_tokens,
                    req.max_new_tokens, req.temperature, req.top_k,
                ):
                    ctype = chunk["type"]

                    # ── Internal scheduling event (not forwarded) ─────────
                    if ctype == "scheduled":
                        queue_wait_ms = chunk["queue_wait_ms"]
                        continue

                    # ── Token ─────────────────────────────────────────────
                    elif ctype == "token":
                        tok_generated += 1
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - t0) * 1000
                        if not stopped_early:
                            parts.append(chunk["text"])
                            joined = "".join(parts)
                            for s in STOP_STRINGS:
                                for pat in (f"\n{s}", f"\n {s}"):
                                    idx = joined.find(pat)
                                    if idx != -1:
                                        parts        = [joined[:idx]]
                                        stopped_early = True
                                        break
                                if stopped_early:
                                    break
                            if not stopped_early:
                                yield f"data: {json.dumps(chunk)}\n\n"

                    # ── Done (engine finished) ─────────────────────────────
                    elif ctype == "done":
                        final_chunk   = chunk
                        tok_generated = chunk["total_tokens"]

                    # ── Error from engine ──────────────────────────────────
                    elif ctype == "error":
                        async with _metrics_lock:
                            metrics["errors"] += 1
                        yield f"data: {json.dumps(chunk)}\n\n"

        except TimeoutError:
            async with _metrics_lock:
                metrics["timed_out"] += 1
            timeout_msg = json.dumps({
                "type": "error",
                "message": f"Request timed out after {REQUEST_TIMEOUT:.0f}s"
            })
            yield f"data: {timeout_msg}\n\n"

        except asyncio.CancelledError:
            pass  # client disconnected — normal, just clean up

        finally:
            # ── Build final summary ───────────────────────────────────
            reply   = "".join(parts).strip()
            elapsed = (time.perf_counter() - t0) * 1000

            if final_chunk and not stopped_early:
                tok_out  = final_chunk["total_tokens"]
                total_ms = final_chunk["total_ms"]
                tps      = final_chunk["tps"]
            else:
                tok_out  = tok_generated
                total_ms = round(elapsed, 2)
                tps      = round(tok_out / (total_ms / 1000), 2) if total_ms > 0 else 0.0

            sess.append_assistant(reply)
            sess.tokens_in_engine += len(new_tokens) + tok_generated

            async with _metrics_lock:
                metrics["total_tokens"] += tok_out
                metrics["total_ms"]     += elapsed

            # ── Record latency sample ─────────────────────────────────
            if ttft_ms is not None:
                latency_tracker.record(
                    queue_wait_ms=queue_wait_ms,
                    ttft_ms=ttft_ms,
                    e2e_ms=elapsed,
                    tps=tps,
                )

            # ── Final SSE event (extended with tracing fields) ────────
            done_payload = {
                "type":          "done",
                "total_tokens":  tok_out,
                "total_ms":      total_ms,
                "tps":           tps,
                "queue_wait_ms": round(queue_wait_ms, 1),  # ← NEW
                "ttft_ms":       round(ttft_ms, 1) if ttft_ms else 0,  # ← NEW
                "session_id":    req.session_id,
                "full_response": reply,
            }
            yield f"data: {json.dumps(done_payload)}\n\n"
            yield "data: [DONE]\n\n"

            # ── Release inflight slot ─────────────────────────────────
            async with _inflight_lock:
                _inflight_count = max(0, _inflight_count - 1)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/chat/reset")
async def reset_chat(req: ResetRequest):
    sessions.pop(req.session_id, None)
    await pool.reset_session(req.session_id)
    return {"status": "ok", "session_id": req.session_id}


@app.get("/chat/history")
async def get_history(session_id: str):
    sess = sessions.get(session_id)
    if not sess:
        return {"session_id": session_id, "turns": 0, "history": []}
    return {
        "session_id":        session_id,
        "turns":             sum(1 for m in sess.history if m["role"] == "user"),
        "tokens_in_engine":  sess.tokens_in_engine,
        "last_active_ago_s": round(time.monotonic() - sess.last_active, 1),
        "history":           sess.history,
    }


@app.get("/metrics")
async def get_metrics():
    async with _metrics_lock:
        n       = metrics["total_requests"]
        tok     = metrics["total_tokens"]
        ms      = metrics["total_ms"]
        err     = metrics["errors"]
        rej     = metrics["rejected_429"]
        tmo     = metrics["timed_out"]
        evicted = metrics["sessions_evicted"]
    mem = psutil.virtual_memory()
    lat = latency_tracker.summary()
    return {
        # ── Throughput ────────────────────────────────────────
        "total_requests":         n,
        "total_tokens_generated": tok,
        "avg_tps":                round(tok / (ms / 1000), 2) if ms > 0 else 0,
        # ── Error accounting ──────────────────────────────────
        "total_errors":           err,
        "rejected_429":           rej,
        "timed_out":              tmo,
        "error_rate_pct":         round((err + rej + tmo) / max(n, 1) * 100, 2),
        # ── Queue state ───────────────────────────────────────
        "inflight_requests":      _inflight_count,
        "max_inflight":           MAX_INFLIGHT,
        # ── Sessions ──────────────────────────────────────────
        "active_sessions":        len(sessions),
        "sessions_evicted_total": evicted,
        # ── Engines ───────────────────────────────────────────
        "n_engines":              N_ENGINES,
        "engines_ready":          sum(1 for e in pool.engines if e._ready),
        # ── Memory ────────────────────────────────────────────
        "process_ram_mb":         get_total_ram_mb(),
        "system_ram_used_pct":    mem.percent,
        "system_ram_total_gb":    round(mem.total / 1_000_000_000, 1),
        # ── Latency percentiles (rolling last 500 requests) ───
        "latency":                lat,
        # ── Uptime ────────────────────────────────────────────
        "uptime_s":               round(time.time() - metrics["start_time"], 1),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=False)
