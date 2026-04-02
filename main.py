"""
NanoMind — FastAPI Backend v1.5
152M GPT-2 · AVX2 + OpenMP · Persistent KV-Cache

Changes over v1.4:
  1. MAX_GEN_CEILING lowered to 256 (was 500) — MAX_SESSION_TOKENS raised
     to 744, giving ~50% more history before a context reset triggers.
  2. MAX_INPUT_CHARS = 2000 guard added — rejects absurdly long messages
     before they hit the tokenizer and blow the context window.
  3. SessionData.rebuild_with_sliding_window() — on context overflow, keeps
     as many recent conversation turns as fit rather than discarding all
     history. Users no longer experience sudden amnesia mid-conversation.
  4. CRITICAL: tokens_in_engine is now updated in event_stream's finally
     block, not after the async-for loop. Previously a client disconnect
     (CancelledError) skipped the update entirely, leaving a stale count
     that caused premature or missed context resets on subsequent turns.
  5. tokens_generated is incremented on every TOKEN chunk so the finally
     block has an accurate partial count even on mid-stream disconnects.
  6. STOP_STRINGS matching now also catches patterns without a leading
     space (e.g. "\\nUser:" and "\\n User:"), reducing role-bleed leakage.

Fixes carried forward from v1.4:
  7. _download_file no longer passes local_dir — downloads to HF cache
     and copies manually to avoid cross-device rename failures.
  8. Byte-level UTF-8 buffer in InferenceEngine.generate for correct
     multi-byte character decoding (no more ◆ replacement chars).
"""
import asyncio
import json
import os
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import psutil
import tiktoken
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from huggingface_hub import hf_hub_download

# ─────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
INFERENCE_EXE = BASE_DIR / "inference"
MODEL_BIN     = BASE_DIR / "model.bin"
TOKENIZER_BIN = BASE_DIR / "tokenizer.bin"
HF_REPO_ID    = os.environ.get("HF_REPO_ID", "NOT-OMEGA/NanoMind")

SYSTEM_TOKEN = "System:"
USER_TOKEN   = "User:"
ASST_TOKEN   = "Assistant:"
SEP          = "\n"

BLOCK_SIZE      = 1024
MAX_GEN_CEILING = 256   # lowered from 500 — frees 244 extra history tokens
SAFETY_MARGIN   = 24
# 1024 - 256 - 24 = 744  (was 500 with old ceiling)
MAX_SESSION_TOKENS = BLOCK_SIZE - MAX_GEN_CEILING - SAFETY_MARGIN

# Hard cap on raw input characters — rejects giant pastes before tokenising
MAX_INPUT_CHARS = 2000

N_ENGINES      = int(os.environ.get("N_ENGINES",      "1"))
OMP_PER_ENGINE = int(os.environ.get("OMP_NUM_THREADS", "2"))
SESSION_TTL_S  = int(os.environ.get("SESSION_TTL",    "1800"))

enc            = tiktoken.get_encoding("gpt2")
STOP_TOKEN_IDS = [50256]
STOP_STRINGS   = ["User:", "System:"]

# ─────────────────────────────────────────────────────────────────────────
# Inference Engine
# ─────────────────────────────────────────────────────────────────────────
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

        print(f"[engine-{self.eid}] launching process…")
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
                        stderr_out = await self._proc.stderr.read()
                        raise RuntimeError(
                            f"[engine-{self.eid}] process exited before READY. "
                            f"stderr: {stderr_out.decode(errors='replace')[:500]}"
                        )
                    line = raw.decode().strip()
                    if line:
                        print(f"[engine-{self.eid}] startup: {line}")
                    if line == "READY":
                        self._ready = True
                        print(f"[engine-{self.eid}] READY  "
                              f"pid={self._proc.pid}  "
                              f"OMP_NUM_THREADS={OMP_PER_ENGINE}")
                        break
                    if line.startswith("ERROR"):
                        raise RuntimeError(
                            f"[engine-{self.eid}] startup error: {line}"
                        )
        except TimeoutError:
            self._proc.kill()
            raise RuntimeError(
                f"[engine-{self.eid}] timed out (120s) waiting for READY"
            )

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
        if not self._ready or self._proc is None:
            return
        self._proc.stdin.write(f"RESET|{session_id}\n".encode())
        await self._proc.stdin.drain()
        try:
            async with asyncio.timeout(5):
                while True:
                    raw = await self._proc.stdout.readline()
                    if not raw:
                        break
                    if raw.decode().strip() == "RESET_OK":
                        break
        except TimeoutError:
            pass

    async def generate(self, session_id, new_token_ids, max_new, temperature, top_k):
        if not self._ready or self._proc is None:
            yield {"type": "error", "message": f"Engine-{self.eid} not ready"}
            return

        tokens_csv = ",".join(map(str, new_token_ids))
        stop_csv   = ",".join(map(str, STOP_TOKEN_IDS))
        cmd = (
            f"REQUEST|{session_id}|{tokens_csv}|"
            f"{max_new}|{temperature}|{top_k}|{stop_csv}\n"
        )
        self._proc.stdin.write(cmd.encode())
        await self._proc.stdin.drain()

        # Byte-level buffer for correct multi-byte UTF-8 decoding.
        # GPT-2 BPE tokens can be partial UTF-8 byte sequences — decoding
        # each token individually produces replacement chars for non-ASCII.
        byte_buffer = bytearray()

        try:
            while True:
                raw = await self._proc.stdout.readline()
                if not raw:
                    self._ready = False
                    yield {"type": "error",
                           "message": "Engine process died unexpectedly"}
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
                        decoded_text = byte_buffer.decode("utf-8")
                        byte_buffer.clear()
                    except UnicodeDecodeError:
                        continue
                    if decoded_text:
                        yield {
                            "type":       "token",
                            "id":         token_id,
                            "text":       decoded_text,
                            "elapsed_ms": elapsed,
                        }

                elif line.startswith("DONE"):
                    if byte_buffer:
                        leftover = byte_buffer.decode("utf-8", errors="replace")
                        byte_buffer.clear()
                        if leftover:
                            yield {
                                "type":       "token",
                                "id":         -1,
                                "text":       leftover,
                                "elapsed_ms": 0.0,
                            }
                    parts    = line.split()
                    total_t  = int(parts[1])
                    total_ms = float(parts[2])
                    tps = round(total_t / (total_ms / 1000.0), 2) \
                          if total_ms > 0 else 0
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


# ─────────────────────────────────────────────────────────────────────────
# Engine Pool
# ─────────────────────────────────────────────────────────────────────────
class EnginePool:
    def __init__(self, n: int):
        self.n             = n
        self.engines       = [InferenceEngine(i) for i in range(n)]
        self._locks:        list[asyncio.Lock] = []
        self._session_map:  dict[str, int]     = {}
        self._engine_load:  list[int]          = []
        self._map_lock:     asyncio.Lock | None = None

    async def start(self):
        self._map_lock    = asyncio.Lock()
        self._locks       = [asyncio.Lock() for _ in range(self.n)]
        self._engine_load = [0] * self.n
        await asyncio.gather(*(e.start() for e in self.engines))
        print(f"[pool] {self.n} engine(s) ready  "
              f"(OMP_NUM_THREADS={OMP_PER_ENGINE} each)")

    async def stop(self):
        await asyncio.gather(
            *(e.stop() for e in self.engines), return_exceptions=True
        )

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

    async def generate(self, session_id, new_token_ids, max_new, temp, top_k):
        idx = await self._assign(session_id)
        async with self._locks[idx]:
            async for chunk in self.engines[idx].generate(
                session_id, new_token_ids, max_new, temp, top_k
            ):
                yield chunk

    async def reset_session(self, session_id: str):
        if self._map_lock is None:
            return
        async with self._map_lock:
            idx = self._session_map.get(session_id)
        if idx is not None:
            async with self._locks[idx]:
                await self.engines[idx].reset_session(session_id)
        await self._drop(session_id)

    def get_all_pids(self) -> list:
        return [e.pid for e in self.engines if e.pid]

    def status(self) -> list:
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


pool = EnginePool(N_ENGINES)

# ─────────────────────────────────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────────────────────────────────
class SessionData:
    __slots__ = ("system_prompt", "history", "tokens_in_engine", "last_active")

    def __init__(self, system_prompt: str):
        self.system_prompt    = system_prompt
        self.history: list    = []
        self.tokens_in_engine = 0
        self.last_active      = time.monotonic()

    def touch(self):
        self.last_active = time.monotonic()

    def append_user(self, content: str):
        self.history.append({"role": "user", "content": content})

    def append_assistant(self, content: str):
        self.history.append({"role": "assistant", "content": content})

    def new_turn_tokens(self, user_msg: str) -> list[int]:
        """Tokens for a normal (non-overflow) turn."""
        if self.tokens_in_engine == 0:
            text = (
                f"{SYSTEM_TOKEN} {self.system_prompt}{SEP}"
                f"{USER_TOKEN} {user_msg}{SEP}{ASST_TOKEN} "
            )
        else:
            text = f"{USER_TOKEN} {user_msg}{SEP}{ASST_TOKEN} "
        return enc.encode_ordinary(text)

    def rebuild_with_sliding_window(self, user_msg: str, token_budget: int) -> list[int]:
        """
        Rebuild a full prompt that fits within token_budget.
        Called after a context overflow reset.
        Packs: system prompt + as many recent history turns as fit
               (most-recent-first selection) + current user message.
        This replaces the v1.4 behaviour of discarding all history on reset,
        which caused sudden conversation amnesia for multi-turn sessions.
        """
        prefix = enc.encode_ordinary(f"{SYSTEM_TOKEN} {self.system_prompt}{SEP}")
        suffix = enc.encode_ordinary(f"{USER_TOKEN} {user_msg}{SEP}{ASST_TOKEN} ")
        budget = token_budget - len(prefix) - len(suffix)

        selected: list[list[int]] = []
        for turn in reversed(self.history):
            role  = USER_TOKEN if turn["role"] == "user" else ASST_TOKEN
            chunk = enc.encode_ordinary(f"{role} {turn['content']}{SEP}")
            if len(chunk) > budget:
                break
            selected.insert(0, chunk)
            budget -= len(chunk)

        tokens: list[int] = prefix[:]
        for chunk in selected:
            tokens += chunk
        tokens += suffix
        return tokens


sessions: dict[str, SessionData] = {}
_metrics_lock: asyncio.Lock | None = None
metrics = {
    "total_requests":   0,
    "total_tokens":     0,
    "total_ms":         0.0,
    "errors":           0,
    "start_time":       time.time(),
    "sessions_evicted": 0,
}


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


# ─────────────────────────────────────────────────────────────────────────
# Background Tasks
# ─────────────────────────────────────────────────────────────────────────
async def session_gc_loop():
    while True:
        await asyncio.sleep(300)
        if _metrics_lock is None:
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
    """Blocking HF download — runs in a thread executor.
    Downloads to HF cache (same filesystem), then shutil.copy2() to dest.
    Avoids the cross-device os.rename() failure from passing local_dir
    directly to hf_hub_download in newer huggingface_hub versions.
    """
    dest = dest_dir / filename
    print(f"[HF HUB] Downloading {filename} from {repo_id} …")
    cached = Path(hf_hub_download(repo_id=repo_id, filename=filename))
    if not cached.exists():
        raise FileNotFoundError(
            f"hf_hub_download returned a path that does not exist: {cached}"
        )
    print(f"[HF HUB] Copying {filename} → {dest}")
    shutil.copy2(cached, dest)
    if not dest.exists():
        raise FileNotFoundError(
            f"Copy appeared to succeed but {dest} is missing. "
            "Possible causes: disk full, /app not writable."
        )
    size_mb = dest.stat().st_size // 1_000_000
    print(f"[HF HUB] {filename} ready ({size_mb} MB)")


async def _startup_background():
    loop = asyncio.get_running_loop()

    if MODEL_BIN.exists():
        print(f"[startup] model.bin present "
              f"({MODEL_BIN.stat().st_size // 1_000_000} MB) — skip download")
    else:
        try:
            await loop.run_in_executor(
                None, _download_file, HF_REPO_ID, "model.bin", BASE_DIR
            )
        except Exception as e:
            print(f"[ERROR] model.bin download failed: {e}")
            return

    if TOKENIZER_BIN.exists():
        print("[startup] tokenizer.bin present — skip download")
    else:
        try:
            await loop.run_in_executor(
                None, _download_file, HF_REPO_ID, "tokenizer.bin", BASE_DIR
            )
        except Exception as e:
            print(f"[ERROR] tokenizer.bin download failed: {e}")
            return

    if not MODEL_BIN.exists():
        print("[ERROR] model.bin missing after download — aborting pool start")
        return
    if not TOKENIZER_BIN.exists():
        print("[ERROR] tokenizer.bin missing after download — aborting pool start")
        return

    print(f"[startup] All files OK — starting {N_ENGINES} engine(s)…")
    try:
        await pool.start()
    except Exception as e:
        print(f"[ERROR] Pool start failed: {e}")


# ─────────────────────────────────────────────────────────────────────────
# App Lifecycle
# ─────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _metrics_lock
    _metrics_lock = asyncio.Lock()
    asyncio.create_task(_startup_background())
    asyncio.create_task(session_gc_loop())
    yield
    await pool.stop()


app = FastAPI(title="NanoMind", version="1.5", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────
# Request Models
# ─────────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message:        str
    session_id:     str   = Field(default_factory=lambda: str(uuid.uuid4()))
    system_prompt:  str   = "You are a helpful, thoughtful, and articulate AI assistant."
    max_new_tokens: int   = Field(default=200, ge=1,    le=MAX_GEN_CEILING)
    temperature:    float = Field(default=0.7,  ge=0.01, le=2.0)
    top_k:          int   = Field(default=40,   ge=1,    le=200)


class ResetRequest(BaseModel):
    session_id: str


# ─────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────
@app.get("/")
async def serve_ui():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/health")
async def health(response: Response):
    mem         = psutil.virtual_memory()
    ready_count = sum(1 for e in pool.engines if e._ready)
    is_ready    = ready_count > 0
    if not is_ready:
        response.status_code = 503
    return {
        "status":                 "ok" if is_ready else "starting",
        "engines_ready":          ready_count,
        "engines_total":          N_ENGINES,
        "omp_threads_per_engine": OMP_PER_ENGINE,
        "speed_mode":             N_ENGINES == 1,
        "active_sessions":        len(sessions),
        "sessions_evicted":       metrics["sessions_evicted"],
        "process_ram_mb":         get_total_ram_mb(),
        "system_ram_used_pct":    mem.percent,
        "uptime_seconds":         round(time.time() - metrics["start_time"], 1),
    }


@app.get("/pool/status")
async def pool_status():
    return {
        "n_engines":      N_ENGINES,
        "omp_per_engine": OMP_PER_ENGINE,
        "engines":        pool.status(),
        "sessions":       len(sessions),
    }


@app.post("/chat")
async def chat(req: ChatRequest):
    if not any(e._ready for e in pool.engines):
        raise HTTPException(503, "Engine loading… retry in ~30s")

    # ── Input length guard ────────────────────────────────────────────────
    if len(req.message) > MAX_INPUT_CHARS:
        raise HTTPException(
            400,
            f"Message too long ({len(req.message)} chars). "
            f"Maximum is {MAX_INPUT_CHARS} characters."
        )

    sess = sessions.get(req.session_id)
    if sess is None:
        sess = SessionData(req.system_prompt)
        sessions[req.session_id] = sess

    sess.touch()
    new_tokens = sess.new_turn_tokens(req.message)

    # ── Context window overflow — sliding window rebuild ──────────────────
    # v1.4: hard reset discarded all history (conversation amnesia).
    # v1.5: rebuild keeps as many recent turns as fit in the budget.
    if sess.tokens_in_engine + len(new_tokens) + req.max_new_tokens \
            > MAX_SESSION_TOKENS:
        await pool.reset_session(req.session_id)
        sess.tokens_in_engine = 0
        budget     = MAX_SESSION_TOKENS - req.max_new_tokens
        new_tokens = sess.rebuild_with_sliding_window(req.message, budget)

    sess.append_user(req.message)
    async with _metrics_lock:
        metrics["total_requests"] += 1

    async def event_stream():
        parts:            list[str] = []
        t0              = time.perf_counter()
        stopped_early   = False
        final_chunk     = None
        tokens_generated = 0   # tracked throughout — used in finally even on disconnect

        try:
            async for chunk in pool.generate(
                req.session_id, new_tokens,
                req.max_new_tokens, req.temperature, req.top_k,
            ):
                if chunk["type"] == "token":
                    # Count every token regardless of stopped_early so the
                    # finally block has an accurate partial count on disconnect.
                    tokens_generated += 1
                    if not stopped_early:
                        parts.append(chunk["text"])
                        joined = "".join(parts)
                        for s in STOP_STRINGS:
                            # Match both "\nUser:" and "\n User:" (no leading space)
                            for pattern in (f"\n{s}", f"\n {s}"):
                                idx = joined.find(pattern)
                                if idx != -1:
                                    parts         = [joined[:idx]]
                                    stopped_early = True
                                    break
                            if stopped_early:
                                break
                        if not stopped_early:
                            yield f"data: {json.dumps(chunk)}\n\n"
                    # stopped_early: silently consume remaining tokens until DONE

                elif chunk["type"] == "done":
                    final_chunk      = chunk
                    tokens_generated = chunk["total_tokens"]  # authoritative count

                elif chunk["type"] == "error":
                    async with _metrics_lock:
                        metrics["errors"] += 1
                    yield f"data: {json.dumps(chunk)}\n\n"

            # ── Normal completion path ────────────────────────────────────
            reply   = "".join(parts).strip()
            elapsed = (time.perf_counter() - t0) * 1000

            if final_chunk is not None and not stopped_early:
                tok_out  = final_chunk["total_tokens"]
                total_ms = final_chunk["total_ms"]
                tps      = final_chunk["tps"]
            else:
                tok_out  = len(enc.encode_ordinary(reply)) if reply else 0
                total_ms = round(elapsed, 2)
                tps      = round(tok_out / (total_ms / 1000.0), 2) \
                           if total_ms > 0 else 0
                tokens_generated = tok_out

            sess.append_assistant(reply)
            async with _metrics_lock:
                metrics["total_tokens"] += tok_out
                metrics["total_ms"]     += elapsed

            done_payload = json.dumps({
                "type":          "done",
                "total_tokens":  tok_out,
                "total_ms":      total_ms,
                "tps":           tps,
                "session_id":    req.session_id,
                "full_response": reply,
            })
            yield f"data: {done_payload}\n\n"

        except Exception as e:
            async with _metrics_lock:
                metrics["errors"] += 1
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

        finally:
            # Always update token count — runs even on CancelledError (client disconnect).
            # v1.4 updated tokens_in_engine after the async-for loop, which was
            # skipped entirely on disconnect, leaving a permanently stale count.
            sess.tokens_in_engine += len(new_tokens) + tokens_generated
            yield "data: [DONE]\n\n"

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
        evicted = metrics["sessions_evicted"]
    mem = psutil.virtual_memory()
    return {
        "total_requests":         n,
        "total_tokens":           tok,
        "total_errors":           err,
        "avg_tps":                round(tok / (ms / 1000), 2) if ms > 0 else 0,
        "active_sessions":        len(sessions),
        "sessions_evicted_total": evicted,
        "n_engines":              N_ENGINES,
        "omp_per_engine":         OMP_PER_ENGINE,
        "engines_ready":          sum(1 for e in pool.engines if e._ready),
        "engines_busy":           sum(1 for lk in pool._locks if lk.locked())
                                  if pool._locks else 0,
        "process_ram_mb":         get_total_ram_mb(),
        "system_ram_used_pct":    mem.percent,
        "system_ram_total_gb":    round(mem.total / 1_000_000_000, 1),
        "uptime_s":               round(time.time() - metrics["start_time"], 1),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=False)
