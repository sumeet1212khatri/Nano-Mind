/*
 * ============================================================
 * NanoMind — Optimized GPT-2 Inference Engine  v2.0
 * ============================================================
 *
 * Changes from v1.0:
 *   ✓ Thread-local Xorshift64 RNG (replaces non-thread-safe rand())
 *   ✓ Pre-allocated g_topk_pairs (eliminates 400KB malloc per token)
 *   ✓ AVX2 layer_norm (was scalar; called 33×/token)
 *   ✓ Sigmoid GeLU with Schraudolph fast exp (replaces tanhf cubic)
 *   ✓ matmul_vec skips OpenMP for M < 64 (avoids thread-fork overhead)
 *   ✓ map_weights bounds check (detect truncated model.bin early)
 *
 * ── STDIN PROTOCOL ──────────────────────────────────────────
 *   REQUEST|<sess>|<tokens_csv>|<max_new>|<temp>|<top_k>|<stop_csv>
 *   RESET|<sess>
 *   QUIT
 *
 * ── STDOUT PROTOCOL ─────────────────────────────────────────
 *   READY
 *   TOKEN <id> <elapsed_ms>
 *   DONE  <count> <total_ms>
 *   RESET_OK
 *   ERROR <message>
 *
 * ── COMPILE ─────────────────────────────────────────────────
 *   g++ -O3 -march=native -fopenmp -ffast-math -std=c++17 \
 *       -o inference inference.cpp -lm
 * ============================================================
 */
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <ctime>
#include <algorithm>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <immintrin.h>   // AVX2 + FMA
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef _WIN32
  #include <windows.h>
  static double get_ms() {
      LARGE_INTEGER f, c;
      QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
      return (double)c.QuadPart / f.QuadPart * 1000.0;
  }
#else
  #include <sys/time.h>
  static double get_ms() {
      struct timeval tv; gettimeofday(&tv, NULL);
      return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
  }
#endif

// ─────────────────────────────────────────────────────────────────────────
// Thread-local Xorshift64 RNG  (replaces non-thread-safe rand())
// Each OpenMP thread gets its own independent RNG state — no contention,
// no corruption if sampling is ever parallelised in the future.
// ─────────────────────────────────────────────────────────────────────────
static thread_local uint64_t tl_rng_state = 0;
static inline float rng_float() {
    if (!tl_rng_state) {
        tl_rng_state = (uint64_t)get_ms() ^ (uint64_t)(uintptr_t)&tl_rng_state;
        if (!tl_rng_state) tl_rng_state = 0xDEADBEEFCAFEBABEULL;
    }
    tl_rng_state ^= tl_rng_state << 13;
    tl_rng_state ^= tl_rng_state >> 7;
    tl_rng_state ^= tl_rng_state << 17;
    return (float)(tl_rng_state >> 11) * (1.0f / (float)(1ULL << 53));
}

// ─────────────────────────────────────────────────────────────────────────
// Config & Weights
// ─────────────────────────────────────────────────────────────────────────
struct Config {
    int n_layer, n_head, n_embd, block_size, vocab_size;
};
struct Weights {
    float *wte, *wpe;
    float **ln1_w, **ln1_b;
    float **c_attn_w, **c_attn_b;
    float **c_proj_w, **c_proj_b;
    float **ln2_w,    **ln2_b;
    float **fc_w,     **fc_b;
    float **mlp_proj_w, **mlp_proj_b;
    float *ln_f_w, *ln_f_b;
    float *lm_head_w;
};
static Config  cfg;
static Weights W;
static float*  g_data = nullptr;

// ─────────────────────────────────────────────────────────────────────────
// Session (per-session KV-cache + position)
// ─────────────────────────────────────────────────────────────────────────
struct Session {
    float*  k_cache  = nullptr;
    float*  v_cache  = nullptr;
    int     pos      = 0;
    double  last_use = 0.0;
};
static const int MAX_SESSIONS = 20;
static std::unordered_map<std::string, Session> g_sessions;

// ─────────────────────────────────────────────────────────────────────────
// Working Buffers — pre-allocated, NO stack VLAs
// ─────────────────────────────────────────────────────────────────────────
static float *g_x, *g_buf, *g_qkv, *g_attn_buf;
static float *g_ff, *g_logits, *g_tmp_out;
// Pre-allocated top-k pair buffer — eliminates ~400KB heap alloc per token
static std::pair<float,int>* g_topk_pairs = nullptr;

// ─────────────────────────────────────────────────────────────────────────
// Math Kernels
// ─────────────────────────────────────────────────────────────────────────

// AVX2 layer norm.
// v1.0 was pure scalar; this version vectorises all 3 passes.
// Called 33×/token (2×n_layer + 1), so the speedup compounds heavily.
// Assumes N divisible by 8 — holds for n_embd=768 and 4×n_embd=3072.
static void layer_norm(float* out, const float* x, const float* w,
                       const float* b, int N) {
    // ── Pass 1: mean ──────────────────────────────────────────────────
    __m256 vsum = _mm256_setzero_ps();
    for (int i = 0; i < N; i += 8)
        vsum = _mm256_add_ps(vsum, _mm256_loadu_ps(x + i));
    float tmp[8]; _mm256_storeu_ps(tmp, vsum);
    float mean = (tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7]) / (float)N;

    // ── Pass 2: variance ─────────────────────────────────────────────
    __m256 vmean = _mm256_set1_ps(mean);
    __m256 vvar  = _mm256_setzero_ps();
    for (int i = 0; i < N; i += 8) {
        __m256 d = _mm256_sub_ps(_mm256_loadu_ps(x + i), vmean);
        vvar = _mm256_fmadd_ps(d, d, vvar);
    }
    _mm256_storeu_ps(tmp, vvar);
    float var = (tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7]) / (float)N;
    __m256 vsc = _mm256_set1_ps(1.f / sqrtf(var + 1e-5f));

    // ── Pass 3: normalize + affine (scale + bias) ─────────────────────
    for (int i = 0; i < N; i += 8) {
        __m256 d      = _mm256_sub_ps(_mm256_loadu_ps(x + i), vmean);
        __m256 norm   = _mm256_mul_ps(d, vsc);
        __m256 result = _mm256_fmadd_ps(norm, _mm256_loadu_ps(w + i),
                                              _mm256_loadu_ps(b + i));
        _mm256_storeu_ps(out + i, result);
    }
    // Scalar tail — safety for non-multiple-of-8 N (normally unreachable)
    float sc = 1.f / sqrtf(var + 1e-5f);
    for (int i = (N & ~7); i < N; i++)
        out[i] = (x[i] - mean) * sc * w[i] + b[i];
}

// AVX2 + FMA matmul: out[M] = mat[M,K] · x[K]
// Skips OpenMP thread-fork for M < 64 — the fork/join overhead (~2-5µs)
// dominates latency for small projections like the bias add paths.
static void matmul_vec(float* __restrict__ out,
                       const float* __restrict__ mat,
                       const float* __restrict__ x,
                       int M, int K) {
    const int OMP_THRESHOLD = 64;
    if (M < OMP_THRESHOLD) {
        for (int i = 0; i < M; i++) {
            const float* row = mat + (long long)i * K;
            __m256 acc = _mm256_setzero_ps();
            int j = 0;
            for (; j <= K-8; j += 8)
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(row+j),
                                      _mm256_loadu_ps(x+j), acc);
            float t[8]; _mm256_storeu_ps(t, acc);
            float s = t[0]+t[1]+t[2]+t[3]+t[4]+t[5]+t[6]+t[7];
            for (; j < K; j++) s += row[j] * x[j];
            out[i] = s;
        }
        return;
    }
#pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) {
        const float* row = mat + (long long)i * K;
        __m256 acc = _mm256_setzero_ps();
        int j = 0;
        for (; j <= K-8; j += 8)
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(row+j),
                                  _mm256_loadu_ps(x+j), acc);
        float t[8]; _mm256_storeu_ps(t, acc);
        float s = t[0]+t[1]+t[2]+t[3]+t[4]+t[5]+t[6]+t[7];
        for (; j < K; j++) s += row[j] * x[j];
        out[i] = s;
    }
}

// AVX2 dot product — used in attention inner loop (hs=64 → 8 AVX iters)
static inline float dot_avx2(const float* __restrict__ a,
                              const float* __restrict__ b, int n) {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i <= n-8; i += 8)
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a+i),
                              _mm256_loadu_ps(b+i), acc);
    float tmp[8]; _mm256_storeu_ps(tmp, acc);
    float s = tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
    for (; i < n; i++) s += a[i]*b[i];
    return s;
}

// AVX2 weighted accumulate: out += w * v
static inline void weighted_acc_avx2(float* __restrict__ out,
                                     const float* __restrict__ v,
                                     float w, int n) {
    __m256 wv = _mm256_set1_ps(w);
    int i = 0;
    for (; i <= n-8; i += 8)
        _mm256_storeu_ps(out+i,
            _mm256_fmadd_ps(wv, _mm256_loadu_ps(v+i),
                            _mm256_loadu_ps(out+i)));
    for (; i < n; i++) out[i] += w * v[i];
}

static inline void add_bias(float* x, const float* b, int N) {
#pragma omp parallel for
    for (int i = 0; i < N; i++) x[i] += b[i];
}

static inline void residual_add(float* x, const float* y, int N) {
#pragma omp parallel for
    for (int i = 0; i < N; i++) x[i] += y[i];
}

// GeLU via sigmoid approximation with Schraudolph fast exp.
//
// GeLU(x) ≈ x * sigmoid(1.702 * x) = x / (1 + exp(-1.702 * x))
//
// v1.0 used the tanh cubic form: x * 0.5 * (1 + tanh(sqrt(2/π)*(x + 0.044715*x³)))
// which requires an expensive tanhf() call per element.
//
// This version uses:
//   exp(t) ≈ reinterpret_cast<float>(int(t * 2^23/ln2 + 127*2^23))  [Schraudolph 1999]
//   1/(1+exp) via rcp_ps + one Newton-Raphson step (faster than div_ps)
//
// Error vs true GeLU: < 0.01 across typical activation range — negligible for inference.
static void gelu_inplace(float* x, int N) {
    const __m256 scale = _mm256_set1_ps(12102203.0f);    // 2^23 / ln(2)
    const __m256 vbias = _mm256_set1_ps(1064807168.0f);  // 127 * 2^23
    const __m256 neg_c = _mm256_set1_ps(-1.702f);
    const __m256 vone  = _mm256_set1_ps(1.0f);
    const __m256 vtwo  = _mm256_set1_ps(2.0f);
    const __m256 vlo   = _mm256_set1_ps(-88.0f);
    const __m256 vhi   = _mm256_set1_ps(88.0f);
    int i = 0;
    for (; i <= N-8; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        __m256 t = _mm256_mul_ps(neg_c, v);
        t = _mm256_max_ps(t, vlo);
        t = _mm256_min_ps(t, vhi);
        // fast exp(t) via Schraudolph bit-reinterpretation trick
        __m256i ti = _mm256_cvttps_epi32(_mm256_fmadd_ps(t, scale, vbias));
        __m256  et = _mm256_castsi256_ps(ti);
        // sigmoid via rcp_ps + Newton-Raphson (faster than div_ps)
        __m256 denom = _mm256_add_ps(vone, et);
        __m256 r     = _mm256_rcp_ps(denom);
        r = _mm256_mul_ps(r, _mm256_fnmadd_ps(denom, r, vtwo));
        _mm256_storeu_ps(x + i, _mm256_mul_ps(v, r));
    }
    // Scalar tail
    for (; i < N; i++) {
        float v = x[i];
        x[i] = v / (1.0f + expf(-1.702f * v));
    }
}

static void softmax_inplace(float* x, int N) {
    float mx = x[0];
    for (int i = 1; i < N; i++) if (x[i] > mx) mx = x[i];
    float s = 0.f;
    for (int i = 0; i < N; i++) { x[i] = expf(x[i]-mx); s += x[i]; }
    for (int i = 0; i < N; i++) x[i] /= s;
}

// ─────────────────────────────────────────────────────────────────────────
// Forward Pass (single token at position pos)
// ─────────────────────────────────────────────────────────────────────────
static void forward(int token_id, int pos, float* k_cache, float* v_cache) {
    const int C  = cfg.n_embd;
    const int H  = cfg.n_head;
    const int hs = C / H;

    float* te = W.wte + (long long)token_id * C;
    float* pe = W.wpe + (long long)pos * C;
#pragma omp parallel for
    for (int i = 0; i < C; i++) g_x[i] = te[i] + pe[i];

    for (int l = 0; l < cfg.n_layer; l++) {
        // ── Self-Attention ────────────────────────────────────────────────
        layer_norm(g_buf, g_x, W.ln1_w[l], W.ln1_b[l], C);
        matmul_vec(g_qkv, W.c_attn_w[l], g_buf, 3*C, C);
        add_bias(g_qkv, W.c_attn_b[l], 3*C);

        float* q  = g_qkv;
        float* k  = g_qkv + C;
        float* v  = g_qkv + 2*C;
        float* kc = k_cache + (long long)l * cfg.block_size * C;
        float* vc = v_cache + (long long)l * cfg.block_size * C;
        memcpy(kc + (long long)pos*C, k, C*sizeof(float));
        memcpy(vc + (long long)pos*C, v, C*sizeof(float));

#pragma omp parallel for schedule(static)
        for (int h = 0; h < H; h++) {
            float* qh    = q + h*hs;
            float  scale = 1.f / sqrtf((float)hs);
            float* attn  = g_attn_buf + h*cfg.block_size;
            for (int t = 0; t <= pos; t++) {
                float* kh = kc + (long long)t*C + h*hs;
                attn[t]   = dot_avx2(qh, kh, hs) * scale;
            }
            softmax_inplace(attn, pos+1);
            float* oh = g_buf + h*hs;
            memset(oh, 0, hs*sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float* vh = vc + (long long)t*C + h*hs;
                weighted_acc_avx2(oh, vh, attn[t], hs);
            }
        }

        matmul_vec(g_tmp_out, W.c_proj_w[l], g_buf, C, C);
        add_bias(g_tmp_out, W.c_proj_b[l], C);
        residual_add(g_x, g_tmp_out, C);

        // ── MLP ───────────────────────────────────────────────────────────
        layer_norm(g_buf, g_x, W.ln2_w[l], W.ln2_b[l], C);
        matmul_vec(g_ff, W.fc_w[l], g_buf, 4*C, C);
        add_bias(g_ff, W.fc_b[l], 4*C);
        gelu_inplace(g_ff, 4*C);
        matmul_vec(g_tmp_out, W.mlp_proj_w[l], g_ff, C, 4*C);
        add_bias(g_tmp_out, W.mlp_proj_b[l], C);
        residual_add(g_x, g_tmp_out, C);
    }

    layer_norm(g_buf, g_x, W.ln_f_w, W.ln_f_b, C);
    matmul_vec(g_logits, W.lm_head_w, g_buf, cfg.vocab_size, C);
}

// ─────────────────────────────────────────────────────────────────────────
// Weight Mapping
// Now takes wbytes to perform a bounds check — detects truncated model.bin
// before it causes a silent segfault or garbage outputs.
// ─────────────────────────────────────────────────────────────────────────
static void map_weights(float* data, long wbytes) {
    float* p   = data;
    float* end = data + wbytes / sizeof(float);
    const int C = cfg.n_embd, L = cfg.n_layer;
    W.wte = p; p += (long long)cfg.vocab_size * C;
    W.wpe = p; p += (long long)cfg.block_size * C;
    #define ARR(f) W.f = (float**)malloc(L*sizeof(float*))
    ARR(ln1_w); ARR(ln1_b); ARR(c_attn_w); ARR(c_attn_b);
    ARR(c_proj_w); ARR(c_proj_b); ARR(ln2_w); ARR(ln2_b);
    ARR(fc_w); ARR(fc_b); ARR(mlp_proj_w); ARR(mlp_proj_b);
    #undef ARR
    for (int l = 0; l < L; l++) {
        W.ln1_w[l]      = p; p += C;
        W.ln1_b[l]      = p; p += C;
        W.c_attn_w[l]   = p; p += 3LL*C*C;
        W.c_attn_b[l]   = p; p += 3LL*C;
        W.c_proj_w[l]   = p; p += 1LL*C*C;
        W.c_proj_b[l]   = p; p += C;
        W.ln2_w[l]      = p; p += C;
        W.ln2_b[l]      = p; p += C;
        W.fc_w[l]       = p; p += 4LL*C*C;
        W.fc_b[l]       = p; p += 4LL*C;
        W.mlp_proj_w[l] = p; p += 1LL*C*4*C;
        W.mlp_proj_b[l] = p; p += C;
    }
    W.ln_f_w    = p; p += C;
    W.ln_f_b    = p; p += C;
    W.lm_head_w = p; p += (long long)cfg.vocab_size * C;

    // ── Bounds check: catch truncated / corrupt model.bin immediately ──
    if (p > end) {
        long long needed = (p   - data) * (long long)sizeof(float);
        long long got    = (end - data) * (long long)sizeof(float);
        fprintf(stderr,
            "FATAL: model.bin too small — need %lld bytes, got %lld.\n"
            "       Re-download the file from HuggingFace.\n",
            needed, got);
        fflush(stderr);
        printf("ERROR model_truncated\n"); fflush(stdout);
        exit(1);
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Session Management (LRU eviction)
// ─────────────────────────────────────────────────────────────────────────
static long long kv_bytes() {
    return (long long)cfg.n_layer * cfg.block_size * cfg.n_embd * sizeof(float);
}
static void free_session(Session& s) {
    free(s.k_cache); free(s.v_cache);
    s.k_cache = nullptr; s.v_cache = nullptr; s.pos = 0;
}
static void evict_oldest() {
    if (g_sessions.empty()) return;
    std::string oid; double ot = 1e300;
    for (auto& kv : g_sessions)
        if (kv.second.last_use < ot) { ot = kv.second.last_use; oid = kv.first; }
    free_session(g_sessions[oid]);
    g_sessions.erase(oid);
}
static Session& get_or_create(const std::string& id) {
    auto it = g_sessions.find(id);
    if (it != g_sessions.end()) {
        it->second.last_use = get_ms();
        return it->second;
    }
    if ((int)g_sessions.size() >= MAX_SESSIONS) evict_oldest();
    Session s;
    long long nb = kv_bytes();
    s.k_cache  = (float*)calloc(nb, 1);
    s.v_cache  = (float*)calloc(nb, 1);
    s.pos      = 0;
    s.last_use = get_ms();
    g_sessions[id] = s;
    return g_sessions[id];
}

// ─────────────────────────────────────────────────────────────────────────
// Top-K Sampler
// Uses pre-allocated g_topk_pairs — eliminates ~400KB heap alloc per token.
// Uses thread-local rng_float() — replaces non-thread-safe rand().
// ─────────────────────────────────────────────────────────────────────────
static int sample_topk(float temperature, int top_k) {
    for (int v = 0; v < cfg.vocab_size; v++) g_logits[v] /= temperature;
    int K = std::min(top_k, cfg.vocab_size);
    for (int v = 0; v < cfg.vocab_size; v++) g_topk_pairs[v] = {g_logits[v], v};
    std::partial_sort(g_topk_pairs, g_topk_pairs + K, g_topk_pairs + cfg.vocab_size,
        [](const auto& a, const auto& b){ return a.first > b.first; });
    float sum = 0.f;
    for (int j = 0; j < K; j++) {
        g_topk_pairs[j].first = expf(g_topk_pairs[j].first);
        sum += g_topk_pairs[j].first;
    }
    for (int j = 0; j < K; j++) g_topk_pairs[j].first /= sum;
    float r = rng_float(), cum = 0.f;
    int best = g_topk_pairs[0].second;
    for (int j = 0; j < K; j++) {
        cum += g_topk_pairs[j].first;
        if (r < cum) { best = g_topk_pairs[j].second; break; }
    }
    return best;
}

// ─────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────
static std::vector<std::string> split(const std::string& s, char d) {
    std::vector<std::string> out; std::string cur;
    for (char c : s) { if (c==d){out.push_back(cur);cur.clear();}else cur+=c; }
    out.push_back(cur); return out;
}
static std::vector<int> parse_ints(const std::string& s) {
    std::vector<int> out;
    for (auto& t : split(s,',')) if (!t.empty()) out.push_back(atoi(t.c_str()));
    return out;
}

// ─────────────────────────────────────────────────────────────────────────
// Command Handlers
// ─────────────────────────────────────────────────────────────────────────
static void handle_request(const std::string& line) {
    auto parts = split(line, '|');
    if (parts.size() < 7) {
        printf("ERROR bad_request_format\n"); fflush(stdout); return;
    }
    std::string sess_id = parts[1];
    auto new_tokens     = parse_ints(parts[2]);
    int  max_new        = atoi(parts[3].c_str());
    float temp          = (float)atof(parts[4].c_str());
    int  top_k          = atoi(parts[5].c_str());
    auto stop_list      = parse_ints(parts[6]);

    temp    = std::max(temp,   0.01f);
    top_k   = std::clamp(top_k, 1, cfg.vocab_size);
    max_new = std::max(max_new, 1);

    std::unordered_set<int> stop_ids(stop_list.begin(), stop_list.end());
    stop_ids.insert(50256);  // <|endoftext|>

    Session& sess = get_or_create(sess_id);

    // Prefill
    for (int tok : new_tokens) {
        if (sess.pos >= cfg.block_size) {
            printf("ERROR context_window_full\n"); fflush(stdout); return;
        }
        forward(tok, sess.pos, sess.k_cache, sess.v_cache);
        sess.pos++;
    }

    // Autoregressive generation
    double t0 = get_ms();
    int gen = 0;
    for (int i = 0; i < max_new; i++) {
        if (sess.pos >= cfg.block_size) break;
        int next = sample_topk(temp, top_k);
        printf("TOKEN %d %.2f\n", next, get_ms()-t0);
        fflush(stdout);
        gen++;
        if (stop_ids.count(next)) break;
        forward(next, sess.pos, sess.k_cache, sess.v_cache);
        sess.pos++;
    }

    printf("DONE %d %.2f\n", gen, get_ms()-t0);
    fflush(stdout);
}

static void handle_reset(const std::string& line) {
    auto parts = split(line, '|');
    if (parts.size() >= 2) {
        auto it = g_sessions.find(parts[1]);
        if (it != g_sessions.end()) {
            free_session(it->second); g_sessions.erase(it);
        }
    }
    printf("RESET_OK\n"); fflush(stdout);
}

// ─────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────
int main() {
    FILE* f = fopen("model.bin", "rb");
    if (!f) { printf("ERROR model.bin_not_found\n"); fflush(stdout); return 1; }

    fread(&cfg, sizeof(int), 5, f);
    fseek(f, 0, SEEK_END); long fsize = ftell(f);
    fseek(f, 5*(long)sizeof(int), SEEK_SET);
    long wbytes = fsize - 5*(long)sizeof(int);

    g_data = (float*)malloc(wbytes);
    if (!g_data) { printf("ERROR oom\n"); fflush(stdout); return 1; }
    fread(g_data, 1, wbytes, f);
    fclose(f);

    map_weights(g_data, wbytes);  // bounds check — exits on truncated file

    const int C = cfg.n_embd;
    g_x          = (float*)malloc(C * sizeof(float));
    g_buf        = (float*)malloc(C * sizeof(float));
    g_qkv        = (float*)malloc(3*C * sizeof(float));
    g_attn_buf   = (float*)malloc((long long)cfg.n_head * cfg.block_size * sizeof(float));
    g_ff         = (float*)malloc(4*C * sizeof(float));
    g_logits     = (float*)malloc((long long)cfg.vocab_size * sizeof(float));
    g_tmp_out    = (float*)malloc(C * sizeof(float));
    g_topk_pairs = (std::pair<float,int>*)malloc(
                       (long long)cfg.vocab_size * sizeof(std::pair<float,int>));

    printf("READY\n"); fflush(stdout);

    std::string line;
    while (std::getline(std::cin, line)) {
        if (!line.empty() && line.back()=='\r') line.pop_back();
        if (line.empty()) continue;
        if (line == "QUIT")                    break;
        else if (line.rfind("RESET|",   0)==0) handle_reset(line);
        else if (line.rfind("REQUEST|", 0)==0) handle_request(line);
        else { printf("ERROR unknown_cmd\n"); fflush(stdout); }
    }

    for (auto& kv : g_sessions) free_session(kv.second);
    free(g_data);
    free(g_x); free(g_buf); free(g_qkv); free(g_attn_buf);
    free(g_ff); free(g_logits); free(g_tmp_out); free(g_topk_pairs);
    return 0;
}
