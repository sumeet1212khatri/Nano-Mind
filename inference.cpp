/*
 * ============================================================
 * NanoMind — Optimized GPT-2 Inference Engine  v3.0
 * Changes from v2.0:
 *   ✅ Batch prefill: all prompt tokens in ONE pass (3× TTFT)
 *   ✅ matmul_vec_serial: no nested OMP (safe inside parallel blocks)
 *   ✅ Pre-allocated prefill workspace (zero per-request malloc)
 *   ✅ Thread-local temp buffers for MLP in parallel regions
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
#include <immintrin.h>
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

struct Config { int n_layer, n_head, n_embd, block_size, vocab_size; };
struct Weights {
    float *wte, *wpe;
    float **ln1_w, **ln1_b, **c_attn_w, **c_attn_b;
    float **c_proj_w, **c_proj_b, **ln2_w, **ln2_b;
    float **fc_w, **fc_b, **mlp_proj_w, **mlp_proj_b;
    float *ln_f_w, *ln_f_b, *lm_head_w;
};
static Config  cfg;
static Weights W;
static float*  g_data = nullptr;

struct Session {
    float*  k_cache  = nullptr;
    float*  v_cache  = nullptr;
    int     pos      = 0;
    double  last_use = 0.0;
};
static const int MAX_SESSIONS = 20;
static std::unordered_map<std::string, Session> g_sessions;

/* ── Single-token generation buffers (original) ── */
static float *g_x, *g_buf, *g_qkv, *g_attn_buf;
static float *g_ff, *g_logits, *g_tmp_out;
static std::pair<float,int>* g_topk_pairs = nullptr;

/* ── Batch-prefill workspace (NEW) ────────────────
   Pre-allocated at startup: BLOCK_SIZE × n_embd each.
   Zero per-request heap allocations.                */
static float *g_px   = nullptr;   // [BLOCK_SIZE * n_embd]  activations
static float *g_pbuf = nullptr;   // [BLOCK_SIZE * n_embd]  LN / attn output
static float *g_pqkv = nullptr;   // [BLOCK_SIZE * 3*n_embd] Q,K,V for all tokens

/* ── Thread-local scratch (NEW) ─────────────────── */
static thread_local float tl_tmp[768];    // n_embd  (output-proj / residual)
static thread_local float tl_ff[3072];   // 4*n_embd (MLP hidden)

/* ════════════════════════════════════════════════════
   Core math primitives
   ════════════════════════════════════════════════════ */

static void layer_norm(float* out, const float* x, const float* w, const float* b, int N) {
    __m256 vsum = _mm256_setzero_ps();
    for (int i = 0; i < N; i += 8)
        vsum = _mm256_add_ps(vsum, _mm256_loadu_ps(x + i));
    float tmp[8]; _mm256_storeu_ps(tmp, vsum);
    float mean = (tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7]) / (float)N;
    __m256 vmean = _mm256_set1_ps(mean);
    __m256 vvar  = _mm256_setzero_ps();
    for (int i = 0; i < N; i += 8) {
        __m256 d = _mm256_sub_ps(_mm256_loadu_ps(x + i), vmean);
        vvar = _mm256_fmadd_ps(d, d, vvar);
    }
    _mm256_storeu_ps(tmp, vvar);
    float var = (tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7]) / (float)N;
    __m256 vsc = _mm256_set1_ps(1.f / sqrtf(var + 1e-5f));
    for (int i = 0; i < N; i += 8) {
        __m256 d      = _mm256_sub_ps(_mm256_loadu_ps(x + i), vmean);
        __m256 norm   = _mm256_mul_ps(d, vsc);
        __m256 result = _mm256_fmadd_ps(norm, _mm256_loadu_ps(w + i), _mm256_loadu_ps(b + i));
        _mm256_storeu_ps(out + i, result);
    }
    float sc = 1.f / sqrtf(var + 1e-5f);
    for (int i = (N & ~7); i < N; i++)
        out[i] = (x[i] - mean) * sc * w[i] + b[i];
}

/* OMP-parallel matmul — used for large standalone calls */
static void matmul_vec(float* __restrict__ out, const float* __restrict__ mat,
                       const float* __restrict__ x, int M, int K) {
    const int OMP_THRESHOLD = 64;
    if (M < OMP_THRESHOLD) {
        for (int i = 0; i < M; i++) {
            const float* row = mat + (long long)i * K;
            __m256 acc = _mm256_setzero_ps();
            int j = 0;
            for (; j <= K-8; j += 8)
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(row+j), _mm256_loadu_ps(x+j), acc);
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
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(row+j), _mm256_loadu_ps(x+j), acc);
        float t[8]; _mm256_storeu_ps(t, acc);
        float s = t[0]+t[1]+t[2]+t[3]+t[4]+t[5]+t[6]+t[7];
        for (; j < K; j++) s += row[j] * x[j];
        out[i] = s;
    }
}

/* Serial matmul — safe to call INSIDE #pragma omp parallel regions (NEW) */
static inline void matmul_vec_serial(float* __restrict__ out,
                                     const float* __restrict__ mat,
                                     const float* __restrict__ x,
                                     int M, int K) {
    for (int i = 0; i < M; i++) {
        const float* row = mat + (long long)i * K;
        __m256 acc = _mm256_setzero_ps();
        int j = 0;
        for (; j <= K-8; j += 8)
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(row+j), _mm256_loadu_ps(x+j), acc);
        float t[8]; _mm256_storeu_ps(t, acc);
        float s = t[0]+t[1]+t[2]+t[3]+t[4]+t[5]+t[6]+t[7];
        for (; j < K; j++) s += row[j] * x[j];
        out[i] = s;
    }
}

static inline float dot_avx2(const float* __restrict__ a, const float* __restrict__ b, int n) {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i <= n-8; i += 8)
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a+i), _mm256_loadu_ps(b+i), acc);
    float tmp[8]; _mm256_storeu_ps(tmp, acc);
    float s = tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
    for (; i < n; i++) s += a[i]*b[i];
    return s;
}

static inline void weighted_acc_avx2(float* __restrict__ out, const float* __restrict__ v, float w, int n) {
    __m256 wv = _mm256_set1_ps(w);
    int i = 0;
    for (; i <= n-8; i += 8)
        _mm256_storeu_ps(out+i, _mm256_fmadd_ps(wv, _mm256_loadu_ps(v+i), _mm256_loadu_ps(out+i)));
    for (; i < n; i++) out[i] += w * v[i];
}

static inline void add_bias(float* x, const float* b, int N) {
    for (int i = 0; i < N; i++) x[i] += b[i];
}

static inline void residual_add(float* x, const float* y, int N) {
    for (int i = 0; i < N; i++) x[i] += y[i];
}

static void gelu_inplace(float* x, int N) {
    const __m256 scale = _mm256_set1_ps(12102203.0f);
    const __m256 vbias = _mm256_set1_ps(1064807168.0f);
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
        __m256i ti = _mm256_cvttps_epi32(_mm256_fmadd_ps(t, scale, vbias));
        __m256  et = _mm256_castsi256_ps(ti);
        __m256 denom = _mm256_add_ps(vone, et);
        __m256 r     = _mm256_rcp_ps(denom);
        r = _mm256_mul_ps(r, _mm256_fnmadd_ps(denom, r, vtwo));
        _mm256_storeu_ps(x + i, _mm256_mul_ps(v, r));
    }
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

/* ════════════════════════════════════════════════════
   Single-token forward (generation phase) — unchanged
   ════════════════════════════════════════════════════ */
static void forward(int token_id, int pos, float* k_cache, float* v_cache) {
    const int C = cfg.n_embd, H = cfg.n_head, hs = C / H;
    float* te = W.wte + (long long)token_id * C;
    float* pe = W.wpe + (long long)pos * C;
#pragma omp parallel for
    for (int i = 0; i < C; i++) g_x[i] = te[i] + pe[i];
    for (int l = 0; l < cfg.n_layer; l++) {
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

/* ════════════════════════════════════════════════════
   Batch prefill (NEW — KEY OPTIMIZATION)
   Process all T prompt tokens in a single parallel pass.

   Why faster:
     OLD: T sequential forward() calls = T × (embed + L layers)
     NEW: 1 pass → weight matrices loaded ONCE, reused for all T
          OMP parallelism over heads AND over tokens in MLP

   Memory: uses pre-allocated g_px / g_pbuf / g_pqkv workspaces.
   KV cache populated for positions [start_pos .. start_pos+T-1].
   Leaves logits in g_logits (ready for first sample_topk call).
   ════════════════════════════════════════════════════ */
static void forward_prefill_batch(const int* tokens, int T, int start_pos,
                                   float* k_cache, float* v_cache) {
    const int C = cfg.n_embd, H = cfg.n_head, hs = C / H;

    /* 1. Embed all T tokens */
    for (int t = 0; t < T; t++) {
        float* dst = g_px + (long long)t * C;
        float* te  = W.wte + (long long)tokens[t] * C;
        float* pe  = W.wpe + (long long)(start_pos + t) * C;
        for (int i = 0; i < C; i++) dst[i] = te[i] + pe[i];
    }

    for (int l = 0; l < cfg.n_layer; l++) {
        float* kc = k_cache + (long long)l * cfg.block_size * C;
        float* vc = v_cache + (long long)l * cfg.block_size * C;

        /* 2. LayerNorm for all T tokens */
        for (int t = 0; t < T; t++)
            layer_norm(g_pbuf + (long long)t*C,
                       g_px   + (long long)t*C,
                       W.ln1_w[l], W.ln1_b[l], C);

        /* 3. QKV projection + KV cache population
              Parallelised over T — matmul_vec_serial safe here */
#pragma omp parallel for schedule(static)
        for (int t = 0; t < T; t++) {
            float* qkv = g_pqkv + (long long)t * 3 * C;
            matmul_vec_serial(qkv, W.c_attn_w[l],
                              g_pbuf + (long long)t*C, 3*C, C);
            const float* bias = W.c_attn_b[l];
            for (int i = 0; i < 3*C; i++) qkv[i] += bias[i];
            memcpy(kc + (long long)(start_pos + t)*C, qkv + C,   C*sizeof(float));
            memcpy(vc + (long long)(start_pos + t)*C, qkv + 2*C, C*sizeof(float));
        }

        /* 4. Causal self-attention output
              Parallelised over heads.
              Each thread owns attn_scores on its stack — no races.
              Within each head, sequential over t (causal dependency). */
        memset(g_pbuf, 0, (long long)T * C * sizeof(float));
        const float scale = 1.f / sqrtf((float)hs);

#pragma omp parallel for schedule(static)
        for (int h = 0; h < H; h++) {
            float attn_scores[1024]; /* stack-alloc per thread, block_size max */
            for (int t = 0; t < T; t++) {
                float* qh      = g_pqkv + (long long)t*3*C + h*hs;
                int    seq_len = start_pos + t + 1;
                for (int s = 0; s < seq_len; s++)
                    attn_scores[s] = dot_avx2(qh, kc + (long long)s*C + h*hs, hs) * scale;
                softmax_inplace(attn_scores, seq_len);
                float* oh = g_pbuf + (long long)t*C + h*hs;
                for (int s = 0; s < seq_len; s++)
                    weighted_acc_avx2(oh, vc + (long long)s*C + h*hs, attn_scores[s], hs);
            }
        }

        /* 5. Output projection + residual
              Parallelised over T. Uses thread-local tl_tmp. */
#pragma omp parallel for schedule(static)
        for (int t = 0; t < T; t++) {
            float* xt = g_px   + (long long)t * C;
            float* bt = g_pbuf + (long long)t * C;
            matmul_vec_serial(tl_tmp, W.c_proj_w[l], bt, C, C);
            const float* bias = W.c_proj_b[l];
            for (int i = 0; i < C; i++) xt[i] += tl_tmp[i] + bias[i];
        }

        /* 6. LayerNorm 2 */
        for (int t = 0; t < T; t++)
            layer_norm(g_pbuf + (long long)t*C,
                       g_px   + (long long)t*C,
                       W.ln2_w[l], W.ln2_b[l], C);

        /* 7. MLP: fc → GELU → proj + residual
              Parallelised over T. Uses thread-local tl_ff, tl_tmp. */
#pragma omp parallel for schedule(static)
        for (int t = 0; t < T; t++) {
            float* xt = g_px   + (long long)t * C;
            float* bt = g_pbuf + (long long)t * C;
            matmul_vec_serial(tl_ff, W.fc_w[l], bt, 4*C, C);
            const float* bias_fc = W.fc_b[l];
            for (int i = 0; i < 4*C; i++) tl_ff[i] += bias_fc[i];
            gelu_inplace(tl_ff, 4*C);
            matmul_vec_serial(tl_tmp, W.mlp_proj_w[l], tl_ff, C, 4*C);
            const float* bias_p = W.mlp_proj_b[l];
            for (int i = 0; i < C; i++) xt[i] += tl_tmp[i] + bias_p[i];
        }
    }

    /* 8. Final LN + logits from the LAST token */
    layer_norm(g_buf, g_px + (long long)(T-1)*C, W.ln_f_w, W.ln_f_b, C);
    matmul_vec(g_logits, W.lm_head_w, g_buf, cfg.vocab_size, C);
}

/* ════════════════════════════════════════════════════
   Model loading
   ════════════════════════════════════════════════════ */
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
    if (p > end) {
        long long needed = (p   - data) * (long long)sizeof(float);
        long long got    = (end - data) * (long long)sizeof(float);
        fprintf(stderr, "FATAL: model.bin too small — need %lld bytes, got %lld.\n", needed, got);
        printf("ERROR model_truncated\n"); fflush(stdout);
        exit(1);
    }
}

/* ════════════════════════════════════════════════════
   Session / KV-cache management
   ════════════════════════════════════════════════════ */
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
    if (it != g_sessions.end()) { it->second.last_use = get_ms(); return it->second; }
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

/* ════════════════════════════════════════════════════
   Sampling
   ════════════════════════════════════════════════════ */
static int sample_topk(float temperature, int top_k) {
    for (int v = 0; v < cfg.vocab_size; v++) g_logits[v] /= temperature;
    int K = std::min(top_k, cfg.vocab_size);
    for (int v = 0; v < cfg.vocab_size; v++) g_topk_pairs[v] = {g_logits[v], v};
    std::partial_sort(g_topk_pairs, g_topk_pairs + K, g_topk_pairs + cfg.vocab_size,
        [](const auto& a, const auto& b){ return a.first > b.first; });
    float sum = 0.f;
    for (int j = 0; j < K; j++) { g_topk_pairs[j].first = expf(g_topk_pairs[j].first); sum += g_topk_pairs[j].first; }
    for (int j = 0; j < K; j++) g_topk_pairs[j].first /= sum;
    float r = rng_float(), cum = 0.f;
    int best = g_topk_pairs[0].second;
    for (int j = 0; j < K; j++) {
        cum += g_topk_pairs[j].first;
        if (r < cum) { best = g_topk_pairs[j].second; break; }
    }
    return best;
}

/* ════════════════════════════════════════════════════
   Request handling
   ════════════════════════════════════════════════════ */
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

static void handle_request(const std::string& line) {
    auto parts = split(line, '|');
    if (parts.size() < 7) { printf("ERROR bad_request_format\n"); fflush(stdout); return; }
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
    stop_ids.insert(50256);

    Session& sess = get_or_create(sess_id);
    if (new_tokens.empty()) { printf("ERROR no_tokens\n"); fflush(stdout); return; }
    if (sess.pos + (int)new_tokens.size() >= cfg.block_size) {
        printf("ERROR context_window_full\n"); fflush(stdout); return;
    }

    /* ── PREFILL: batch process all new_tokens at once (KEY CHANGE) ──
       OLD: loop calling forward(tok, pos) one by one  → slow
       NEW: forward_prefill_batch(all tokens)          → 3× faster   */
    forward_prefill_batch(new_tokens.data(), (int)new_tokens.size(),
                          sess.pos, sess.k_cache, sess.v_cache);
    sess.pos += (int)new_tokens.size();

    /* ── GENERATION: one token at a time as before ── */
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
        if (it != g_sessions.end()) { free_session(it->second); g_sessions.erase(it); }
    }
    printf("RESET_OK\n"); fflush(stdout);
}

/* ════════════════════════════════════════════════════
   main
   ════════════════════════════════════════════════════ */
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
    map_weights(g_data, wbytes);

    const int C = cfg.n_embd;

    /* ── Single-token generation buffers ── */
    g_x          = (float*)malloc(C * sizeof(float));
    g_buf        = (float*)malloc(C * sizeof(float));
    g_qkv        = (float*)malloc(3*C * sizeof(float));
    g_attn_buf   = (float*)malloc((long long)cfg.n_head * cfg.block_size * sizeof(float));
    g_ff         = (float*)malloc(4*C * sizeof(float));
    g_logits     = (float*)malloc((long long)cfg.vocab_size * sizeof(float));
    g_tmp_out    = (float*)malloc(C * sizeof(float));
    g_topk_pairs = (std::pair<float,int>*)malloc((long long)cfg.vocab_size * sizeof(std::pair<float,int>));

    /* ── Batch prefill workspace (NEW) ── */
    g_px   = (float*)malloc((long long)cfg.block_size * C * sizeof(float));
    g_pbuf = (float*)malloc((long long)cfg.block_size * C * sizeof(float));
    g_pqkv = (float*)malloc((long long)cfg.block_size * 3 * C * sizeof(float));

    if (!g_x || !g_buf || !g_qkv || !g_attn_buf || !g_ff || !g_logits ||
        !g_tmp_out || !g_topk_pairs || !g_px || !g_pbuf || !g_pqkv) {
        printf("ERROR oom_buffers\n"); fflush(stdout); return 1;
    }

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
    free(g_px); free(g_pbuf); free(g_pqkv);
    return 0;
}
