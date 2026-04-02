# ============================================================
# KVInfer SLM v2 — Kaggle Final (Single GPU, Zero Bugs)
# T4 x1 or T4 x2 — uses only GPU 0
# 12 Hours → ~17,000 steps → Loss ~2.6
# ============================================================

# ════════════════════════════════════════════════════════════
# CELL 1 — Install & Imports
# ════════════════════════════════════════════════════════════

# !pip install -q datasets tiktoken

import os, math, time, struct
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from tqdm.auto import tqdm
from datasets import load_dataset
import tiktoken

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

SAVE_DIR  = "/kaggle/working/KVInfer_v2"
BIN_PATH  = "/kaggle/working/train_v2.bin"
CKPT_PATH = os.path.join(SAVE_DIR, "checkpoint.pt")
os.makedirs(SAVE_DIR, exist_ok=True)

# Single GPU only — no DataParallel
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)

enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token

print(f"Device : {device}")
print(f"GPU    : {torch.cuda.get_device_name(0)}")
print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")


# ════════════════════════════════════════════════════════════
# CELL 2 — Dataset Preparation
# ════════════════════════════════════════════════════════════

SYSTEM = "System: You are a helpful, thoughtful, and articulate AI assistant.\n"

def format_hermes(row):
    text = SYSTEM
    for msg in row.get('conversations', []):
        role = "User: " if msg.get('from') in ('human', 'user') else "Assistant: "
        val  = (msg.get('value') or '').strip()
        if val:
            text += f"{role}{val}\n"
    return text if len(text) > 60 else None

def format_alpaca_gpt4(row):
    prompt = (row.get('instruction') or '').strip()
    inp    = (row.get('input')       or '').strip()
    out    = (row.get('output')      or '').strip()
    if not prompt or not out:
        return None
    if inp:
        prompt += "\n" + inp
    return f"{SYSTEM}User: {prompt}\nAssistant: {out}"

def format_wizard(row):
    convs = row.get('conversations', [])
    if not convs:
        return None
    text = SYSTEM
    for msg in convs:
        role = "User: " if msg.get('from') in ('human', 'user') else "Assistant: "
        val  = (msg.get('value') or '').strip()
        if val:
            text += f"{role}{val}\n"
    return text if len(text) > 60 else None

def format_platypus(row):
    prompt = (row.get('instruction') or '').strip()
    out    = (row.get('output')      or '').strip()
    if not prompt or not out:
        return None
    return f"{SYSTEM}User: {prompt}\nAssistant: {out}"

def write_dataset(ds, fh, fmt_fn, name, max_samples=None):
    count, skipped = 0, 0
    items = (list(ds) if max_samples is None
             else list(ds.select(range(min(max_samples, len(ds))))))
    for row in tqdm(items, desc=f"Writing {name}"):
        try:
            text = fmt_fn(row)
        except Exception:
            skipped += 1
            continue
        if not text or len(text.strip()) < 20:
            skipped += 1
            continue
        ids = enc.encode_ordinary(text)
        ids.append(EOT)
        if len(ids) < 10 or len(ids) > 2048:
            skipped += 1
            continue
        fh.write(np.array(ids, dtype=np.uint16).tobytes())
        count += len(ids)
    print(f"  {name}: {count:,} tokens, {skipped} skipped")
    return count

if not os.path.exists(BIN_PATH):
    print("Downloading datasets...")
    total = 0
    with open(BIN_PATH, "wb") as f:
        print("\n[1/4] OpenHermes 2.5...")
        ds = load_dataset("teknium/OpenHermes-2.5", split="train")
        ds = ds.shuffle(seed=42).select(range(500_000))
        total += write_dataset(ds, f, format_hermes, "OpenHermes-500k")

        print("\n[2/4] Alpaca GPT-4...")
        ds = load_dataset("vicgalle/alpaca-gpt4", split="train")
        total += write_dataset(ds, f, format_alpaca_gpt4, "Alpaca-GPT4")

        print("\n[3/4] WizardLM Evol V2...")
        ds = load_dataset("WizardLM/WizardLM_evol_instruct_V2_196k", split="train")
        total += write_dataset(ds, f, format_wizard, "WizardLM-Evol")

        print("\n[4/4] Open-Platypus...")
        ds = load_dataset("garage-bAInd/Open-Platypus", split="train")
        total += write_dataset(ds, f, format_platypus, "Platypus")

    print(f"\n✅ Total: {total:,} tokens | {os.path.getsize(BIN_PATH)/1e9:.2f} GB")
else:
    print(f"✅ train_v2.bin exists ({os.path.getsize(BIN_PATH)/1e9:.2f} GB)")


# ════════════════════════════════════════════════════════════
# CELL 3 — Model Architecture
# ════════════════════════════════════════════════════════════

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias   = nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn        = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj        = nn.Linear(config.n_embd, config.n_embd,     bias=config.bias)
        self.attn_dropout  = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head        = config.n_head
        self.n_embd        = config.n_embd
        self.dropout       = config.dropout
        self.flash         = hasattr(F, 'scaled_dot_product_attention')
    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(
                torch.ones(T, T, device=x.device).tril() == 0, float('-inf')
            )
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y   = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1  = LayerNorm(config.n_embd, config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln2  = LayerNorm(config.n_embd, config.bias)
        self.mlp  = MLP(config)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int   = 1024
    vocab_size: int   = 50304
    n_layer:    int   = 16
    n_head:     int   = 12
    n_embd:     int   = 768
    dropout:    float = 0.0
    bias:       bool  = True

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos  = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x    = self.transformer.drop(
            self.transformer.wte(idx) + self.transformer.wpe(pos)
        )
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss   = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1), ignore_index=-1
            )
            return logits, loss
        return self.lm_head(x[:, [-1], :]), None
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())


# ════════════════════════════════════════════════════════════
# CELL 4 — Model + Optimizer Setup
# ════════════════════════════════════════════════════════════

# Hyperparameters — tuned for single T4 (14.5 GB)
SEQ_LEN    = 512   # fits comfortably
BATCH_SIZE = 12    # safe for T4, no OOM
GRAD_ACCUM = 8     # effective batch = 96
MAX_ITERS  = 17_000
SAVE_EVERY = 2_000
LOG_EVERY  = 100
LR_MAX     = 5e-4
LR_MIN     = 5e-5
WARMUP     = 300

config = GPTConfig()
model  = GPT(config)
model  = model.to(device)

print(f"Parameters : {model.get_num_params()/1e6:.2f}M")
print(f"BATCH_SIZE : {BATCH_SIZE}  (eff. {BATCH_SIZE * GRAD_ACCUM})")
print(f"MAX_ITERS  : {MAX_ITERS}  (~12 hours on T4)")

# Data
data = np.memmap(BIN_PATH, dtype=np.uint16, mode='r')
print(f"Data       : {len(data):,} tokens")

def get_batch():
    ix = torch.randint(len(data) - SEQ_LEN, (BATCH_SIZE,))
    x  = torch.stack([
        torch.from_numpy(data[i   : i+SEQ_LEN  ].astype(np.int64)) for i in ix
    ])
    y  = torch.stack([
        torch.from_numpy(data[i+1 : i+1+SEQ_LEN].astype(np.int64)) for i in ix
    ])
    return x.pin_memory().to(device, non_blocking=True), \
           y.pin_memory().to(device, non_blocking=True)

# Optimizer
decay_params   = [p for n, p in model.named_parameters() if p.dim() >= 2 and p.requires_grad]
nodecay_params = [p for n, p in model.named_parameters() if p.dim() <  2 and p.requires_grad]
optimizer = torch.optim.AdamW(
    [
        {'params': decay_params,   'weight_decay': 0.1},
        {'params': nodecay_params, 'weight_decay': 0.0},
    ],
    lr=LR_MAX, betas=(0.9, 0.95), fused=True
)
scaler = torch.amp.GradScaler('cuda')

def get_lr(step):
    if step < WARMUP:
        return LR_MAX * step / max(WARMUP, 1)
    if step > MAX_ITERS:
        return LR_MIN
    ratio = (step - WARMUP) / (MAX_ITERS - WARMUP)
    return LR_MIN + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (LR_MAX - LR_MIN)

# torch.compile
print("\nCompiling model...")
try:
    model = torch.compile(model)
    raw_model = model._orig_mod
    print("✅ torch.compile success")
except Exception as e:
    raw_model = model
    print(f"⚠️  torch.compile skipped: {e}")

# Resume checkpoint
start_iter = 0
if os.path.exists(CKPT_PATH):
    print(f"\nResuming: {CKPT_PATH}")
    ckpt  = torch.load(CKPT_PATH, map_location=device)
    state = {k.replace('_orig_mod.', ''): v for k, v in ckpt['model'].items()}
    raw_model.load_state_dict(state)
    optimizer.load_state_dict(ckpt['optimizer'])
    scaler.load_state_dict(ckpt['scaler'])
    start_iter = ckpt['iter']
    print(f"Resumed from step {start_iter}")


# ════════════════════════════════════════════════════════════
# CELL 5 — Training Loop
# ════════════════════════════════════════════════════════════

print("\n" + "="*55)
print(f" Training  : {MAX_ITERS} steps | eff. batch {BATCH_SIZE*GRAD_ACCUM}")
print(f" ETA       : ~12 hours on T4 x1")
print("="*55 + "\n")

model.train()
optimizer.zero_grad(set_to_none=True)

t_start    = time.time()
loss_accum = 0.0

for step in range(start_iter, MAX_ITERS):

    # update LR
    lr = get_lr(step)
    for pg in optimizer.param_groups:
        pg['lr'] = lr

    # gradient accumulation
    for micro in range(GRAD_ACCUM):
        X, Y = get_batch()
        with torch.amp.autocast('cuda', dtype=torch.float16):
            _, loss = model(X, Y)
            loss    = loss / GRAD_ACCUM
        scaler.scale(loss).backward()
        loss_accum += loss.item()

    # step
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # log
    if step % LOG_EVERY == 0:
        elapsed   = time.time() - t_start
        done      = step - start_iter + 1
        eta_h     = ((MAX_ITERS - step) * elapsed / max(done, 1)) / 3600
        tok_s     = done * BATCH_SIZE * GRAD_ACCUM * SEQ_LEN / elapsed
        avg_loss  = loss_accum / LOG_EVERY
        print(
            f"Step {step:5d}/{MAX_ITERS} | "
            f"Loss: {avg_loss:.4f} | "
            f"LR: {lr:.2e} | "
            f"ETA: {eta_h:.1f}h | "
            f"Tok/s: {tok_s:,.0f}"
        )
        loss_accum = 0.0

    # checkpoint
    if step > 0 and step % SAVE_EVERY == 0:
        print(f"\n💾 Saving step {step}...")
        torch.save({
            'iter':      step,
            'model':     raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler':    scaler.state_dict(),
            'config':    config,
        }, CKPT_PATH)
        print(f"✅ Saved → {CKPT_PATH}\n")

print("\n✅ Training Complete!")


#in my case in 10000 steps traning loss is less then 1.5 and there is an limit of 12hr strainght gpu usage it will automaticly dissconnect and your whole 12hr work is lost so when 11hr traning is complet and traning loss is less the 1.5 the stop cell 5 and add this cell between cell 5 and cell 6 
# Cell- 5.5
# Current checkpoint se export
print("Loading checkpoint...")
ckpt  = torch.load(CKPT_PATH, map_location=device)
state = {k.replace('_orig_mod.', ''): v for k, v in ckpt['model'].items()}
raw_model.load_state_dict(state)
print(f"Loaded from step {ckpt['iter']}")


# Current checkpoint se export
print("Loading checkpoint...")
ckpt  = torch.load(CKPT_PATH, map_location=device)
state = {k.replace('_orig_mod.', ''): v for k, v in ckpt['model'].items()}
raw_model.load_state_dict(state)
print(f"Loaded from step {ckpt['iter']}")
# ════════════════════════════════════════════════════════════
# CELL 6 — Export model.bin + tokenizer.bin
# ════════════════════════════════════════════════════════════

print("\nExporting...")
raw_model.eval()

def serialize(t, shape_hint=None):
    if t is None:
        return struct.pack(f'{shape_hint}f', *np.zeros(shape_hint, dtype=np.float32))
    d = t.detach().cpu().view(-1).numpy().astype(np.float32)
    return struct.pack(f'{len(d)}f', *d)

MODEL_BIN = os.path.join(SAVE_DIR, "model.bin")
TOK_BIN   = os.path.join(SAVE_DIR, "tokenizer.bin")

with open(MODEL_BIN, "wb") as f:
    f.write(struct.pack('iiiii',
        config.n_layer, config.n_head, config.n_embd,
        config.block_size, config.vocab_size))
    f.write(serialize(raw_model.transformer.wte.weight))
    f.write(serialize(raw_model.transformer.wpe.weight))
    for block in raw_model.transformer.h:
        f.write(serialize(block.ln1.weight))
        f.write(serialize(block.ln1.bias,           shape_hint=config.n_embd))
        f.write(serialize(block.attn.c_attn.weight))
        f.write(serialize(block.attn.c_attn.bias,   shape_hint=3*config.n_embd))
        f.write(serialize(block.attn.c_proj.weight))
        f.write(serialize(block.attn.c_proj.bias,   shape_hint=config.n_embd))
        f.write(serialize(block.ln2.weight))
        f.write(serialize(block.ln2.bias,           shape_hint=config.n_embd))
        f.write(serialize(block.mlp.c_fc.weight))
        f.write(serialize(block.mlp.c_fc.bias,      shape_hint=4*config.n_embd))
        f.write(serialize(block.mlp.c_proj.weight))
        f.write(serialize(block.mlp.c_proj.bias,    shape_hint=config.n_embd))
    f.write(serialize(raw_model.transformer.ln_f.weight))
    f.write(serialize(raw_model.transformer.ln_f.bias, shape_hint=config.n_embd))
    f.write(serialize(raw_model.lm_head.weight))

print(f"✅ model.bin     : {os.path.getsize(MODEL_BIN)/1e6:.1f} MB")

with open(TOK_BIN, "wb") as f:
    f.write(struct.pack('i', config.vocab_size))
    for i in range(config.vocab_size):
        try:    b = enc.decode_bytes([i])
        except: b = b"<ERR>"
        f.write(struct.pack('i', len(b)))
        f.write(b)

print(f"✅ tokenizer.bin : {os.path.getsize(TOK_BIN)/1e3:.1f} KB")
print(f"\n🎉 Done! → {SAVE_DIR}")


# ════════════════════════════════════════════════════════════
# CELL 7 — Quality Test
# ════════════════════════════════════════════════════════════

@torch.no_grad()
def chat(prompt, max_tokens=200, temperature=0.7, top_k=40):
    raw_model.eval()
    text = f"{SYSTEM}User: {prompt}\nAssistant:"
    ids  = enc.encode_ordinary(text)
    ctx  = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    out  = []
    for _ in range(max_tokens):
        crop      = ctx[:, -config.block_size:]
        logits, _ = raw_model(crop)
        logits    = logits[:, -1, :] / temperature
        v, _      = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float('-inf')
        next_t    = torch.multinomial(F.softmax(logits, dim=-1), 1)
        if next_t.item() == EOT:
            break
        out.append(next_t.item())
        ctx = torch.cat([ctx, next_t], dim=1)
        decoded = enc.decode(out)
        if "\nUser:" in decoded or "\nSystem:" in decoded:
            print(decoded.split("\nUser:")[0].split("\nSystem:")[0].strip())
            return
    print(enc.decode(out).strip())

print("="*55)
for p in [
    "What is the capital of Japan?",
    "Explain machine learning in simple terms.",
    "Write a Python function to reverse a string.",
    "What are 3 benefits of regular exercise?",
]:
    print(f"\nUser : {p}")
    print("Bot  : ", end="", flush=True)
    chat(p)
