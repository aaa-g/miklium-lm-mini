import os
import re
import json
import time
import random
import numpy as np

np.random.seed(0)
random.seed(0)

# ── resume hyperparams ────────────────────────────────────────────────────────
additional_steps  = 5000
batch_size        = 4
learning_rate     = 0.002    # lower lr for continued training
beta1, beta2, eps = 0.9, 0.95, 1e-8

# ── load checkpoint ───────────────────────────────────────────────────────────
CHECKPOINT = "website/miklium-lm-mini_1.7M.miklium_model"
print(f"Loading checkpoint: {CHECKPOINT} ...")
with open(CHECKPOINT, "r") as f:
    saved = json.load(f)

uchars     = saved["vocab"]
n_layer    = saved["n_layer"]
n_embd     = saved["n_embd"]
block_size = saved["block_size"]
n_head     = saved["n_head"]
head_dim   = n_embd // n_head

params = {k: np.array(v) for k, v in saved["params"].items()}
grads  = {k: np.zeros_like(v) for k, v in params.items()}

vocab_size   = len(uchars) + 1
BOS          = len(uchars)
token_to_idx = {t: i for i, t in enumerate(uchars)}
idx_to_token = {i: t for i, t in enumerate(uchars)}
idx_to_token[BOS] = saved["idx_to_token"][str(BOS)]   # load from saved, avoid literal

# ── load training data ────────────────────────────────────────────────────────
with open("assets/training_data/data.txt", "r") as f:
    text = f.read()

def get_tokens(t):
    return re.findall(r"<[^>]+>|[\w']+|[^\w\s\n]|\s+", t)

all_tokens_raw = []
for sec in text.split("<eos>"):
    if not sec.strip():
        continue
    all_tokens_raw.append(BOS)
    sec_content = sec.strip() + " <eos>"
    all_tokens_raw.extend(
        [token_to_idx[t] for t in get_tokens(sec_content) if t in token_to_idx]
    )
all_tokens = np.array(all_tokens_raw)

total_params = sum(p.size for p in params.values())
print(f"Resumed miklium-lm-mini ({total_params/1e6:.2f}M params) — "
      f"{additional_steps} more steps at lr={learning_rate}")

# ── model functions ───────────────────────────────────────────────────────────
def rmsnorm_fwd(x):
    ms = np.mean(x ** 2, axis=-1, keepdims=True)
    scale = (ms + 1e-5) ** -0.5
    return x * scale, ms, scale

def rmsnorm_bwd(dy, x, ms, scale):
    n = x.shape[-1]
    dscale = np.sum(dy * x, axis=-1, keepdims=True)
    return dy * scale - x * scale ** 3 * dscale / n

def forward(tokens):
    T = len(tokens)
    x = params["wte"][tokens] + params["wpe"][np.arange(T)]
    cache = []
    for i in range(n_layer):
        x_norm1, ms1, sc1 = rmsnorm_fwd(x)
        q = x_norm1 @ params[f"layer{i}.attn_wq"].T
        k = x_norm1 @ params[f"layer{i}.attn_wk"].T
        v = x_norm1 @ params[f"layer{i}.attn_wv"].T
        q = q.reshape(T, n_head, head_dim).transpose(1, 0, 2)
        k = k.reshape(T, n_head, head_dim).transpose(1, 0, 2)
        v = v.reshape(T, n_head, head_dim).transpose(1, 0, 2)
        att = q @ k.transpose(0, 2, 1) / head_dim ** 0.5
        mask = np.triu(np.ones((T, T)), k=1) * -1e10
        att = att + mask
        exp_att = np.exp(att - np.max(att, axis=-1, keepdims=True))
        probs = exp_att / (np.sum(exp_att, axis=-1, keepdims=True) + 1e-10)
        y_attn = (probs @ v).transpose(1, 0, 2).reshape(T, n_embd)
        y_out  = y_attn @ params[f"layer{i}.attn_wo"].T
        x = x + y_out
        x_norm2, ms2, sc2 = rmsnorm_fwd(x)
        h      = x_norm2 @ params[f"layer{i}.mlp_fc1"].T
        h_relu = np.maximum(0, h)
        y_mlp  = h_relu @ params[f"layer{i}.mlp_fc2"].T
        x = x + y_mlp
        cache.append((x_norm1, ms1, sc1, q, k, v, probs, y_attn, x_norm2, ms2, sc2, h, h_relu))
    logits = x @ params["lm_head"].T
    return logits, cache, x

def backward(tokens, logits, cache, x_final):
    T       = len(tokens)
    targets = tokens[1:]
    sl      = logits[:-1] - np.max(logits[:-1], axis=-1, keepdims=True)
    exps    = np.exp(sl)
    probs   = exps / (np.sum(exps, axis=-1, keepdims=True) + 1e-10)
    loss    = -np.mean(np.log(probs[np.arange(T - 1), targets] + 1e-10))
    dlogits = probs.copy()
    dlogits[np.arange(T - 1), targets] -= 1
    dlogits /= T - 1
    grads["lm_head"] += dlogits.T @ x_final[:-1]
    dx      = dlogits @ params["lm_head"]
    dx_full = np.zeros((T, n_embd))
    dx_full[:-1] = dx
    dx = dx_full
    for i in reversed(range(n_layer)):
        x_norm1, ms1, sc1, q, k, v, p, y_attn_h, x_norm2, ms2, sc2, h, h_relu = cache[i]
        grads[f"layer{i}.mlp_fc2"] += dx.T @ h_relu
        dh_relu = dx @ params[f"layer{i}.mlp_fc2"]
        dh = dh_relu * (h > 0)
        grads[f"layer{i}.mlp_fc1"] += dh.T @ x_norm2
        dx += rmsnorm_bwd(dh @ params[f"layer{i}.mlp_fc1"], x_norm2, ms2, sc2)
        grads[f"layer{i}.attn_wo"] += dx.T @ y_attn_h
        dy_attn = dx @ params[f"layer{i}.attn_wo"]
        dy_v    = dy_attn.reshape(T, n_head, head_dim).transpose(1, 0, 2)
        dp      = dy_v @ v.transpose(0, 2, 1)
        dv      = p.transpose(0, 2, 1) @ dy_v
        datt    = p * (dp - np.sum(dp * p, axis=-1, keepdims=True))
        datt   /= head_dim ** 0.5
        dq = (datt @ k).transpose(1, 0, 2).reshape(T, n_embd)
        dk = (datt.transpose(0, 2, 1) @ q).transpose(1, 0, 2).reshape(T, n_embd)
        dv = dv.transpose(1, 0, 2).reshape(T, n_embd)
        grads[f"layer{i}.attn_wq"] += dq.T @ x_norm1
        grads[f"layer{i}.attn_wk"] += dk.T @ x_norm1
        grads[f"layer{i}.attn_wv"] += dv.T @ x_norm1
        dx_norm1 = (dq @ params[f"layer{i}.attn_wq"]
                  + dk @ params[f"layer{i}.attn_wk"]
                  + dv @ params[f"layer{i}.attn_wv"])
        dx += rmsnorm_bwd(dx_norm1, x_norm1, ms1, sc1)
    np.add.at(grads["wte"], tokens, dx)
    np.add.at(grads["wpe"], np.arange(T), dx)
    return loss

# ── Adam optimizer state (fresh moments are fine for resuming) ────────────────
m_adam = {k: np.zeros_like(v) for k, v in params.items()}
v_adam = {k: np.zeros_like(v) for k, v in params.items()}

# ── training loop ─────────────────────────────────────────────────────────────
start_time = time.time()
for step in range(additional_steps):
    step_loss = 0.0
    for _ in range(batch_size):
        start_idx = np.random.randint(0, len(all_tokens) - block_size)
        tokens = all_tokens[start_idx: start_idx + block_size]
        logits, cache, x_final = forward(tokens)
        step_loss += backward(tokens, logits, cache, x_final) / batch_size

    lr_t = learning_rate * (1.0 - step / additional_steps)
    for k in params:
        np.clip(grads[k], -5.0, 5.0, out=grads[k])
        m_adam[k] = beta1 * m_adam[k] + (1 - beta1) * grads[k]
        v_adam[k] = beta2 * v_adam[k] + (1 - beta2) * grads[k] ** 2
        m_hat = m_adam[k] / (1 - beta1 ** (step + 1))
        v_hat = v_adam[k] / (1 - beta2 ** (step + 1))
        params[k] -= lr_t * m_hat / (np.sqrt(v_hat) + eps)
        grads[k].fill(0)

    if (step + 1) % 10 == 0 or step == 0:
        elapsed   = time.time() - start_time
        steps_done = step + 1
        eta_sec   = elapsed / steps_done * (additional_steps - steps_done)
        eta_fmt   = f"{int(eta_sec//3600):02d}:{int(eta_sec%3600//60):02d}:{int(eta_sec%60):02d}"
        print(f"step {steps_done:4d} / {additional_steps} | loss {step_loss:.4f} | ETA: {eta_fmt}", end="\r")

# ── save updated weights back to same checkpoint ──────────────────────────────
print("\n\n--- Resume Training Complete — saving weights ---")
model_data = {
    "vocab":        uchars,
    "idx_to_token": idx_to_token,
    "params":       {k: v.tolist() for k, v in params.items()},
    "n_layer":      n_layer,
    "n_embd":       n_embd,
    "block_size":   block_size,
    "n_head":       n_head,
}
with open(CHECKPOINT, "w") as f:
    json.dump(model_data, f)
print(f"Saved to {CHECKPOINT}")

# ── quick sanity generation ───────────────────────────────────────────────────
def generate(prompt, length=200, temp=0.4):
    toks   = get_tokens(prompt)
    tokens = [BOS] + [token_to_idx[t] for t in toks if t in token_to_idx]
    start  = len(tokens)
    for _ in range(length):
        ctx    = tokens[-block_size:]
        logits, _, _ = forward(np.array(ctx))
        l = logits[-1] / (temp + 1e-6)
        e = np.exp(l - np.max(l))
        p = e / np.sum(e)
        nxt = np.random.choice(len(p), p=p)
        if nxt == BOS:
            break
        tokens.append(nxt)
    return "".join(idx_to_token.get(t, "") for t in tokens[start:])

print("\n--- Sample generations ---")
for prompt in ["<user> Hello!", "<user> What is the capital of France?", "<user> Why do we yawn?"]:
    print(f"Prompt: {prompt}\nResponse: {generate(prompt)}\n")
