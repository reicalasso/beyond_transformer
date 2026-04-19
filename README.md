# PULSE — Hybrid Gated Delta Network

[CI](https://github.com/reicalasso/beyond_transformer/actions/workflows/ci.yml)
[License: MIT](https://opensource.org/licenses/MIT)
[Python 3.10+](https://www.python.org/downloads/)
[PyTorch 2.1+](https://pytorch.org/)
[Status]()

> A research sequence model that drops the attention–FFN duopoly in favor of a
> **gated delta-rule recurrence** with a matrix-valued state, interleaved every
> few layers with **sliding-window softmax attention** for in-context recall.

---

## TL;DR

Modern long-context architectures converged on a small set of ideas:

1. **Recurrence with data-dependent gating** (Mamba-2, GLA, RWKV-7) for O(1)
  per-token inference.
2. **Delta-rule writes** with matrix-valued state (DeltaNet, Yang et al. 2024)
  for honest associative recall — *targeted overwrite*, not just decay.
3. **Hybrid stacks** with periodic short-window softmax attention
  (Samba, Jamba, Zamba-2) to keep exact local recall.
4. **Stability tricks** at scale: QK-norm, RoPE in the attention sub-layers,
  logit soft-cap, z-loss, μP-style residual init.

PULSE bundles these into a single, tested package. The recurrence path runs
in `O(N · D²)` (matrix-state) per layer with `O(D²)` per token at decode
time, regardless of context length. The SWA layers add a fixed
`O(window · D)` per token.

---

## Architecture

A `PulseForCausalLM` is a stack of two block types, interleaved by a
`swa_every` schedule (default 4 → 75% delta, 25% SWA):

### `DeltaBlock` — recurrent mixer (the workhorse)

```
x → RMSNorm → ShortCausalConv1d → GatedDeltaRule → residual
  → RMSNorm → SwiGLU                              → residual
```

Inside `GatedDeltaRule`, per token and per head:

```
q, k, v   ← Linear(x)            (QK-normed)
α_t       ← σ( W_α · x )         per-head scalar forget gate
β_t       ← σ( W_β · x )         per-head scalar write strength
S_t       =  α_t · S_{t-1} · (I − β_t k_t k_t^T) + β_t · v_t k_t^T   ∈ R^{Dh×Dh}
o_t       =  S_t · q_t
```

The state `S_t` is a true `Dh × Dh` matrix per head — a **proper associative
memory**, not the rank-1 element-wise approximation used in vanilla linear
attention. Two execution paths are exposed and tested for bit-equivalence:
a per-token recurrent reference, and a chunked path that ties into a future
fused intra-chunk WY-representation kernel.

### `AttentionBlock` — sliding-window recall

```
x → RMSNorm → ShortCausalConv1d → SlidingWindowAttention(RoPE, GQA, QK-norm)
                                                  → residual
  → RMSNorm → SwiGLU                              → residual
```

Standard causal softmax attention restricted to the last `window_size`
tokens, with RoPE on `(q, k)`, optional GQA via `num_kv_heads`, and a
ring-buffer KV cache for O(window) decode memory.

### Top-level

- `PulseForCausalLM` adds:
  - Logit **soft-cap** (Gemma-style): `tanh(logits / cap) * cap`.
  - Auxiliary **z-loss**: `λ · (log Z)²` for log-partition stability.
  - O(1)-per-step **incremental generate** that prefills once then carries
  `(conv_state, mixer_state)` per layer.
  - Optional **tied embeddings**, **μP-style residual init** (`1/√(2L)`
  on `out_proj` and `ffn.down`).

---

## Quick start

```bash
git clone https://github.com/reicalasso/beyond_transformer.git
cd beyond_transformer
pip install -e ".[train,dev]"
```

```python
import torch
from pulse import PulseConfig, PulseForCausalLM

cfg = PulseConfig(
    vocab_size=50_257,
    hidden_size=512,
    num_layers=12,
    num_heads=8,
    swa_every=4,            # 9 delta + 3 swa
    swa_window_size=512,
    delta_chunk_size=64,
)
model = PulseForCausalLM(cfg)

ids = torch.randint(0, cfg.vocab_size, (1, 64))
out = model(ids, labels=ids)
print(out["loss"].item())           # CE + z-loss

generated = model.generate(ids[:, :8], max_new_tokens=120, top_k=50, top_p=0.9)
```

```bash
# Sanity check
python scripts/smoke_test.py

# Microbenchmark (forward / train / decode latency + memory)
python scripts/bench.py --hidden-size 384 --num-layers 8 --seq-len 512

# Train on TinyStories (defaults: 512-dim, 12-layer, 20k steps)
python scripts/train.py
python scripts/train.py --config configs/tinystories_small.yaml
python scripts/train.py --config configs/rtx4090_tinystories.yaml

# Probe a checkpoint
python scripts/test_best_model.py --checkpoint output/pulse/best_model.pt
```

---

## Configuration

All hyperparameters live in `PulseConfig`. The most architecture-defining ones:


| Field                 | Default | Meaning                                         |
| --------------------- | ------- | ----------------------------------------------- |
| `hidden_size`         | 512     | Model dimension.                                |
| `num_layers`          | 12      | Total blocks (delta + swa).                     |
| `num_heads`           | 8       | Heads in both block types.                      |
| `num_kv_heads`        | None    | None → MHA in SWA layers; smaller value → GQA.  |
| `swa_every`           | 4       | Place an SWA block every Nth layer.             |
| `swa_window_size`     | 512     | Tokens visible to SWA layers.                   |
| `delta_chunk_size`    | 64      | Chunk granularity in delta-rule prefill.        |
| `qk_norm`             | True    | L2-normalize Q and K (stable at scale).         |
| `gate_bias_init`      | 4.0     | logit s.t. `σ(α) ≈ 0.98` at init (long memory). |
| `conv_kernel_size`    | 4       | Short causal conv before each mixer.            |
| `logit_soft_cap`      | 30.0    | Gemma-style; set None to disable.               |
| `z_loss_coef`         | 1e-4    | Auxiliary stability loss; set 0 to disable.     |
| `tie_embeddings`      | True    | Share input/output embedding weights.           |
| `init_scale_residual` | True    | Scale residual projection weights by `1/√(2L)`. |


YAML configs live in `configs/` and override CLI defaults; CLI flags then
override YAML.

---

## Layout

```
src/pulse/
├── __init__.py            # public API: PulseConfig, PulseModel, PulseForCausalLM
├── config.py              # dataclass config, layer-pattern resolver
├── model.py               # PulseModel, PulseForCausalLM
├── train.py               # TinyStories training loop
├── modules/
│   ├── norm.py            # RMSNorm + L2-normalize (QK-norm)
│   ├── rope.py            # RoPE with growable cache
│   ├── conv.py            # ShortCausalConv1d (streaming)
│   ├── ffn.py             # SwiGLU
│   ├── delta.py           # GatedDeltaRule (recurrent + chunked paths)
│   ├── swa.py             # SlidingWindowAttention (RoPE + GQA + ring-buffer cache)
│   └── block.py           # DeltaBlock, AttentionBlock
├── kernels/               # placeholder for future Triton kernel
└── legacy/                # the original PULSE v3 dual-timescale prototype

tests/
├── test_norm.py           # RMSNorm + L2-norm correctness
├── test_rope.py           # RoPE half-rotation + offset + relative property
├── test_conv.py           # streaming conv == full conv
├── test_delta.py          # recurrent ≡ chunked, prefill ≡ decode, gradients
├── test_swa.py            # window cap, causality, GQA, padding mask
├── test_block.py          # streaming equivalence per block
├── test_model.py          # end-to-end shapes, soft-cap, prefill ≡ decode
└── test_legacy_compat.py  # the legacy prototype still imports and runs

scripts/
├── smoke_test.py          # shape + finiteness sanity check
├── bench.py               # forward/train/decode latency + memory
├── train.py               # → pulse.train.main
└── test_best_model.py     # generation + perplexity probe on a checkpoint
```

---

## Tests, lint, types

```bash
ruff check src tests scripts          # lint
ruff format --check src tests scripts # formatting
mypy src/pulse                        # static types
pytest -q                             # 46 tests, < 1 s on CPU
```

CI runs all four on every push/PR across Python 3.10/3.11/3.12.

The architecture is correctness-tested, not just shape-tested. The
non-negotiable invariants are:

- `forward_recurrent` ≡ `forward_chunked` for any `chunk_size` (delta).
- Per-block streaming decode ≡ full-sequence prefill (delta + swa + block).
- Full-model prefill ≡ split-prefill+decode at the logits level.
- Gradients are finite and flow to all parameters.
- SWA padding mask actually masks padded positions out of softmax.
- RoPE preserves vector norms and satisfies the relative-position invariant.

---

## What this is — and isn't

- **Is**: a clean, tested, modern hybrid recurrent + sliding-window
architecture in pure PyTorch, suitable for research and small-scale
pretraining experiments. The math is correct; the architecture matches
the 2024–2026 frontier of efficient sequence models.
- **Isn't**: tuned for inference-time throughput. The chunked delta path
uses a Python loop over chunks — competitive with reference PyTorch
implementations but ~10× slower than a fused Triton kernel. A drop-in
Triton WY-representation kernel slot is reserved under
`src/pulse/kernels/`; contributions welcome.

---

## Roadmap

- Triton WY-representation kernel for fully parallel intra-chunk delta.
- HF-compatible `PreTrainedModel` shim for the wider ecosystem.
- MoE SwiGLU (DeepSeekMoE-style with shared expert) behind a config flag.
- Larger-scale ablation tables: hybrid ratios, delta vs GLA, with/without
QK-norm, with/without z-loss, with/without residual init scaling.
- Optional μP-coordinate-check script for HP transfer experiments.

---

## Legacy

The original PULSE v3 dual-timescale prototype is preserved verbatim under
`pulse.legacy` for reproducibility. It is **not** loaded by the new public
API — use it only if you specifically need the v3 architecture:

```python
from pulse.legacy import PulseConfig, PulseForCausalLM
```

---

*Feedback, issues, and architecture critiques welcome.*