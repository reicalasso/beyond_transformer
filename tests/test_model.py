"""End-to-end tests for PulseModel and PulseForCausalLM."""

from __future__ import annotations

import torch

from pulse import PulseConfig, PulseForCausalLM, PulseModel


def _config(**overrides) -> PulseConfig:
    cfg = dict(
        vocab_size=64,
        hidden_size=32,
        num_layers=4,
        num_heads=4,
        ffn_mult=2.0,
        conv_kernel_size=3,
        swa_every=2,
        swa_window_size=8,
        delta_chunk_size=4,
        max_seq_len=64,
        rope_max_seq_len=64,
    )
    cfg.update(overrides)
    return PulseConfig(**cfg)


def test_resolved_layer_pattern() -> None:
    cfg = _config(num_layers=4, swa_every=2)
    assert cfg.resolved_layer_types() == ["delta", "swa", "delta", "swa"]


def test_model_forward_shapes() -> None:
    cfg = _config()
    model = PulseModel(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 12))
    with torch.no_grad():
        h, states = model(ids)
    assert h.shape == (2, 12, cfg.hidden_size)
    assert len(states) == cfg.num_layers


def test_lm_forward_loss_finite() -> None:
    cfg = _config()
    model = PulseForCausalLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 12))
    out = model(ids, labels=ids)
    assert torch.isfinite(out["loss"])
    assert out["logits"].shape == (2, 12, cfg.vocab_size)
    assert "ce_loss" in out and "z_loss" in out


def test_lm_logit_soft_cap_bounds() -> None:
    cfg = _config(logit_soft_cap=5.0)
    model = PulseForCausalLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        out = model(ids)
    logits = out["logits"]
    assert (logits.abs() <= 5.0 + 1e-3).all()


def test_lm_generate_token_count() -> None:
    cfg = _config()
    model = PulseForCausalLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (1, 4))
    out = model.generate(ids, max_new_tokens=5, top_k=5, top_p=1.0, temperature=1.0)
    assert out.shape == (1, 9)


def test_prefill_then_decode_state_consistency() -> None:
    """Prefill once, then decode step-by-step; each step's logit must equal
    a fresh full-sequence forward up to that point (greedy / argmax check)."""
    cfg = _config(num_layers=2, swa_every=2, swa_window_size=16, delta_chunk_size=4)
    model = PulseForCausalLM(cfg)

    ids = torch.randint(0, cfg.vocab_size, (1, 6))
    with torch.no_grad():
        out_full = model(ids)
        logits_full = out_full["logits"]  # [1, 6, V]

        out_pref = model(ids[:, :3])
        states = out_pref["layer_states"]
        # Step 4
        out_step = model(ids[:, 3:4], layer_states=states)
        torch.testing.assert_close(
            out_step["logits"][:, -1], logits_full[:, 3], rtol=1e-3, atol=1e-3
        )
        # Step 5
        out_step = model(ids[:, 4:5], layer_states=out_step["layer_states"])
        torch.testing.assert_close(
            out_step["logits"][:, -1], logits_full[:, 4], rtol=1e-3, atol=1e-3
        )


def test_param_count_and_tied_embedding() -> None:
    cfg = _config(tie_embeddings=True)
    model = PulseForCausalLM(cfg)
    assert model.lm_head.weight.data_ptr() == model.model.embed.weight.data_ptr()


def test_param_count_untied() -> None:
    cfg = _config(tie_embeddings=False)
    model = PulseForCausalLM(cfg)
    assert model.lm_head.weight.data_ptr() != model.model.embed.weight.data_ptr()


def test_padding_mask_in_swa() -> None:
    """Smoke: model should accept and forward an attention mask."""
    cfg = _config()
    model = PulseForCausalLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    mask = torch.tensor([[1, 1, 1, 0, 0, 1, 1, 1]])
    with torch.no_grad():
        out = model(ids, attention_mask=mask)
    assert torch.isfinite(out["logits"]).all()
