"""
bench_llama.py — Profile a single Llama-7B decoder block.

Usage examples (run via srun or sbatch on the GPU cluster):
    python bench_llama.py --stage prefill --batch 1 --seq-len 1024
    python bench_llama.py --stage decode  --batch 1 --seq-len 1024

Output columns: Self-Attn (ms), MLPs (ms), Misc (ms), Total (ms)
"""

import argparse
import math
import time
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Llama-7B hyper-parameters ────────────────────────────────────────────────
HIDDEN_DIM       = 4096
NUM_HEADS        = 32
NUM_KV_HEADS     = 32          # no GQA in 7B
INTERMEDIATE_SIZE = 11008
HEAD_DIM         = HIDDEN_DIM // NUM_HEADS
MAX_SEQ_LEN      = 32768

NUM_WARMUP = 10
NUM_ITERS  = 50

# ── Building blocks ──────────────────────────────────────────────────────────

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * x / rms


class LlamaRotaryEmbedding(nn.Module):
    """Pre-computed RoPE cosine/sine tables."""

    def __init__(self, dim: int, max_seq_len: int = MAX_SEQ_LEN, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos[position_ids].unsqueeze(1)  # (B, 1, T, head_dim)
    sin = sin[position_ids].unsqueeze(1)
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    return q, k


class LlamaSelfAttention(nn.Module):
    def __init__(self, hidden_dim: int = HIDDEN_DIM, num_heads: int = NUM_HEADS):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T, C = hidden_states.shape
        nh, hs  = self.num_heads, self.head_dim

        q = self.q_proj(hidden_states).view(B, T, nh, hs).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, T, nh, hs).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, T, nh, hs).transpose(1, 2)

        kv_len = T + (past_key_value[0].shape[2] if past_key_value is not None else 0)
        cos, sin = self.rotary_emb(kv_len)
        q, k = _apply_rotary_emb(q, k, cos, sin, position_ids)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        present_key_value = (k, v)

        scale = hs ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Causal mask during prefill only
        if T > 1:
            total_kv = k.shape[2]
            mask = torch.tril(
                torch.ones(T, total_kv, dtype=torch.bool, device=hidden_states.device),
                diagonal=total_kv - T,
            )
            attn = attn.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        out  = torch.matmul(attn, v)
        out  = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out), present_key_value


class LlamaMLP(nn.Module):
    def __init__(self, hidden_dim: int = HIDDEN_DIM, intermediate_size: int = INTERMEDIATE_SIZE):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_size, bias=False)
        self.up_proj   = nn.Linear(hidden_dim, intermediate_size, bias=False)
        self.down_proj  = nn.Linear(intermediate_size, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaDecoderBlock(nn.Module):
    def __init__(
        self,
        hidden_dim:        int = HIDDEN_DIM,
        num_heads:         int = NUM_HEADS,
        intermediate_size: int = INTERMEDIATE_SIZE,
    ):
        super().__init__()
        self.input_layernorm          = LlamaRMSNorm(hidden_dim)
        self.self_attn                = LlamaSelfAttention(hidden_dim, num_heads)
        self.post_attention_layernorm = LlamaRMSNorm(hidden_dim)
        self.mlp                      = LlamaMLP(hidden_dim, intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids:  torch.Tensor,
        past_key_value: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, Tuple]:
        # Self-attention sub-layer
        residual      = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out, present_kv = self.self_attn(hidden_states, position_ids, past_key_value)
        hidden_states = residual + attn_out

        # MLP sub-layer
        residual      = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, present_kv


# ── Timing helper ────────────────────────────────────────────────────────────

def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _timed(fn, device: torch.device, warmup: int = NUM_WARMUP, iters: int = NUM_ITERS):
    """Return (last_result, avg_time_ms) of fn()."""
    for _ in range(warmup):
        result = fn()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        result = fn()
    _sync(device)
    return result, (time.perf_counter() - t0) / iters * 1e3


# ── Prefill benchmark ────────────────────────────────────────────────────────

def bench_prefill(batch: int, seq_len: int, device: torch.device) -> None:
    model = LlamaDecoderBlock().to(device).eval()

    hidden = torch.randn(batch, seq_len, HIDDEN_DIM, device=device)
    pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)

    with torch.no_grad():
        # --- Total ---
        _, total_ms = _timed(lambda: model(hidden, pos_ids), device)

        # --- Self-attention only ---
        normed = model.input_layernorm(hidden)
        (attn_out, _), attn_ms = _timed(lambda: model.self_attn(normed, pos_ids), device)

        # --- MLP only ---
        post_attn   = hidden + attn_out
        normed_post = model.post_attention_layernorm(post_attn)
        _, mlp_ms = _timed(lambda: model.mlp(normed_post), device)

    misc_ms = max(total_ms - attn_ms - mlp_ms, 0.0)

    print("Prefill Stage Time Results (ms)")
    print(f"Input size: Batch={batch}, Seq_len={seq_len}, Hidden_dim={HIDDEN_DIM}")
    print(f"{'Self-Attn':>10}, {'MLPs':>10}, {'Misc':>10}, {'Total':>10}")
    print(f"{attn_ms:10.3f}, {mlp_ms:10.3f}, {misc_ms:10.3f}, {total_ms:10.3f}")


# ── Decode benchmark ─────────────────────────────────────────────────────────

def bench_decode(batch: int, seq_len: int, device: torch.device) -> None:
    """
    Profile the Decode stage of a single Llama decoder block.

    During decode, the model receives ONE new token (seq_len=1) together with
    a KV-cache that holds the key/value tensors for all *previous* tokens.
    """
    model = LlamaDecoderBlock().to(device).eval()

    # The single new token's hidden state: shape (batch, 1, hidden_dim)
    hidden  = torch.randn(batch, 1, HIDDEN_DIM, device=device)
    # Position of the current token (0-indexed)
    pos_ids = torch.tensor([[seq_len - 1]], device=device).expand(batch, -1)

    ########## Solution Block ##########
    # Declare random PyTorch tensors that match the shapes of the KV cache.
    #
    # The KV cache holds key and value tensors for all *previous* tokens.
    # At decode step for token at position (seq_len - 1), there are
    # (seq_len - 1) cached tokens.
    #
    # Required shapes:
    #   past_k : (batch, NUM_HEADS, seq_len - 1, HEAD_DIM)
    #   past_v : (batch, NUM_HEADS, seq_len - 1, HEAD_DIM)
    #
    # Store them as a tuple:  past_key_value = (past_k, past_v)

    past_k = torch.randn(batch, NUM_HEADS, seq_len - 1, HEAD_DIM)
    past_v = torch.randn(batch, NUM_HEADS, seq_len - 1, HEAD_DIM)

    past_key_value = (past_k, past_v)   # ← replace with real tensors

    ########## End Solution Block ##########

    if past_key_value is None:
        raise RuntimeError(
            "past_key_value is not implemented. "
            "Fill in the Solution Block in bench_decode()."
        )

    with torch.no_grad():
        # --- Total ---
        _, total_ms = _timed(lambda: model(hidden, pos_ids, past_key_value), device)

        # --- Self-attention only ---
        normed = model.input_layernorm(hidden)
        (attn_out, _), attn_ms = _timed(
            lambda: model.self_attn(normed, pos_ids, past_key_value), device
        )

        # --- MLP only ---
        post_attn   = hidden + attn_out
        normed_post = model.post_attention_layernorm(post_attn)
        _, mlp_ms = _timed(lambda: model.mlp(normed_post), device)

    misc_ms = max(total_ms - attn_ms - mlp_ms, 0.0)

    print("Decode Stage Time Results (ms)")
    print(f"Input size: Batch={batch}, Seq_len={seq_len}, Hidden_dim={HIDDEN_DIM}")
    print(f"{'Self-Attn':>10}, {'MLPs':>10}, {'Misc':>10}, {'Total':>10}")
    print(f"{attn_ms:10.3f}, {mlp_ms:10.3f}, {misc_ms:10.3f}, {total_ms:10.3f}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark a single Llama-7B decoder block.")
    p.add_argument("--stage",   choices=["prefill", "decode"], required=True)
    p.add_argument("--batch",   type=int, default=1)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--device",  type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    device = torch.device(args.device)
    print(f"Running on device: {device}\n")

    if args.stage == "prefill":
        bench_prefill(args.batch, args.seq_len, device)
    else:
        bench_decode(args.batch, args.seq_len, device)
