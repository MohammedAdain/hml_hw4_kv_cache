"""
bench_llm_ops.py — Benchmark Self-Attention operators.

Default configuration matches Llama-7B:
    num_heads=32, embed_dim_per_head=128, hidden_dim=4096

Usage examples (run via srun or sbatch on the GPU cluster):

  # Attention prefill (regular)
  python bench_llm_ops.py --bench attn --stage prefill --batch 16 --seq-len 256

  # Attention prefill (fused / FlashAttention)
  python bench_llm_ops.py --bench attn --stage prefill --batch 16 --seq-len 256 --fused

  # Attention decode (regular)
  python bench_llm_ops.py --bench attn --stage decode --batch 16 --seq-len 256

  # Attention decode (fused)
  python bench_llm_ops.py --bench attn --stage decode --batch 16 --seq-len 256 --fused
"""

import argparse
import time
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Default Llama-7B configuration ──────────────────────────────────────────
DEFAULT_NUM_HEADS        = 32
DEFAULT_EMBED_DIM_PER_HEAD = 128   # head_dim
DEFAULT_HIDDEN_DIM       = 4096    # = num_heads × head_dim

NUM_WARMUP = 10
NUM_ITERS  = 50

# ── Timing helpers ───────────────────────────────────────────────────────────

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


# ── Attention benchmarks ─────────────────────────────────────────────────────

def bench_attn_prefill(
    batch:      int,
    seq_len:    int,
    num_heads:  int,
    head_dim:   int,
    fused:      bool,
    device:     torch.device,
) -> None:
    """
    Benchmark self-attention for the Prefill stage.

    Regular mode: you implement and time each step (Q·K^T, masking, softmax, Attn·V).
    Fused mode: uses torch's scaled_dot_product_attention (FlashAttention-style).
    """
    # Q, K, V tensors: shape (batch, num_heads, seq_len, head_dim)
    Q = torch.randn(batch, num_heads, seq_len, head_dim, device=device)
    K = torch.randn(batch, num_heads, seq_len, head_dim, device=device)
    V = torch.randn(batch, num_heads, seq_len, head_dim, device=device)
    scale = head_dim ** -0.5

    # Causal mask: upper-triangular True entries are masked out
    # Shape: (1, 1, seq_len, seq_len) — ready for broadcasting with (B, nh, T, T)
    causal_mask = torch.triu(
        torch.ones(1, 1, seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1
    )

    print("Benchmarking for Attention prefill stage in ms")
    print(f"Input size: Batch={batch}, Seq_len={seq_len}, Num_heads={num_heads}, Embed_dim={head_dim}")

    if fused:
        with torch.no_grad():
            _, total_ms = _timed(
                lambda: F.scaled_dot_product_attention(Q, K, V, is_causal=True),
                device,
            )
        print(f"{'Total':>8}")
        print(f"{total_ms:8.3f}")
        return

    ########## Solution Block — Prefill Attention ##########
    # Implement each step of self-attention and measure its execution time.
    #
    # Available tensors:
    #   Q, K, V   — shape (batch, num_heads, seq_len, head_dim)
    #   scale     — float, = head_dim ** -0.5
    #   causal_mask — shape (1, 1, seq_len, seq_len), True = position to mask out
    #
    # For each step:
    #   1. Call _sync(device) to ensure prior GPU work is complete
    #   2. Record start time with time.perf_counter()
    #   3. Perform the operation
    #   4. Call _sync(device) again
    #   5. Compute elapsed time in milliseconds
    #
    # Hint: causal_mask already has shape (1, 1, T, T) and broadcasts directly
    #       with the attention scores of shape (batch, num_heads, T, T).

    with torch.no_grad():
        # 1. Q·K^T — compute scaled attention scores
        #    Result shape: (batch, num_heads, seq_len, seq_len)
        _sync(device)
        t0 = time.perf_counter()
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale   # ← implement: Q @ K^T * scale
        _sync(device)
        qkt_ms = (time.perf_counter() - t0) * 1e3

        # 2. Causal masking — fill future positions with -inf
        #    Use masked_fill with causal_mask
        _sync(device)
        t0 = time.perf_counter()
        masked = scores.masked_fill(causal_mask, float('-inf'))   # ← implement: apply causal_mask to scores
        _sync(device)
        mask_ms = (time.perf_counter() - t0) * 1e3

        # 3. Softmax — normalize attention weights over the last dimension
        _sync(device)
        t0 = time.perf_counter()
        weights = F.softmax(masked, dim=-1)  # ← implement: softmax of masked scores
        _sync(device)
        softmax_ms = (time.perf_counter() - t0) * 1e3

        # 4. Attention · V — compute weighted sum of values
        #    Result shape: (batch, num_heads, seq_len, head_dim)
        _sync(device)
        t0 = time.perf_counter()
        output = torch.matmul(weights, V)   # ← implement: weights @ V
        _sync(device)
        attn_v_ms = (time.perf_counter() - t0) * 1e3

    ########## End Solution Block — Prefill Attention ##########

    if scores is None or masked is None or weights is None or output is None:
        raise RuntimeError(
            "Prefill attention not implemented. "
            "Fill in the Solution Block in bench_attn_prefill()."
        )

    total_ms = qkt_ms + mask_ms + softmax_ms + attn_v_ms

    print(f"{'Q_K^T':>8}, {'Masking':>8}, {'Softmax':>8}, {'Attn_V':>8}, {'Total':>8}")
    print(f"{qkt_ms:8.3f}, {mask_ms:8.3f}, {softmax_ms:8.3f}, {attn_v_ms:8.3f}, {total_ms:8.3f}")


def bench_attn_decode(
    batch:      int,
    seq_len:    int,
    num_heads:  int,
    head_dim:   int,
    fused:      bool,
    device:     torch.device,
) -> None:
    """
    Benchmark self-attention for the Decode stage.

    During decode, only ONE new query token is computed; the keys and values
    for all previous (seq_len) tokens come from the KV cache.

    You must implement the Solution Block below to set up the tensors and
    compute each attention step with timing.
    """

    scale = head_dim ** -0.5

    print("Benchmarking for Attention decode stage in ms")
    print(f"Input size: Batch={batch}, Seq_len={seq_len}, Num_heads={num_heads}, Embed_dim={head_dim}")

    ########## Solution Block — Decode Attention ##########
    # Implement decode-stage attention with KV-cache tensors.
    #
    # In the decode stage:
    #   - The model generates ONE new token at a time.
    #   - Q_new is the query for that single new token.
    #   - K_cache and V_cache hold keys/values for all previous tokens.
    #   - No causal mask is needed (the single query attends to ALL past positions).
    #
    # Step 0: Declare random tensors with the correct shapes:
    #   Q_new   : (batch, num_heads,       1, head_dim)  ← single query
    #   K_cache : (batch, num_heads, seq_len, head_dim)  ← cached keys
    #   V_cache : (batch, num_heads, seq_len, head_dim)  ← cached values
    #   Then set: Q, K, V = Q_new, K_cache, V_cache
    #
    # Then implement and time each step (same pattern as prefill, but no masking).

    # 0. Declare KV-cache tensors
    Q_new = torch.randn(batch, num_heads, 1, head_dim, device=device)
    K_cache = torch.randn(batch, num_heads, seq_len, head_dim, device=device)
    V_cache = torch.randn(batch, num_heads, seq_len, head_dim, device=device)
    Q = Q_new   # ← implement
    K = K_cache   # ← implement
    V = V_cache  # ← implement

    with torch.no_grad():
        # 1. Q·K^T — shape (batch, num_heads, 1, seq_len)
        _sync(device)
        t0 = time.perf_counter()
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale   # ← implement: Q @ K^T * scale
        _sync(device)
        qkt_ms = (time.perf_counter() - t0) * 1e3

        # 2. Softmax — normalize over the last dimension
        _sync(device)
        t0 = time.perf_counter()
        weights = F.softmax(scores, dim=-1)  # ← implement: softmax of scores
        _sync(device)
        softmax_ms = (time.perf_counter() - t0) * 1e3

        # 3. Attention · V — shape (batch, num_heads, 1, head_dim)
        _sync(device)
        t0 = time.perf_counter()
        output = torch.matmul(weights, V)   # ← implement: weights @ V
        _sync(device)
        attn_v_ms = (time.perf_counter() - t0) * 1e3

    ########## End Solution Block — Decode Attention ##########

    if Q is None or K is None or V is None:
        raise RuntimeError(
            "Decode attention not implemented. "
            "Fill in the Solution Block in bench_attn_decode()."
        )

    if fused:
        with torch.no_grad():
            _, total_ms = _timed(
                lambda: F.scaled_dot_product_attention(Q, K, V, is_causal=False),
                device,
            )
        print(f"{'Total':>8}")
        print(f"{total_ms:8.3f}")
        return

    total_ms = qkt_ms + softmax_ms + attn_v_ms

    print(f"{'Q_K^T':>8}, {'Softmax':>8}, {'Attn_V':>8}, {'Total':>8}")
    print(f"{qkt_ms:8.3f}, {softmax_ms:8.3f}, {attn_v_ms:8.3f}, {total_ms:8.3f}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark Self-Attention operators.")
    p.add_argument("--bench",      choices=["attn"], required=True,
                   help="Which operator to benchmark.")
    p.add_argument("--stage",      choices=["prefill", "decode"], default="prefill",
                   help="Attention stage (prefill or decode).")
    p.add_argument("--batch",      type=int, default=16)
    p.add_argument("--seq-len",    type=int, default=256)
    p.add_argument("--num-heads",  type=int, default=DEFAULT_NUM_HEADS)
    p.add_argument("--head-dim",   type=int, default=DEFAULT_EMBED_DIM_PER_HEAD)
    p.add_argument("--fused",      action="store_true",
                   help="Use fused (FlashAttention-style) kernel via scaled_dot_product_attention.")
    p.add_argument("--device",     type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    device = torch.device(args.device)
    print(f"Running on device: {device}\n")

    if args.bench == "attn":
        if args.stage == "prefill":
            bench_attn_prefill(
                args.batch, args.seq_len,
                args.num_heads, args.head_dim,
                args.fused, device,
            )
        else:
            bench_attn_decode(
                args.batch, args.seq_len,
                args.num_heads, args.head_dim,
                args.fused, device,
            )
