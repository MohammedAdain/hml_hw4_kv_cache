# Lab 4: LLM Inference Optimization

This lab combines two assignments that together cover the full picture of LLM inference efficiency: from implementing KV-caching in a from-scratch GPT model, to deep performance profiling of a production-scale Llama architecture on NVIDIA GPUs.

---

## Part 1 — Optimizing LLM Inference with KV-Caching (80 points)

**Notebook:** `part1/lab4_part1.ipynb`

Implement KV-caching for a nanoGPT-style transformer and benchmark the speedup. All instructions, questions (Q1–Q9), and answer cells are in the notebook.

| Step | Topic | Points |
|------|-------|--------|
| Step 1 | Understanding Self-Attention (Q1–Q4) | 15 |
| Step 2 | Benchmarking Baseline Generation (Q5–Q6) | 15 |
| Step 3 | Implementing & Benchmarking KV-Caching (Q7–Q9) | 50 |

Answer all questions directly in the notebook markdown cells provided. Use `output_check()` in the notebook to validate your KV-cache implementation before submitting.

---

## Part 2 — Understanding LLM Performance

**Scripts:** `part2/bench_llama.py`, `part2/bench_llm_ops.py`

You will perform deep performance profiling of Llama's decode layer and key LLM operators on NVIDIA GPUs, and understand the motivation for FlashAttention.

### Preliminary

All experiments run on the GPU cluster. **Do NOT run jobs on the login node.**

```bash
# Load the PyTorch container module (includes torch, matplotlib, numpy)
module load pytorch

# Submit via sbatch
sbatch hw2.sbatch

# Or request an interactive GPU session
salloc -N1 --ntasks-per-node=1 --gres=gpu:H100:1 --mem-per-gpu=224GB -t1:00:00
```

---

### Section A — Profile Llama Decode Layer (30 points)

Script: `bench_llama.py`

Profiles the Llama-7B decoder block (Self-Attention, MLPs, Misc) in ms.

#### A.1 Prefill Stage (10 pt)

```bash
python bench_llama.py --stage prefill --batch 1 --seq-len 1024
```

Expected output:
```
Prefill Stage Time Results (ms)
Input size: Batch=1, Seq_len=1024, Hidden_dim=4096
Self-Attn,       MLPs,       Misc,      Total
    9.766,     15.232,      0.404,     25.402
```

> **Note:** These values were measured on an NVIDIA A100-SXM4-80GB. Your results will vary depending on GPU model — any reasonable, ballpark numbers are acceptable.

- **Q1 (5 pt):** Set `batch=1`. Benchmark Prefill latency for `seq_len=[256,512,1024,2048,4096,8192,12288]`. Plot Self-Attention and MLP latency. Explain.
- **Q2 (5 pt):** Set `seq_len=1024`. Sweep `batch=[1,2,4,8,16,32,64]`. Plot and explain.

#### A.2 Decode Stage (20 pt)

Complete the `bench_decode()` function's Solution Block (declare the KV-cache tensors with the correct shapes), then:

```bash
python bench_llama.py --stage decode --batch 1 --seq-len 1024
```

- **Q3 (5 pt):** Submit your completed `bench_decode()` implementation.
- **Q4 (5 pt):** Redo Q1 for the Decode stage.
- **Q5 (5 pt):** Redo Q2 for the Decode stage.
- **Q6 (5 pt):** What is the dominant operator in Prefill vs. Decode when sequence length is long?

---

### Section B — Profile Self-Attention (50 points)

Script: `bench_llm_ops.py`

Default config: `num_heads=32`, `embed_dim_per_head=128`, `hidden_dim=4096` (Llama-7B).

#### B.1 Attention — Prefill Stage (20 pt)

Complete the Solution Block in `bench_attn_prefill()` — implement each attention step (Q·K^T, masking, softmax, Attn·V) and time them individually. Then run:

```bash
python bench_llm_ops.py --bench attn --stage prefill --batch 16 --seq-len 256
python bench_llm_ops.py --bench attn --stage prefill --batch 16 --seq-len 256 --fused
```

- **Q7 (10 pt):** Submit your prefill attention implementation. Benchmark regular attention for `seq_len=[256,512,1024,2048,4096,8192,12288,16384]` with fixed `batch=16` and `batch=[1,4,8,16,32]` with fixed `seq_len=256`. Skip OOM cases. Plot throughput vs. intensity. Explain.
- **Q8 (10 pt):** Benchmark fused attention. Compute speedup vs. regular attention. Plot speedup line chart. Explain.

#### B.2 Attention — Decode Stage (30 pt)

Complete the Solution Block in `bench_attn_decode()` — declare KV-cache tensors and implement each attention step with timing. Then run:

```bash
python bench_llm_ops.py --bench attn --stage decode --batch 16 --seq-len 256
```

- **Q9 (10 pt):** Submit your `bench_attn_decode()` implementation (with KV-cache).
- **Q10 (5 pt):** Set `batch=16`. Benchmark for `seq_len=[256,512,1024,2048,4096,8192,16384,32768]`. Plot throughput vs. intensity. Explain.
- **Q11 (5 pt):** Set `seq_len=2048`. Benchmark for `batch=[16,32,64,128,256,512,1024]`. Plot throughput vs. intensity. Explain.
- **Q12 (10 pt):** Benchmark fused attention for the Q10/Q11 cases. Compare with regular attention. Explain.

---

## Submission

Submit a single archive containing all deliverables for both parts:

### What to include

| Part | Files | Details |
|------|-------|---------|
| **Part 1** | `part1/lab4_part1.ipynb` | Completed notebook with all Solution Blocks filled in, Q1–Q9 answered in markdown cells, and benchmark graphs generated |
| **Part 2** | `part2/bench_llama.py` | With completed `bench_decode()` Solution Block |
| **Part 2** | `part2/bench_llm_ops.py` | With completed prefill and decode Solution Blocks |
| **Part 2** | `part2/report.pdf` | PDF report answering Q1–Q12 with all plots and explanations |

### How to submit

```bash
tar -czf firstname_StudentID.tar.gz your_folder/
```

Submit the archive to Canvas.
