#!/bin/bash
#SBATCH -J hml_lab4                       # Job name
#SBATCH -N1 --ntasks-per-node=1           # Number of nodes and tasks
#SBATCH --gres=gpu:H100:1                 # GPU type (H100) and number of GPUs
#SBATCH --mem-per-gpu=224GB               # Memory per GPU
#SBATCH -t1:00:00                         # Duration of the job (1 hour)
#SBATCH -o output.%J.log                  # Output file
#SBATCH --mail-type=BEGIN,END,FAIL        # Mail preferences
#SBATCH --mail-user=YOUR_EMAIL@gatech.edu # Replace with your GT email

# Load the PyTorch container module (includes torch, matplotlib, numpy)
module load pytorch

# ── Edit the command below to run the script you want ────────────────────────

# Example: Prefill profiling
python bench_llama.py --stage prefill --batch 1 --seq-len 1024

# Example: Decode profiling (uncomment after completing the Solution Block)
# python bench_llama.py --stage decode --batch 1 --seq-len 1024

# Example: GEMM benchmark
# python bench_llm_ops.py --bench gemm --batch 16 --seq-len 128

# Example: Attention prefill (regular)
# python bench_llm_ops.py --bench attn --stage prefill --batch 16 --seq-len 256

# Example: Attention prefill (fused)
# python bench_llm_ops.py --bench attn --stage prefill --batch 16 --seq-len 256 --fused

# Example: Attention decode (regular, after completing the Solution Block)
# python bench_llm_ops.py --bench attn --stage decode --batch 16 --seq-len 256
