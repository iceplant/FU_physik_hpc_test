# PyTorch Benchmark: CPU ↔ GPU ↔ Multi‑GPU

## Overview
This repository contains:
- **`benchmark_train.py`**: A PyTorch script that trains a small CNN on synthetic data (28x28 single-channel images)
- **`requirements.txt`**: Dependencies for running the benchmark
- **`run_benchmark.sbatch`**: Slurm submission script with torchrun for distributed training

The benchmark features:
- Synthetic data generation with different seeds per GPU
- Configurable model size via hidden dimensions
- Adjustable runtime via samples and epochs
- Support for CPU, single GPU, and multi-GPU training
- Distributed training using PyTorch DDP

## 1. Prerequisites

1. **Python ≥ 3.9**
2. **CUDA-compatible GPUs** (for GPU training)
3. **Slurm workload manager**

## 2. Setting Up Your Environment

1. **Load Required Modules**
```bash
module purge
module load Python/3.9.5-GCCcore-10.3.0
module load cuda/11.7
```

2. **Create Virtual Environment**
```bash
# Create venv in project directory
python -m venv venv
source venv/bin/activate

# Install requirements using UV (recommended)
export UV_CACHE_DIR="/srv/data/$USER/cache/uv"
mkdir -p $UV_CACHE_DIR
uv pip install -r requirements.txt

# Or using pip
pip install -r requirements.txt
```

## 3. Running the Benchmark

### Single GPU/CPU Run
```bash
# For CPU/single GPU
python benchmark_train.py \
    --epochs 5 \
    --batch_size 64 \
    --num_samples 10000 \
    --hidden_dim 128 \
    --local_rank -1
```

### Multi-GPU Run (Distributed Training)
```bash
# Submit as Slurm job (recommended)
sbatch run_benchmark.sbatch

# Or run directly with torchrun
torchrun --nproc_per_node=4 benchmark_train.py \
    --epochs 5 \
    --batch_size 64 \
    --num_samples 10000 \
    --hidden_dim 128
```

### Scaling Options
- Quick test: `--epochs 5 --num_samples 10000` (~1 minute)
- Medium run: `--epochs 10 --num_samples 100000` (~10 minutes)
- Long benchmark: `--epochs 20 --num_samples 500000 --hidden_dim 512` (~1 hour)


## 4. Output & Monitoring

The benchmark reports:
- Training loss per epoch
- Total runtime
- Device information and world size (number of GPUs)
- Output saved to: `pt_bench_[jobid].out`
- Errors logged to: `pt_bench_[jobid].err`


Monitor your job:
```bash
squeue -u $USER
tail -f pt_bench_*.out
```

Note: Training metrics are shown only from rank 0 to avoid duplicate output in distributed mode.



