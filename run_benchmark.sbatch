#!/bin/bash
#SBATCH --job-name=pt_benchmark
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=pt_bench_%j.out
#SBATCH --error=pt_bench_%j.err

# Load modules
module purge
module load Python/3.9.5-GCCcore-10.3.0
module load cuda/11.7

# Activate virtual environment
source /srv/data/$USER/hpc_test/slurm_test_job/venv/bin/activate

# Launch with torchrun
srun torchrun \
    --nproc_per_node=$SLURM_GPUS \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=$(hostname) \
    --master_port=29500 \
    benchmark_train.py \
    --epochs 10 \
    --batch_size 128 \
    --num_samples 100000 \
    --hidden_dim 256


