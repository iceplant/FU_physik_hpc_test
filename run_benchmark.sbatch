#!/bin/bash
#SBATCH --job-name=rocky_kamen_rubio_pt_benchmark
#SBATCH --ntasks=1                     # Number of individual tasks, usually 1 except when using MPI, etc.
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpus:4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=pt_bench_%j.out
#SBATCH --error=pt_bench_%j.err
#SBATCH --mem-per-cpu=100M              # Memory per cpu in MB (see also --mem) 
#SBATCH --mail-type=END                # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=kamenrur95@zedat.fu-berlin.de    # Email to which notifications will be sent 
 
# store job info in output file, if you want...
scontrol show job $SLURM_JOBID

# Load modules
module purge
module load Python/3.9.5-GCCcore-10.3.0
module load cuda/11.7

# Check if venv exists
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [ ! -d "${SCRIPT_DIR}/venv" ]; then
    echo "Error: Virtual environment not found at ${SCRIPT_DIR}/venv"
    echo "Please create the virtual environment first by following the README.md instructions"
    exit 1
fi

# Activate virtual environment
source "${SCRIPT_DIR}/venv/bin/activate"

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


