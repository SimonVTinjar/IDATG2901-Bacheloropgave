#!/bin/bash
#SBATCH --job-name=ref_editor_train
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=/cluster/work/%u/outputs/slurm-%j.out
#SBATCH --error=/cluster/work/%u/outputs/slurm-%j.err

# -----------------------------
# SETUP
# -----------------------------

mkdir -p /cluster/work/$USER/outputs

cd "$SLURM_SUBMIT_DIR"

echo "Jobb startet"
echo "Node: $(hostname)"
echo "Mappe: $(pwd)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# -----------------------------
# PYTHON ENV
# -----------------------------

source .venv/bin/activate

# -----------------------------
# TEST CUDA
# -----------------------------

python - << 'EOF'
import torch

print("Torch:", torch.__version__)
print("CUDA tilgjengelig:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
EOF

# -----------------------------
# START TRENING
# -----------------------------

python 04_train_reference_editor.py

echo "Jobb ferdig"