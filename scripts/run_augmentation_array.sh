#!/bin/bash
#SBATCH --job-name=guava-aug
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-9
#SBATCH --output=logs/augmentation_%A_%a.out
#SBATCH --error=logs/augmentation_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zh3510@nyu.edu

# ---- Configuration (edit these) ----
VIDEO_LIST_FILE="${VIDEO_LIST_FILE:-video_lists/part_${SLURM_ARRAY_TASK_ID}.txt}"
TRACKED_ROOT="${TRACKED_ROOT:-/scratch/yf23/zhewen/guava-aug/assets/phoenix14t/tracked_nohand}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/yf23/zhewen/guava-aug/assets/phoenix14t/augmented_nohand}"
GUAVA_ROOT="/scratch/yf23/zhewen/guava-aug"
MODEL_PATH="assets/GUAVA"
# -------------------------------------

# Environment setup
source /share/apps/NYUAD5/Modules/current/init/profile.sh
module load cuda/11.8.0

eval "$(/scratch/yf23/miniconda3/bin/conda shell.bash hook)"
conda activate GUAVA

export PYTHONPATH="${GUAVA_ROOT}"
export TORCH_HOME="/scratch/yf23/.torch"

mkdir -p logs

echo "=== SLURM Array Task ${SLURM_ARRAY_TASK_ID} / Job ${SLURM_JOB_ID} ==="
echo "Video list: ${VIDEO_LIST_FILE}"
echo "Output root: ${OUTPUT_ROOT}"

python "${GUAVA_ROOT}/scripts/run_guava_augmentation.py" \
    --video_list_file "${VIDEO_LIST_FILE}" \
    --tracked_root "${TRACKED_ROOT}" \
    --output_root "${OUTPUT_ROOT}" \
    --guava_root "${GUAVA_ROOT}" \
    --model_path "${MODEL_PATH}" \
    --difficulties easy \
    --part_id $SLURM_ARRAY_TASK_ID
