#!/bin/bash
#SBATCH --job-name=h2s_track
#SBATCH --output=logs/h2s_track_%A_%a.out
#SBATCH --error=logs/h2s_track_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --partition=nvidia
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zh3510@nyu.edu
#SBATCH --array=0-9

export PATH=/scratch/yf23/miniconda3/envs/GUAVA/bin:/usr/bin:/bin:/usr/sbin:/sbin

PART_ID=$(printf "%02d" $SLURM_ARRAY_TASK_ID)
PART_FILE="/scratch/yf23/zhewen/guava-aug/scripts/how2sign_splits/part_${PART_ID}.txt"
cd /scratch/yf23/zhewen/guava-aug/EHM-Tracker

echo "========================================"
echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Part file: $PART_FILE"
echo "========================================"

while IFS= read -r video_name; do
    [ -z "$video_name" ] && continue

    VIDEO_PATH="/scratch/yf23/zhewen/guava-aug/assets/how2sign/original_mp4/${video_name}.mp4"
    OUTPUT_DIR="/scratch/yf23/zhewen/guava-aug/assets/how2sign/tracked_nohand/${video_name}"

    if [ -d "$OUTPUT_DIR" ] && [ "$(ls -A $OUTPUT_DIR 2>/dev/null)" ]; then
        echo "Skipping $video_name (already exists)"
        continue
    fi

    echo "Processing: $video_name"

    /scratch/yf23/miniconda3/envs/GUAVA/bin/python \
        /scratch/yf23/zhewen/guava-aug/EHM-Tracker/tracking_video.py \
        -i "$VIDEO_PATH" \
        -o "$OUTPUT_DIR" \
        -v 0 \
        -n 1 \
        --not_check_hand

done < "$PART_FILE"

echo "Done with part $PART_ID"
