#!/usr/bin/env python3
"""
Run GUAVA view augmentation with different difficulty levels.
"""
import subprocess
import random
import json
import os
import argparse
from pathlib import Path


# Augmentation difficulty configurations
DIFFICULTY_CONFIGS = {
    'easy': {
        'yaw_range': (-0.15, 0.15),      # ±8.6°
        'pitch_range': (-0.10, 0.10),    # ±5.7°
        'zoom_range': (0.90, 1.10)       # ±10% distance
    },
    'medium': {
        'yaw_range': (-0.35, 0.35),      # ±20°
        'pitch_range': (-0.25, 0.25),    # ±14.3°
        'zoom_range': (0.75, 1.30)       # 25%-30% distance
    },
    'hard': {
        'yaw_range': (-0.52, 0.52),      # ±30°
        'pitch_range': (-0.40, 0.40),    # ±23°
        'zoom_range': (0.60, 1.50)       # 40%-50% distance
    }
}


def generate_random_params(difficulty, seed):
    """
    Generate random augmentation parameters for a given difficulty level.

    Args:
        difficulty: 'easy', 'medium', or 'hard'
        seed: Random seed for reproducibility

    Returns:
        Tuple of (yaw, pitch, zoom) in radians and scale factor
    """
    random.seed(seed)
    config = DIFFICULTY_CONFIGS[difficulty]

    yaw = random.uniform(*config['yaw_range'])
    pitch = random.uniform(*config['pitch_range'])
    zoom = random.uniform(*config['zoom_range'])

    return yaw, pitch, zoom


def run_guava_augmentation(video_name, tracked_path, output_dir, difficulty, seed,
                           guava_root='/home/nyuair/zhewen/GUAVA',
                           model_path='assets/GUAVA',
                           device='0'):
    """
    Run GUAVA augmentation for a single video.

    Args:
        video_name: Name of the video (for logging)
        tracked_path: Path to EHM-Tracker output directory
        output_dir: Output directory for augmented video
        difficulty: 'easy', 'medium', or 'hard'
        seed: Random seed for parameter generation
        guava_root: Root directory of GUAVA project
        model_path: Path to GUAVA model checkpoint
        device: GPU device ID

    Returns:
        Dictionary with augmentation parameters and status
    """
    # Generate random parameters
    yaw, pitch, zoom = generate_random_params(difficulty, seed)

    # Construct GUAVA command using conda environment
    conda_python = '/home/nyuair/anaconda3/envs/GUAVA/bin/python'
    cmd = [
        conda_python, 'main/test.py',
        '-d', str(device),
        '-m', model_path,
        '-s', output_dir,
        '--data_path', tracked_path,
        '--skip_self_act',
        '--render_fixed_viewpoint',
        '--fixed_yaw', str(yaw),
        '--fixed_pitch', str(pitch),
        '--fixed_zoom', str(zoom),
        '--bg_color', '1.0'  # White background
    ]

    print(f"\n{'='*80}")
    print(f"Processing: {video_name} ({difficulty})")
    print(f"Parameters: yaw={yaw:.4f}, pitch={pitch:.4f}, zoom={zoom:.4f}")
    print(f"Seed: {seed}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")

    # Run GUAVA
    try:
        # Set PYTHONPATH to include GUAVA root
        env = os.environ.copy()
        env['PYTHONPATH'] = guava_root

        result = subprocess.run(cmd, cwd=guava_root, check=True,
                               capture_output=True, text=True, env=env)
        status = 'success'
        error = None
        print(f"✓ {video_name} ({difficulty}): Success")
    except subprocess.CalledProcessError as e:
        status = 'failed'
        error = e.stderr
        print(f"✗ {video_name} ({difficulty}): Failed")
        print(f"Error: {error}")

    # Create parameter log
    params_log = {
        'video_name': video_name,
        'difficulty': difficulty,
        'seed': seed,
        'yaw': yaw,
        'pitch': pitch,
        'zoom': zoom,
        'status': status,
        'error': error,
        'tracked_path': tracked_path,
        'output_dir': output_dir
    }

    return params_log


def batch_augmentation(video_list, tracked_root, output_root, difficulties=['easy', 'medium', 'hard'],
                      base_seed=42, guava_root='/home/nyuair/zhewen/GUAVA',
                      model_path='assets/GUAVA', device='0'):
    """
    Run batch augmentation for multiple videos and difficulties.

    Args:
        video_list: List of video names (without extension)
        tracked_root: Root directory containing EHM-Tracker outputs
        output_root: Root directory for augmented videos
        difficulties: List of difficulty levels to generate
        base_seed: Base random seed (will be incremented for each video/difficulty)
        guava_root: Root directory of GUAVA project
        model_path: Path to GUAVA model checkpoint
        device: GPU device ID

    Returns:
        Dictionary mapping video names to their augmentation parameters
    """
    all_params = {}

    for idx, video_name in enumerate(video_list):
        all_params[video_name] = {}

        for diff_idx, difficulty in enumerate(difficulties):
            # Generate unique seed for reproducibility
            seed = base_seed + idx * len(difficulties) + diff_idx

            # Paths
            tracked_path = os.path.join(tracked_root, video_name)
            output_dir = os.path.join(output_root, difficulty, video_name)

            # Check if video already exists
            video_file = os.path.join(output_dir, 'render_fixed_viewpoint', video_name,
                                     f'{video_name}_fixed_viewpoint_video.mp4')
            if os.path.exists(video_file):
                print(f"⊙ {video_name} ({difficulty}): Already exists, skipping")
                params = {
                    'video_name': video_name,
                    'difficulty': difficulty,
                    'seed': seed,
                    'status': 'skipped',
                    'output_dir': output_dir
                }
            else:
                # Run augmentation
                params = run_guava_augmentation(
                    video_name, tracked_path, output_dir, difficulty, seed,
                    guava_root, model_path, device
                )

            all_params[video_name][difficulty] = params

    # Save all parameters to JSON
    params_file = os.path.join(output_root, 'augmentation_params.json')
    os.makedirs(output_root, exist_ok=True)
    with open(params_file, 'w') as f:
        json.dump(all_params, f, indent=2)

    print(f"\nAll parameters saved to: {params_file}")

    # Summary
    total = len(video_list) * len(difficulties)
    success = sum(1 for v in all_params.values()
                  for d in v.values() if d['status'] == 'success')
    print(f"\nAugmentation complete: {success}/{total} successful")

    return all_params


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run GUAVA view augmentation')
    parser.add_argument('--video_list', type=str, nargs='+', required=True,
                        help='List of video names (without extension)')
    parser.add_argument('--tracked_root', type=str, required=True,
                        help='Root directory containing EHM-Tracker outputs')
    parser.add_argument('--output_root', type=str, required=True,
                        help='Root directory for augmented videos')
    parser.add_argument('--difficulties', type=str, nargs='+',
                        default=['easy', 'medium', 'hard'],
                        help='Difficulty levels to generate')
    parser.add_argument('--base_seed', type=int, default=42,
                        help='Base random seed for reproducibility')
    parser.add_argument('--guava_root', type=str,
                        default='/home/nyuair/zhewen/GUAVA',
                        help='Root directory of GUAVA project')
    parser.add_argument('--model_path', type=str, default='assets/GUAVA',
                        help='Path to GUAVA model checkpoint')
    parser.add_argument('--device', type=str, default='0',
                        help='GPU device ID')

    args = parser.parse_args()

    batch_augmentation(
        args.video_list,
        args.tracked_root,
        args.output_root,
        args.difficulties,
        args.base_seed,
        args.guava_root,
        args.model_path,
        args.device
    )
