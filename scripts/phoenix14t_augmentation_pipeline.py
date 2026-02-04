#!/usr/bin/env python3
"""
Phoenix14T Video Augmentation Pipeline

This script orchestrates the complete pipeline:
1. Select 20 random videos from Phoenix14T test set
2. Convert .pt tensors to MP4 format
3. Run EHM-Tracker on MP4 videos
4. Generate 3 difficulty levels of view augmentation (60 total videos)
"""
import os
import sys
import random
import json
import subprocess
import argparse
from pathlib import Path
from datetime import datetime


# Default paths
DEFAULT_PHOENIX_ROOT = "/home/nyuair/zhewen/GUAVA/assets/phoenix14t/vid/videos/fullFrame-256x256px/test"
DEFAULT_OUTPUT_ROOT = "/home/nyuair/zhewen/GUAVA/outputs/phoenix14t_aug"
DEFAULT_MP4_DIR = "/home/nyuair/zhewen/GUAVA/assets/phoenix14t/vid/videos_mp4/test"
DEFAULT_TRACKED_DIR = "/home/nyuair/zhewen/GUAVA/assets/phoenix14t/tracked_videos"
DEFAULT_GUAVA_ROOT = "/home/nyuair/zhewen/GUAVA"
DEFAULT_EHM_ROOT = "/home/nyuair/zhewen/GUAVA/EHM-Tracker"


def select_random_videos(phoenix_root, num_videos=20, seed=42):
    """
    Select random videos from Phoenix14T test set.

    Args:
        phoenix_root: Root directory containing .pt video files
        num_videos: Number of videos to select
        seed: Random seed for reproducibility

    Returns:
        List of selected video filenames
    """
    print(f"\n{'='*80}")
    print("STEP 1: Selecting random videos")
    print(f"{'='*80}\n")

    pt_files = sorted([f for f in os.listdir(phoenix_root) if f.endswith('.pt')])
    print(f"Found {len(pt_files)} .pt videos in {phoenix_root}")

    random.seed(seed)
    selected = random.sample(pt_files, num_videos)
    selected.sort()

    print(f"\nSelected {len(selected)} videos (seed={seed}):")
    for i, video in enumerate(selected, 1):
        print(f"  {i:2d}. {video}")

    return selected


def convert_videos(selected_videos, phoenix_root, mp4_dir, guava_root):
    """
    Convert selected .pt videos to MP4 format.

    Args:
        selected_videos: List of .pt video filenames
        phoenix_root: Directory containing .pt files
        mp4_dir: Output directory for MP4 files
        guava_root: GUAVA root directory

    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*80}")
    print("STEP 2: Converting videos to MP4")
    print(f"{'='*80}\n")

    script_path = os.path.join(guava_root, 'scripts', 'convert_pt_to_mp4.py')

    cmd = [
        'python', script_path,
        '--input_dir', phoenix_root,
        '--output_dir', mp4_dir,
        '--video_list'
    ] + selected_videos

    print(f"Running: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=True)
        print("\n✓ Video conversion completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Video conversion failed: {e}")
        return False


def run_ehm_tracker(mp4_dir, tracked_dir, ehm_root, device='0'):
    """
    Run EHM-Tracker on converted MP4 videos.

    Args:
        mp4_dir: Directory containing MP4 videos
        tracked_dir: Output directory for tracking results
        ehm_root: EHM-Tracker root directory
        device: GPU device ID

    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*80}")
    print("STEP 3: Running EHM-Tracker")
    print(f"{'='*80}\n")

    cmd = [
        'python', 'tracking_video.py',
        '--in_root', mp4_dir,
        '--output_dir', tracked_dir,
        '--save_vis_video',
        '--check_hand_score', '0.0',
        '-n', '1',
        '-v', device
    ]

    print(f"Running: {' '.join(cmd)}")
    print(f"Working directory: {ehm_root}\n")

    env = os.environ.copy()
    env['PYTHONPATH'] = '.'

    try:
        result = subprocess.run(cmd, cwd=ehm_root, env=env, check=True)
        print("\n✓ EHM-Tracker completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ EHM-Tracker failed: {e}")
        return False


def run_augmentation(selected_videos, tracked_dir, output_root, guava_root,
                    difficulties=['easy', 'medium', 'hard'], base_seed=42, device='0'):
    """
    Run GUAVA augmentation for all videos and difficulties.

    Args:
        selected_videos: List of video filenames (with .pt extension)
        tracked_dir: Directory containing EHM-Tracker outputs
        output_root: Root directory for augmented videos
        guava_root: GUAVA root directory
        difficulties: List of difficulty levels
        base_seed: Base random seed
        device: GPU device ID

    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*80}")
    print("STEP 4: Running GUAVA augmentation")
    print(f"{'='*80}\n")

    # Remove .pt extension from video names
    video_names = [v.replace('.pt', '').replace('.mp4', '') for v in selected_videos]

    script_path = os.path.join(guava_root, 'scripts', 'run_guava_augmentation.py')

    cmd = [
        'python', script_path,
        '--video_list'
    ] + video_names + [
        '--tracked_root', tracked_dir,
        '--output_root', output_root,
        '--difficulties'
    ] + difficulties + [
        '--base_seed', str(base_seed),
        '--guava_root', guava_root,
        '--device', device
    ]

    print(f"Running: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=True)
        print("\n✓ GUAVA augmentation completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ GUAVA augmentation failed: {e}")
        return False


def validate_results(output_root, selected_videos, difficulties=['easy', 'medium', 'hard']):
    """
    Validate the augmentation results.

    Args:
        output_root: Root directory containing augmented videos
        selected_videos: List of original video filenames
        difficulties: List of difficulty levels

    Returns:
        Validation report dictionary
    """
    print(f"\n{'='*80}")
    print("STEP 5: Validating results")
    print(f"{'='*80}\n")

    video_names = [v.replace('.pt', '').replace('.mp4', '') for v in selected_videos]
    expected_total = len(video_names) * len(difficulties)

    report = {
        'expected_videos': expected_total,
        'found_videos': 0,
        'missing_videos': [],
        'difficulties': {}
    }

    for difficulty in difficulties:
        report['difficulties'][difficulty] = {
            'expected': len(video_names),
            'found': 0,
            'missing': []
        }

        for video_name in video_names:
            # Check for output video
            video_path = os.path.join(
                output_root, difficulty, video_name,
                'render_fixed_viewpoint', video_name,
                f'{video_name}_fixed_viewpoint_video.mp4'
            )

            if os.path.exists(video_path):
                report['difficulties'][difficulty]['found'] += 1
                report['found_videos'] += 1
            else:
                report['difficulties'][difficulty]['missing'].append(video_name)
                report['missing_videos'].append(f"{difficulty}/{video_name}")

    # Load augmentation parameters
    params_file = os.path.join(output_root, 'augmentation_params.json')
    if os.path.exists(params_file):
        with open(params_file, 'r') as f:
            params = json.load(f)
        report['params_file'] = params_file
        report['params_loaded'] = True
    else:
        report['params_file'] = None
        report['params_loaded'] = False

    # Print report
    print(f"Expected videos: {report['expected_videos']}")
    print(f"Found videos: {report['found_videos']}")
    print(f"Success rate: {report['found_videos']}/{report['expected_videos']} "
          f"({100*report['found_videos']/report['expected_videos']:.1f}%)\n")

    for difficulty in difficulties:
        diff_report = report['difficulties'][difficulty]
        print(f"{difficulty.upper()}: {diff_report['found']}/{diff_report['expected']} videos")
        if diff_report['missing']:
            print(f"  Missing: {', '.join(diff_report['missing'][:5])}"
                  f"{' ...' if len(diff_report['missing']) > 5 else ''}")

    if report['params_loaded']:
        print(f"\n✓ Parameters file found: {params_file}")
    else:
        print(f"\n✗ Parameters file not found: {params_file}")

    if report['missing_videos']:
        print(f"\n⚠ {len(report['missing_videos'])} videos missing")
    else:
        print(f"\n✓ All videos generated successfully!")

    return report


def main():
    parser = argparse.ArgumentParser(
        description='Phoenix14T Video Augmentation Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Run full pipeline
  python scripts/phoenix14t_augmentation_pipeline.py

  # Skip video selection and conversion (if already done)
  python scripts/phoenix14t_augmentation_pipeline.py --skip_conversion

  # Skip EHM-Tracker (if already done)
  python scripts/phoenix14t_augmentation_pipeline.py --skip_tracking

  # Only run augmentation
  python scripts/phoenix14t_augmentation_pipeline.py --skip_conversion --skip_tracking
        """
    )

    # Paths
    parser.add_argument('--phoenix_root', type=str, default=DEFAULT_PHOENIX_ROOT,
                        help='Directory containing Phoenix14T .pt videos')
    parser.add_argument('--mp4_dir', type=str, default=DEFAULT_MP4_DIR,
                        help='Output directory for MP4 videos')
    parser.add_argument('--tracked_dir', type=str, default=DEFAULT_TRACKED_DIR,
                        help='Output directory for EHM-Tracker results')
    parser.add_argument('--output_root', type=str, default=DEFAULT_OUTPUT_ROOT,
                        help='Root directory for augmented videos')
    parser.add_argument('--guava_root', type=str, default=DEFAULT_GUAVA_ROOT,
                        help='GUAVA root directory')
    parser.add_argument('--ehm_root', type=str, default=DEFAULT_EHM_ROOT,
                        help='EHM-Tracker root directory')

    # Parameters
    parser.add_argument('--num_videos', type=int, default=20,
                        help='Number of videos to select')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for video selection')
    parser.add_argument('--difficulties', type=str, nargs='+',
                        default=['easy', 'medium', 'hard'],
                        help='Difficulty levels to generate')
    parser.add_argument('--device', type=str, default='0',
                        help='GPU device ID')

    # Skip options
    parser.add_argument('--skip_selection', action='store_true',
                        help='Skip video selection (use existing selection)')
    parser.add_argument('--skip_conversion', action='store_true',
                        help='Skip video conversion (use existing MP4s)')
    parser.add_argument('--skip_tracking', action='store_true',
                        help='Skip EHM-Tracker (use existing tracking results)')
    parser.add_argument('--skip_augmentation', action='store_true',
                        help='Skip GUAVA augmentation (only validate)')
    parser.add_argument('--video_list_file', type=str,
                        help='JSON file with pre-selected video list')

    args = parser.parse_args()

    print(f"\n{'#'*80}")
    print("# Phoenix14T Video Augmentation Pipeline")
    print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*80}\n")

    # Step 1: Select videos
    if args.skip_selection and args.video_list_file:
        print(f"Loading video list from: {args.video_list_file}")
        with open(args.video_list_file, 'r') as f:
            selected_videos = json.load(f)
    elif args.skip_selection:
        print("⚠ Skipping video selection, but no video list provided!")
        print("Please provide --video_list_file or remove --skip_selection")
        return 1
    else:
        selected_videos = select_random_videos(
            args.phoenix_root, args.num_videos, args.seed
        )

        # Save selection
        selection_file = os.path.join(args.output_root, 'selected_videos.json')
        os.makedirs(args.output_root, exist_ok=True)
        with open(selection_file, 'w') as f:
            json.dump(selected_videos, f, indent=2)
        print(f"\nVideo selection saved to: {selection_file}")

    # Step 2: Convert videos
    if not args.skip_conversion:
        success = convert_videos(
            selected_videos, args.phoenix_root, args.mp4_dir, args.guava_root
        )
        if not success:
            print("\n✗ Pipeline failed at video conversion")
            return 1
    else:
        print("\n⊘ Skipping video conversion")

    # Step 3: Run EHM-Tracker
    if not args.skip_tracking:
        success = run_ehm_tracker(
            args.mp4_dir, args.tracked_dir, args.ehm_root, args.device
        )
        if not success:
            print("\n✗ Pipeline failed at EHM-Tracker")
            return 1
    else:
        print("\n⊘ Skipping EHM-Tracker")

    # Step 4: Run augmentation
    if not args.skip_augmentation:
        success = run_augmentation(
            selected_videos, args.tracked_dir, args.output_root, args.guava_root,
            args.difficulties, args.seed, args.device
        )
        if not success:
            print("\n✗ Pipeline failed at GUAVA augmentation")
            return 1
    else:
        print("\n⊘ Skipping GUAVA augmentation")

    # Step 5: Validate results
    report = validate_results(args.output_root, selected_videos, args.difficulties)

    # Save report
    report_file = os.path.join(args.output_root, 'validation_report.json')
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nValidation report saved to: {report_file}")

    print(f"\n{'#'*80}")
    print("# Pipeline Complete!")
    print(f"# Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*80}\n")

    return 0 if report['found_videos'] == report['expected_videos'] else 1


if __name__ == '__main__':
    sys.exit(main())
