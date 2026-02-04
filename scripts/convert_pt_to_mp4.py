#!/usr/bin/env python3
"""
Convert Phoenix14T .pt tensor videos to MP4 format for EHM-Tracker compatibility.
"""
import torch
import imageio
import numpy as np
import os
import argparse
from pathlib import Path
from tqdm import tqdm


def convert_pt_to_mp4(pt_path, mp4_path, fps=25):
    """
    Convert a .pt tensor video to MP4 format.

    Args:
        pt_path: Path to input .pt file with shape [T, 3, H, W], dtype uint8
        mp4_path: Path to output .mp4 file
        fps: Frames per second for output video (default: 25)
    """
    # Load tensor: [T, 3, H, W], uint8
    video_tensor = torch.load(pt_path)

    # Verify expected format
    assert video_tensor.dtype == torch.uint8, f"Expected uint8, got {video_tensor.dtype}"
    assert len(video_tensor.shape) == 4, f"Expected 4D tensor, got shape {video_tensor.shape}"
    assert video_tensor.shape[1] == 3, f"Expected 3 channels, got {video_tensor.shape[1]}"

    # Convert to numpy: [T, H, W, 3]
    video_np = video_tensor.permute(0, 2, 3, 1).numpy()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(mp4_path), exist_ok=True)

    # Save as MP4 with H.264 codec
    imageio.mimsave(mp4_path, video_np, fps=fps, codec='libx264', quality=8)

    return video_np.shape[0]  # Return number of frames


def batch_convert(input_dir, output_dir, video_list=None, fps=25):
    """
    Batch convert multiple .pt videos to MP4.

    Args:
        input_dir: Directory containing .pt files
        output_dir: Directory to save .mp4 files
        video_list: Optional list of specific video filenames to convert
        fps: Frames per second for output videos
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get list of files to convert
    if video_list:
        pt_files = [f for f in video_list if f.endswith('.pt')]
    else:
        pt_files = sorted([f.name for f in input_path.glob('*.pt')])

    print(f"Converting {len(pt_files)} videos from {input_dir} to {output_dir}")

    results = []
    for pt_file in tqdm(pt_files, desc="Converting videos"):
        pt_path = input_path / pt_file
        mp4_file = pt_file.replace('.pt', '.mp4')
        mp4_path = output_path / mp4_file

        try:
            num_frames = convert_pt_to_mp4(str(pt_path), str(mp4_path), fps=fps)
            results.append({
                'filename': pt_file,
                'status': 'success',
                'num_frames': num_frames,
                'output': str(mp4_path)
            })
            print(f"✓ {pt_file}: {num_frames} frames -> {mp4_path}")
        except Exception as e:
            results.append({
                'filename': pt_file,
                'status': 'failed',
                'error': str(e)
            })
            print(f"✗ {pt_file}: Failed - {e}")

    # Summary
    success_count = sum(1 for r in results if r['status'] == 'success')
    print(f"\nConversion complete: {success_count}/{len(pt_files)} successful")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Phoenix14T .pt videos to MP4')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing .pt video files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save .mp4 video files')
    parser.add_argument('--video_list', type=str, nargs='+',
                        help='Optional list of specific video filenames to convert')
    parser.add_argument('--fps', type=int, default=25,
                        help='Frames per second for output videos (default: 25)')

    args = parser.parse_args()

    batch_convert(args.input_dir, args.output_dir, args.video_list, args.fps)
