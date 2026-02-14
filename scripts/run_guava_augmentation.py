#!/usr/bin/env python3
"""
Run GUAVA view augmentation with different difficulty levels.

Modes:
  subprocess (default): Spawns a separate python process per (video, difficulty).
                        Simple but slow — reloads model every time.
  batch (--batch):      Loads model once, renders all param sets in-process.
                        ~2.5-3x faster for multiple params on the same video.
"""
import subprocess
import random
import json
import os
import sys
import time
import copy
import argparse
import hashlib
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

    # Construct GUAVA command using current python (from activated conda env)
    cmd = [
        sys.executable, 'main/test.py',
        '-d', str(device),
        '-m', model_path,
        '-s', output_dir,
        '--data_path', tracked_path,
        '--skip_self_act',
        '--render_fixed_viewpoint',
        '--fixed_yaw', str(yaw),
        '--fixed_pitch', str(pitch),
        '--fixed_zoom', str(zoom),
        '--skip_frame_png',
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
                      model_path='assets/GUAVA', device='0', part_id=None):
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
            # Generate unique seed using deterministic hash of video name
            # This ensures the same video always gets the same seed regardless of
            # which partition/shard it appears in
            video_hash = int(hashlib.md5(video_name.encode()).hexdigest()[:8], 16)
            seed = base_seed + (video_hash % 100000) * len(difficulties) + diff_idx

            # Paths
            # Compatible with both tracked_root/{video} and tracked_root/{video}/{video}
            tracked_path = os.path.join(tracked_root, video_name)
            nested_tracked_path = os.path.join(tracked_root, video_name, video_name)
            if os.path.exists(os.path.join(nested_tracked_path, 'videos_info.json')):
                tracked_path = nested_tracked_path
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

    # Save all parameters to JSON (per-partition to avoid overwrites in array jobs)
    if part_id is not None:
        params_filename = f'augmentation_params_part_{part_id}.json'
    else:
        params_filename = 'augmentation_params.json'
    params_file = os.path.join(output_root, params_filename)
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


def _resolve_tracked_path(tracked_root, video_name):
    """Resolve tracked data path, supporting both flat and nested layouts."""
    tracked_path = os.path.join(tracked_root, video_name)
    nested = os.path.join(tracked_root, video_name, video_name)
    if os.path.exists(os.path.join(nested, 'videos_info.json')):
        return nested
    return tracked_path


def _output_video_path(output_root, difficulty, video_name):
    """Return the expected output MP4 path for a (difficulty, video) pair."""
    return os.path.join(output_root, difficulty, video_name,
                        'render_fixed_viewpoint', video_name,
                        f'{video_name}_fixed_viewpoint_video.mp4')


def batch_augmentation_fast(video_list, tracked_root, output_root,
                            difficulties=('easy', 'medium', 'hard'),
                            base_seed=42, guava_root=None,
                            model_path='assets/GUAVA', device='0',
                            part_id=None, skip_frame_png=True):
    """
    Fast batch augmentation: load model once, render N param sets per video.

    For each video the expensive steps (model load, source inference, gaussian
    setup) happen once.  Only the per-frame rendering loop is repeated for each
    difficulty / param set.
    """
    import torch
    import lightning
    import numpy as np
    import imageio
    from tqdm import tqdm
    from omegaconf import OmegaConf

    if guava_root is None:
        guava_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(guava_root)
    sys.path.insert(0, guava_root)

    from dataset import TrackedData_infer
    from models.UbodyAvatar import (Ubody_Gaussian_inferer, Ubody_Gaussian,
                                    GaussianRenderer)
    from utils.general_utils import (ConfigDict, find_pt_file, add_extra_cfgs,
                                     to8b)
    from utils.camera_utils import generate_novel_view_poses_fixed
    from main.test import inpaint_person_region

    torch.set_float32_matmul_precision('high')

    # --- Stage 1: load model (once) ---
    t0 = time.time()
    model_config_path = os.path.join(model_path, 'config.yaml')
    meta_cfg = ConfigDict(model_config_path=model_config_path)
    meta_cfg = add_extra_cfgs(meta_cfg)
    meta_cfg = copy.deepcopy(meta_cfg)
    lightning.fabric.seed_everything(10)
    device_str = f'cuda:{device}'

    infer_model = Ubody_Gaussian_inferer(meta_cfg.MODEL)
    infer_model.to(device_str).eval()
    render_model = GaussianRenderer(meta_cfg.MODEL)
    render_model.to(device_str).eval()

    ckpt_dir = os.path.join(model_path, 'checkpoints')
    ckpt = find_pt_file(ckpt_dir, 'best') or find_pt_file(ckpt_dir, 'latest')
    state = torch.load(ckpt, map_location='cpu', weights_only=True)
    infer_model.load_state_dict(state['model'], strict=False)
    render_model.load_state_dict(state['render_model'], strict=False)
    print(f"[init] Model loaded in {time.time()-t0:.1f}s  ({ckpt})")

    all_params = {}
    total_tasks = len(video_list) * len(difficulties)
    done = 0

    with torch.no_grad():
        for vidx, video_name in enumerate(video_list):
            all_params[video_name] = {}

            tracked_path = _resolve_tracked_path(tracked_root, video_name)

            # --- Stage 2: load dataset for this video (once per video) ---
            t0 = time.time()
            cfg = copy.deepcopy(meta_cfg)
            OmegaConf.set_readonly(cfg['DATASET'], False)
            cfg['DATASET']['data_path'] = tracked_path
            dataset = TrackedData_infer(cfg=cfg, split='test',
                                        device=device_str, test_full=True)
            video_id = list(dataset.videos_info.keys())[0]
            frames = dataset.videos_info[video_id]['frames_keys']
            print(f"\n[video {vidx+1}/{len(video_list)}] {video_name}  "
                  f"({len(frames)} frames, loaded in {time.time()-t0:.1f}s)")

            # --- Stage 3: source inference + gaussian setup (once per video) ---
            source_info = dataset._load_source_info(video_id)
            vertex_gs_dict, up_point_gs_dict, _ = infer_model(source_info)
            ubody_gaussians = Ubody_Gaussian(cfg.MODEL, vertex_gs_dict,
                                             up_point_gs_dict, pruning=True)
            ubody_gaussians.init_ehm(infer_model.ehm)
            ubody_gaussians.eval()

            # Pre-load all target_info (shared across param sets)
            all_target_info = []
            for frame in frames:
                all_target_info.append(dataset._load_target_info(video_id, frame))

            # Pre-compute deform assets + background (shared across param sets)
            all_deform_assets = []
            all_gt_backgrounds = []
            for idx, target_info in enumerate(tqdm(all_target_info, desc="  Precompute")):
                deform = ubody_gaussians(target_info)
                all_deform_assets.append(deform)
                gt_image = target_info['image'][0]
                gt_mask = target_info['mask'][0]
                all_gt_backgrounds.append(inpaint_person_region(gt_image, gt_mask))

            # --- Stage 4: render each param set (only camera + render changes) ---
            for diff_idx, difficulty in enumerate(difficulties):
                video_hash = int(hashlib.md5(video_name.encode()).hexdigest()[:8], 16)
                seed = base_seed + (video_hash % 100000) * len(difficulties) + diff_idx
                yaw, pitch, zoom = generate_random_params(difficulty, seed)

                out_dir = os.path.join(output_root, difficulty, video_name)
                mp4_path = _output_video_path(output_root, difficulty, video_name)

                if os.path.exists(mp4_path):
                    print(f"  ⊙ {difficulty}: exists, skipping")
                    all_params[video_name][difficulty] = {
                        'video_name': video_name, 'difficulty': difficulty,
                        'seed': seed, 'status': 'skipped', 'output_dir': out_dir
                    }
                    done += 1
                    continue

                t_render = time.time()
                novel_cam_params = generate_novel_view_poses_fixed(
                    source_info, image_size=dataset.image_size,
                    tanfov=dataset.tanfov, num_keyframes=120,
                    fixed_yaw=yaw, fixed_pitch=pitch, fixed_zoom=zoom
                )

                render_out_dir = os.path.join(out_dir, 'render_fixed_viewpoint', video_name)
                os.makedirs(render_out_dir, exist_ok=True)
                if not skip_frame_png:
                    png_dir = os.path.join(render_out_dir, 'render')
                    os.makedirs(png_dir, exist_ok=True)

                rendering_imgs = []
                for idx in range(len(frames)):
                    render_cam = novel_cam_params[idx % 120]
                    render_results = render_model(
                        all_deform_assets[idx], render_cam,
                        bg=0.0, compute_alpha=True, composite_bg=False
                    )
                    render_image = render_results['renders'][0]
                    render_alpha = render_results['alpha_images'][0].clamp(0.0, 1.0)
                    final = render_image * render_alpha + all_gt_backgrounds[idx] * (1 - render_alpha)
                    if not skip_frame_png:
                        import torchvision
                        torchvision.utils.save_image(final, os.path.join(png_dir, f'{idx:05d}.png'))
                    rendering_imgs.append(to8b(final.detach().cpu().numpy()))

                rendering_imgs = np.stack(rendering_imgs, 0).transpose(0, 2, 3, 1)
                imageio.mimwrite(
                    os.path.join(render_out_dir, f'{video_name}_fixed_viewpoint_video.mp4'),
                    rendering_imgs, fps=30, quality=8
                )

                elapsed = time.time() - t_render
                done += 1
                print(f"  ✓ {difficulty} (yaw={yaw:.3f} pitch={pitch:.3f} zoom={zoom:.2f})  "
                      f"{elapsed:.1f}s  [{done}/{total_tasks}]")

                all_params[video_name][difficulty] = {
                    'video_name': video_name, 'difficulty': difficulty,
                    'seed': seed, 'yaw': yaw, 'pitch': pitch, 'zoom': zoom,
                    'render_time': elapsed, 'status': 'success',
                    'tracked_path': tracked_path, 'output_dir': out_dir
                }

            dataset._lmdb_engine.close()

    # Save params
    if part_id is not None:
        fname = f'augmentation_params_part_{part_id}.json'
    else:
        fname = 'augmentation_params.json'
    params_file = os.path.join(output_root, fname)
    os.makedirs(output_root, exist_ok=True)
    with open(params_file, 'w') as f:
        json.dump(all_params, f, indent=2)

    success = sum(1 for v in all_params.values()
                  for d in v.values() if d['status'] == 'success')
    print(f"\nDone: {success}/{total_tasks} successful.  Params: {params_file}")
    return all_params


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run GUAVA view augmentation')
    parser.add_argument('--video_list', type=str, nargs='+', required=False,
                        help='List of video names (without extension)')
    parser.add_argument('--video_list_file', type=str, default=None,
                        help='File containing video names (one per line)')
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
                        default=None,
                        help='Root directory of GUAVA project (auto-detected if not set)')
    parser.add_argument('--model_path', type=str, default='assets/GUAVA',
                        help='Path to GUAVA model checkpoint')
    parser.add_argument('--device', type=str, default='0',
                        help='GPU device ID')
    parser.add_argument('--part_id', type=int, default=None,
                        help='Partition ID (e.g. SLURM_ARRAY_TASK_ID) for per-shard params file')
    parser.add_argument('--batch', action='store_true', default=False,
                        help='Use fast batch mode: load model once, render all params in-process')
    parser.add_argument('--skip_frame_png', action='store_true', default=True,
                        help='Skip writing per-frame PNG files (default: True)')
    parser.add_argument('--save_frame_png', action='store_true', default=False,
                        help='Save per-frame PNG files (overrides --skip_frame_png)')

    args = parser.parse_args()

    # Get video list from file or command line
    if args.video_list_file:
        with open(args.video_list_file, 'r') as f:
            video_list = [line.strip() for line in f if line.strip()]
    elif args.video_list:
        video_list = args.video_list
    else:
        raise ValueError("Must provide either --video_list or --video_list_file")

    guava_root = args.guava_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if args.batch:
        batch_augmentation_fast(
            video_list,
            args.tracked_root,
            args.output_root,
            args.difficulties,
            args.base_seed,
            guava_root,
            args.model_path,
            args.device,
            args.part_id,
            skip_frame_png=not args.save_frame_png,
        )
    else:
        batch_augmentation(
            video_list,
            args.tracked_root,
            args.output_root,
            args.difficulties,
            args.base_seed,
            guava_root,
            args.model_path,
            args.device,
            args.part_id
        )
