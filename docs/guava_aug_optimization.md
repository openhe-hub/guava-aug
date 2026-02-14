# GUAVA Augmentation Optimization

## Overview

When generating multiple augmented views of the same tracked video, the original subprocess-based approach reloads the model and recomputes everything from scratch for each parameter set. The optimized batch mode (`--batch`) loads once and only repeats the camera-dependent rendering step, achieving **5-22x speedup** depending on the number of parameter sets.

## Benchmark Environment

- **GPU**: NVIDIA RTX A6000 (48GB)
- **Video**: `01October_2012_Monday_tagesschau-5370` (137 frames)
- **PyTorch**: 2.2.0+cu118
- **Date**: 2026-02-14

## Pipeline Comparison

### Old Mode (subprocess): Each param set spawns a new process

```
Param set 1 (easy)            Param set 2 (medium)          Param set 3 (hard)
┌─────────────────────┐      ┌─────────────────────┐      ┌─────────────────────┐
│ Load model    ~80s  │      │ Load model    ~80s  │      │ Load model    ~80s  │
│ Load data      ~1s  │      │ Load data      ~1s  │      │ Load data      ~1s  │
│ Source infer   ~1s  │      │ Source infer   ~1s  │      │ Source infer   ~1s  │
│ Gaussian init  ~0s  │      │ Gaussian init  ~0s  │      │ Gaussian init  ~0s  │
│ ┌─────────────────┐ │      │ ┌─────────────────┐ │      │ ┌─────────────────┐ │
│ │ Per-frame ×137: │ │      │ │ Per-frame ×137: │ │      │ │ Per-frame ×137: │ │
│ │  load_target    │ │      │ │  load_target    │ │      │ │  load_target    │ │
│ │  deform         │ │      │ │  deform         │ │      │ │  deform         │ │
│ │  render         │ │      │ │  render         │ │      │ │  render         │ │
│ │  inpaint bg     │ │      │ │  inpaint bg     │ │      │ │  inpaint bg     │ │
│ │  composite      │ │      │ │  composite      │ │      │ │  composite      │ │
│ └──── ~120s ──────┘ │      │ └──── ~120s ──────┘ │      │ └──── ~120s ──────┘ │
│ Write MP4           │      │ Write MP4           │      │ Write MP4           │
└─────────────────────┘      └─────────────────────┘      └─────────────────────┘
      ~200s                        ~200s                        ~200s

Total: ~600s (10 min)
```

### New Mode (batch + precompute): Load once, render N times

```
┌══════════════════════════════════════════════════════════════════┐
│                     One-time Initialization                      │
│  Load model              ~80s                                    │
│  Load data                ~0s                                    │
│  Source inference          ~1s                                   │
│  Gaussian init             ~0s                                   │
│                                                                  │
│  ┌────────────────────────────────────────────┐                  │
│  │ Precompute per-frame ×137:          ~10s   │                  │
│  │   load_target  ──→ target_info[]           │  camera-agnostic │
│  │   deform       ──→ deform_assets[]         │  motion only     │
│  │   inpaint bg   ──→ gt_backgrounds[]        │  image only      │
│  └────────────────────────────────────────────┘                  │
│                          ~90s                                    │
└══════════════════════════════════════════════════════════════════┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
   │ easy        │    │ medium      │    │ hard        │
   │             │    │             │    │             │
   │ Gen cam     │    │ Gen cam     │    │ Gen cam     │
   │ ┌─────────┐ │    │ ┌─────────┐ │    │ ┌─────────┐ │
   │ │ ×137:   │ │    │ │ ×137:   │ │    │ │ ×137:   │ │
   │ │ render  │ │    │ │ render  │ │    │ │ render  │ │
   │ │ compose │ │    │ │ compose │ │    │ │ compose │ │
   │ └── ~7s ─┘  │    │ └── ~7s ─┘  │     │ └── ~8s ─┘ │
   │ Write MP4   │    │ Write MP4   │    │ Write MP4   │
   └── ~9s ──────┘    └── ~7s ──────┘    └── ~8s ──────┘

Total: ~115s (excl. model load) / ~175s (incl. model load)
```

## Key Insight: What Depends on Camera Parameters?

| Operation             | Identity | Motion | Camera | Reusable?          |
|-----------------------|:--------:|:------:|:------:|--------------------|
| Model loading         |          |        |        | Global (once)      |
| Source inference      |    ✓     |        |        | Per video (once)   |
| load_target (frames)  |          |   ✓    |        | Per video (once)   |
| deform (Gaussian)     |    ✓     |   ✓    |        | Per video (once)   |
| inpaint (background)  |          |        |        | Per video (once)   |
| **generate_cam_params** |        |        |   ✓    | **Per param set**  |
| **render (rasterize)**  |        |        |   ✓    | **Per param set**  |
| **alpha composite**     |        |        |   ✓    | **Per param set**  |

Only the last 3 steps truly depend on camera parameters (~0.06s/frame). Everything else is reusable.

## Measured Timings (RTX A6000, 137 frames)

### Per-stage Breakdown

| Stage                        | Time   | Frequency       |
|------------------------------|--------|-----------------|
| Model loading                | 80.5s  | Once            |
| Dataset loading              | 0.0s   | Once per video  |
| Source inference             | 0.3s   | Once per video  |
| Gaussian setup               | 0.0s   | Once per video  |
| Precompute (deform + bg)     | 10.0s  | Once per video  |
| **Render per param set**     | **~8s** | **Per param set** |

### Marginal Cost Comparison

| N (param sets) | Subprocess  | Batch     | Speedup |
|:--------------:|:-----------:|:---------:|:-------:|
| 1              | 200s        | 98s       | 2.0x    |
| 3              | 600s        | 115s      | 5.2x    |
| 10             | 2000s       | 170s      | 11.8x   |
| 50             | 10000s      | 490s      | 20.4x   |
| 100            | 20000s      | 890s      | 22.5x   |

## Usage

### Batch Mode (recommended for multiple params)

```bash
python scripts/run_guava_augmentation.py --batch \
  --video_list VIDEO_NAME_1 VIDEO_NAME_2 \
  --tracked_root /path/to/tracked \
  --output_root /path/to/output \
  --difficulties easy medium hard \
  --device 0
```

### Subprocess Mode (legacy, simple)

```bash
python scripts/run_guava_augmentation.py \
  --video_list VIDEO_NAME \
  --tracked_root /path/to/tracked \
  --output_root /path/to/output \
  --difficulties easy medium hard \
  --device 0
```

### Options

| Flag                | Description                                      |
|---------------------|--------------------------------------------------|
| `--batch`           | Enable fast batch mode (load model once)         |
| `--video_list`      | Space-separated video names                      |
| `--video_list_file` | File with one video name per line                |
| `--tracked_root`    | Root dir of EHM-Tracker outputs                  |
| `--output_root`     | Root dir for augmented outputs                   |
| `--difficulties`    | Difficulty levels (default: easy medium hard)    |
| `--device`          | GPU device ID (default: 0)                       |
| `--part_id`         | SLURM array task ID for per-shard param files    |
| `--save_frame_png`  | Also save per-frame PNGs (default: MP4 only)     |

## Output Structure

```
{output_root}/
├── augmentation_params.json
├── easy/{video_name}/render_fixed_viewpoint/{video_name}/
│   └── {video_name}_fixed_viewpoint_video.mp4
├── medium/{video_name}/render_fixed_viewpoint/{video_name}/
│   └── {video_name}_fixed_viewpoint_video.mp4
└── hard/{video_name}/render_fixed_viewpoint/{video_name}/
    └── {video_name}_fixed_viewpoint_video.mp4
```
