# Cross-Reenactment: 只换脸不换视角模式

## 概述

GUAVA 支持 **cross-reenactment**：将 target 视频中的动作和表情迁移到 source 图片的人物上。默认情况下渲染时会使用 target 视频的相机参数，导致输出视角随 target 头部运动而变化（包含 zoom、roll/pitch/yaw 等变换）。

通过 `--keep_source_cam` 参数，可以锁定相机为 source 图片的原始视角，实现 **只换脸（身份+动作迁移）而不改变视角** 的效果。

## 前置条件

1. **GUAVA 模型权重**：`assets/GUAVA/checkpoints/best_160000.pt`
2. **Target 视频已跟踪**：通过 EHM-Tracker 处理，生成包含 `videos_info.json`、`optim_tracking_ehm.pkl` 等文件的目录
3. **Source 图片已跟踪**：同样需要通过 EHM-Tracker 处理

## 使用方法

### 第一步：跟踪 Source 图片

如果 source 图片尚未跟踪，需要先用 EHM-Tracker 处理：

```bash
cd GUAVA/EHM-Tracker

PYTHONPATH='.' python -m src.tracking_single_image \
    --input_dir /path/to/source_image.jpeg \
    --output_dir /path/to/tracked_output \
    --save_vis_video
```

输出目录结构：
```
tracked_output/source_image.jpeg/
├── base_tracking.pkl
├── id_share_params.pkl
├── img_lmdb/
├── optim_tracking_ehm.pkl
├── optim_tracking_flame.pkl
├── videos_info.json
└── viz_tracking.mp4
```

### 第二步：运行 Cross-Reenactment

```bash
cd GUAVA

PYTHONPATH='.' python main/test.py \
    -d '0' \
    -m assets/GUAVA \
    -s outputs/cross_reenact \
    --data_path /path/to/tracked_target_video \
    --source_data_path /path/to/tracked_source_image \
    --skip_self_act \
    --render_cross_act \
    --keep_source_cam
```

### 参数说明

| 参数 | 说明 |
|------|------|
| `-d '0'` | GPU 设备 ID |
| `-m assets/GUAVA` | 模型路径 |
| `-s outputs/cross_reenact` | 输出保存路径 |
| `--data_path` | Target 视频的 EHM-Tracker 跟踪结果目录 |
| `--source_data_path` | Source 图片的 EHM-Tracker 跟踪结果目录 |
| `--skip_self_act` | 跳过 self-reenactment（只做 cross） |
| `--render_cross_act` | 启用 cross-reenactment |
| `--keep_source_cam` | **关键参数**：锁定相机为 source 视角，不跟随 target 运动 |
| `--bg_color 0.0` | 背景颜色（0.0=黑色，1.0=白色，默认 0.0） |

## 相机模式对比

`render_cross_set` 中有三种相机模式，由参数组合决定：

```
if use_aug:                    # --cross_aug
    cam = 自定义固定视角        # 使用 --fixed_yaw/pitch/zoom
elif args.keep_source_cam:     # --keep_source_cam
    cam = source 图片的相机     # 视角完全不变
else:                          # 默认
    cam = target 帧的相机       # 视角跟随 target 头部运动
```

| 模式 | 参数 | 视角行为 | 适用场景 |
|------|------|----------|----------|
| 默认 | 无额外参数 | 跟随 target 视频的头部运动 | 完整动作迁移 |
| **keep_source_cam** | `--keep_source_cam` | **固定为 source 图片视角** | **换脸不换视角** |
| cross_aug | `--cross_aug --fixed_yaw 0.3 ...` | 自定义固定偏移视角 | 数据增强 |

## 输出结构

```
outputs/cross_reenact/render_cross_act/
└── {source_name}/
    └── {source_name}_{target_name}/
        ├── source_image.png          # Source 身份图片
        ├── render/                   # 逐帧渲染结果
        │   ├── 00000.png
        │   ├── 00001.png
        │   └── ...
        └── {source}_{target}_video.mp4  # 合成视频 (30fps)
```

## 批量处理示例

对多张 source 图片和同一个 target 视频做 cross-reenactment：

```bash
TARGET=assets/phoenix14t/tracked_videos/02August_2010_Monday_heute-1243

for src in assets/tracked_templates/*/; do
    PYTHONPATH='.' python main/test.py \
        -d '0' -m assets/GUAVA -s outputs/cross_reenact \
        --data_path $TARGET \
        --source_data_path "$src" \
        --skip_self_act --render_cross_act --keep_source_cam
done
```

## 技术细节

### 数据流

1. **Source 图片** → EHM-Tracker → SMPL-X 体型参数 (shape, joints_offset, head_scale, hand_scale) + FLAME 面部形状 + 相机参数
2. **Target 视频** → EHM-Tracker → 逐帧 SMPL-X 姿态 + FLAME 表情参数
3. **Cross-reenactment**：
   - 用 source 的 DINOv2 特征生成 3D Gaussian Avatar
   - 用 `change_id_info()` 将 source 的体型参数注入 target 的姿态参数
   - 用 source 的相机参数（`--keep_source_cam`）渲染每一帧

### change_id_info 替换的参数

```python
target_info['smplx_coeffs']['shape']          = source  # 体型
target_info['smplx_coeffs']['joints_offset']  = source  # 关节偏移
target_info['smplx_coeffs']['head_scale']     = source  # 头部比例
target_info['smplx_coeffs']['hand_scale']     = source  # 手部比例
target_info['flame_coeffs']['shape_params']   = source  # 面部形状
```

动作和表情参数（body_pose, expression, jaw_pose 等）保持 target 的值。
