# GUAVA视角增强功能

本文档说明了从guava-aug项目同步过来的视角增强功能。

## 新增功能

### 1. 动态视角 + Zoom（相机距离变化）

**功能描述**: 在原有的动态视角基础上，增加了相机距离的周期性变化，实现zoom in/out效果。

**使用方法**:
```bash
python main/test.py \
  -d '0' \
  -m assets/GUAVA \
  -s outputs/example_aug \
  --data_path assets/example/tracked_video/6gvP8f5WQyo__056 \
  --skip_self_act \
  --render_dynamic_novel_views_with_zoom \
  --zoom_range 0.3
```

**参数说明**:
- `--render_dynamic_novel_views_with_zoom`: 启用zoom功能
- `--zoom_range`: 相机距离变化范围（默认0.3，表示0.7x到1.3x的基础半径）

**输出**: `{output_dir}/render_dyn_novel_views_zoom/{video_id}/{video_id}_dynamic_novel_views_with_zoom_video.mp4`

### 2. 固定视角渲染（数据增强）

**功能描述**: 使用固定的相机角度和缩放比例渲染整个序列，适用于数据增强场景。

**使用方法**:
```bash
python main/test.py \
  -d '0' \
  -m assets/GUAVA \
  -s outputs/example_aug \
  --data_path assets/example/tracked_video/6gvP8f5WQyo__056 \
  --skip_self_act \
  --render_fixed_viewpoint \
  --fixed_yaw 0.3 \
  --fixed_pitch 0.2 \
  --fixed_zoom 1.3
```

**参数说明**:
- `--render_fixed_viewpoint`: 启用固定视角渲染
- `--fixed_yaw`: 水平旋转偏移（弧度），正值向右，负值向左（例如：0.3 ≈ 17°）
- `--fixed_pitch`: 垂直旋转偏移（弧度），正值向下，负值向上（例如：0.2 ≈ 11°）
- `--fixed_zoom`: 缩放倍数（默认1.0），>1.0拉远，<1.0拉近（例如：1.3表示30%更远）

**输出**: `{output_dir}/render_fixed_viewpoint/{video_id}/{video_id}_fixed_viewpoint_video.mp4`

## 技术实现

### 修改的文件

1. **utils/camera_utils.py**
   - 新增 `generate_novel_view_poses_with_zoom()`: 生成带zoom的动态视角
   - 新增 `generate_novel_view_poses_fixed()`: 生成固定视角

2. **main/test.py**
   - 新增 `render_novel_views_with_zoom()`: 渲染带zoom的动态视角
   - 新增 `render_novel_views_fixed()`: 渲染固定视角
   - 新增命令行参数支持

### 核心算法

**Zoom功能**:
```python
# 使用正弦波实现平滑的zoom in/out
radius_multiplier = 1.0 + radius_range * np.sin(2 * π * frame_idx / num_keyframes)
current_radius = base_radius * radius_multiplier
```

**固定视角**:
```python
# 所有帧使用相同的相机参数
horizontal_angle = π/2 + fixed_yaw
vertical_angle = π/2 - 0.05 + fixed_pitch
fixed_radius = base_radius * fixed_zoom
```

## 测试结果

### 已测试功能
- ✅ Self-reenactment (原功能)
- ✅ Cross-reenactment (原功能)
- ✅ Dynamic novel views with zoom (新功能)
- ⏳ Fixed viewpoint rendering (待测试)

### 性能指标
| 模式 | 速度 | 输出大小 |
|------|------|----------|
| Self-reenactment | ~6 it/s | 3.1MB |
| Cross-reenactment | ~8.5 it/s | 2.0MB |
| Dynamic with zoom | ~8.3 it/s | 1.8MB |

## 应用场景

1. **Zoom功能**:
   - 创建更动态的展示视频
   - 增加视觉吸引力
   - 展示3D模型的深度信息

2. **固定视角**:
   - 数据增强：从不同固定角度生成训练数据
   - 多视角对比：生成多个固定角度的视频进行对比
   - 一致性测试：验证模型在不同视角下的稳定性

## 同步日期

2026-02-01

## 来源

从 https://github.com/openhe-hub/guava-aug.git 同步
Commit: 8a8242b (2026-01-28-clean)
