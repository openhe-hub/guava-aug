# GUAVA Augmentation 与 EHM Tracker 用法总结

本文档总结当前仓库中用于 Phoenix 数据处理的两段流程：
1. EHM Tracker 跟踪视频，生成 `tracked` 数据。
2. GUAVA 固定视角增强，生成 `augmented` 视频。

## 1. 流程总览

1. 原始 MP4 输入到 EHM Tracker。
2. EHM Tracker 输出每个视频的 tracked 目录（含 `videos_info.json`、参数、可视化、lmdb 等）。
3. GUAVA 读取 tracked 目录，在 `main/test.py --render_fixed_viewpoint` 下做视角增强。
4. 输出增强 MP4（通常在 `render_fixed_viewpoint/..._fixed_viewpoint_video.mp4`）。

## 2. EHM Tracker 用法

代码入口：`EHM-Tracker/tracking_video.py`

参数定义：`EHM-Tracker/src/configs/argument_config.py`
- `-i/--in_root`：输入视频文件或目录。
- `-o/--output_dir`：输出目录。
- `-v/--visible_gpus`：可见 GPU，如 `0` 或 `0,1,2,3`。
- `-n/--n_divide`：并行分片数量。
- `--not_check_hand`：关闭手部置信度筛帧。
- `--check_hand_score`：手部筛帧阈值（默认 0.7）。

### 2.1 单视频

```bash
cd EHM-Tracker
export PYTHONPATH='.'
python tracking_video.py \
  -i /path/to/video.mp4 \
  -o /path/to/tracked/video_name \
  -v 0 \
  -n 1 \
  --not_check_hand
```

### 2.2 目录批处理

```bash
cd EHM-Tracker
export PYTHONPATH='.'
python tracking_video.py \
  -i /path/to/videos_dir \
  -o /path/to/tracked_root \
  -v 0,1,2,3 \
  -n 8 \
  --tracking_with_interval
```

## 3. GUAVA 增强用法

批处理脚本：`scripts/run_guava_augmentation.py`

### 3.1 输入/输出约定

- 输入 `tracked_root` 支持两种结构：
  - `tracked_root/{video}`
  - `tracked_root/{video}/{video}`（脚本会自动探测 `videos_info.json`）
- 输出 `output_root/{difficulty}/{video}/render_fixed_viewpoint/{video}/{video}_fixed_viewpoint_video.mp4`

### 3.2 直接指定视频列表

```bash
python scripts/run_guava_augmentation.py \
  --video_list vid1 vid2 vid3 \
  --tracked_root /path/to/tracked_root \
  --output_root /path/to/augmented_root \
  --difficulties easy medium hard \
  --base_seed 42 \
  --guava_root /home/nyuair/zhewen/GUAVA \
  --model_path assets/GUAVA \
  --device 0
```

### 3.3 使用文本列表

```bash
python scripts/run_guava_augmentation.py \
  --video_list_file /path/to/part_00.txt \
  --tracked_root /path/to/tracked_root \
  --output_root /path/to/augmented_root \
  --difficulties easy \
  --base_seed 42 \
  --guava_root /home/nyuair/zhewen/GUAVA \
  --model_path assets/GUAVA \
  --device 0
```

## 4. 当前固定视角渲染逻辑（背景保留）

关键代码：
- `main/test.py` 中 `render_novel_views_fixed()`
- `models/UbodyAvatar/gaussian_render.py`

当前做法：
1. 渲染时打开 alpha：
   - `compute_alpha=True`
   - `composite_bg=False`
2. 从 renderer 取 `alpha_images` 作为前景掩码。
3. 先从 GT 图像中构造背景（对人物区域做修复填充），再合成：
   - `final = render * alpha + background * (1 - alpha)`

这套逻辑避免了纯黑/纯白背景，也避免直接叠 GT 人物造成双人影。

## 5. 帧数一致性检查建议

最小检查：
1. 读取原始视频帧数。
2. 对比 tracked 中 `videos_info.json` 的 `frames_num`。
3. 对比增强视频 MP4 帧数是否与原始一致。

注意：
- 某些节点上 `cv2` 可能缺 `libGL.so.1`，可以用 `decord` 替代做帧计数。
- 如果 tracked 目录“非空即跳过”，半成品目录会被误判已完成，建议遇到异常样本时先删该样本目录再重跑。

## 6. 常见坑

1. `python: command not found`
- 在脚本中用绝对解释器路径（如 `/scratch/.../envs/GUAVA/bin/python`）。
- `EHM-Tracker/tracking_video.py` 子进程建议使用 `sys.executable`。

2. 背景错误（纯色）
- 不要依赖 `--bg_color 1.0` 做背景填充。
- 使用 renderer alpha + 背景合成。

3. 双影/黑影
- 不要把渲染前景直接叠回含人物的 GT 图。
- 先去人物（inpaint）再做 alpha 合成。

4. 跳过逻辑导致漏修复
- 现有数组脚本按“输出目录非空”跳过。
- 出现掉帧样本时，先删除该样本输出目录再重跑对应分片。
