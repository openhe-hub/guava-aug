# jubail3 远程部署与运行指南（GUAVA + EHM）

本文档只针对 `jubail3`（`login3.fast`）上的 GUAVA 项目：
`/scratch/zl4487/zhewen/GUAVA`

## 1. 登录与身份

登录链路：
1. 本地 -> `chatsign@10.224.35.17`
2. 跳转 -> `jubail3`

当前账号与主机：
- 用户：`zl4487`
- 主机：`login3.fast`

## 2. 项目路径与当前目录现状

项目根目录：
- `/scratch/zl4487/zhewen/GUAVA`

当前已确认存在：
- `EHM-Tracker/`
- `main/`, `models/`, `scripts/`, `assets/`, `outputs/`
- `scripts/run_guava_augmentation.py`

当前未发现（需要自行准备）：
- `scripts/run_tracking_array.sh`
- `scripts/run_augmentation_array.sh`
- `scripts/video_splits/part_*.txt`

说明：`jubail3` 这个副本当前更像“单机/开发副本”，不是已经配置好的 `phoenix train 10卡数组任务`副本。

## 3. Python 与环境

默认 shell 下没有 `python`（`which python` 为空）。

已存在可用环境：
- `/scratch/zl4487/miniconda3/envs/GUAVA/bin/python`

建议统一使用绝对解释器，避免脚本在 SLURM 节点上报 `python: command not found`。

## 4. SLURM 命令

`jubail3` 同样可用以下 SLURM 路径：

```bash
export PATH=/opt/slurm/20.11.4-13/bin:$PATH
```

可用命令：
- `/opt/slurm/20.11.4-13/bin/sbatch`
- `/opt/slurm/20.11.4-13/bin/squeue`
- `/opt/slurm/20.11.4-13/bin/sacct`

## 5. 数据目录建议（phoenix 任务）

当前 `assets/` 下未看到 `assets/phoenix14t`（仅有 example/template 等目录）。

如果要在 `jubail3` 复用与 `jubail2` 同样的流程，建议先按如下约定准备目录：
- 输入视频：`/scratch/zl4487/zhewen/GUAVA/assets/phoenix14t/vid/videos_mp4/train`
- tracked 输出：`/scratch/zl4487/zhewen/GUAVA/assets/phoenix14t/tracked_nohand`
- 增强输出：`/scratch/zl4487/zhewen/GUAVA/assets/phoenix14t/augmented_nohand`
- 日志目录：`/scratch/zl4487/zhewen/GUAVA/logs`

## 6. 最小可跑命令（单机）

### 6.1 单视频 EHM 跟踪

```bash
cd /scratch/zl4487/zhewen/GUAVA/EHM-Tracker
export PYTHONPATH='.'
/scratch/zl4487/miniconda3/envs/GUAVA/bin/python tracking_video.py \
  -i /path/to/video.mp4 \
  -o /path/to/tracked/video_name \
  -v 0 \
  -n 1 \
  --not_check_hand
```

### 6.2 单视频/小批量 GUAVA 固定视角增强

```bash
cd /scratch/zl4487/zhewen/GUAVA
/scratch/zl4487/miniconda3/envs/GUAVA/bin/python scripts/run_guava_augmentation.py \
  --video_list your_video_name \
  --tracked_root /path/to/tracked_root \
  --output_root /path/to/augmented_root \
  --difficulties easy \
  --base_seed 42 \
  --guava_root /scratch/zl4487/zhewen/GUAVA \
  --model_path assets/GUAVA \
  --device 0
```

## 7. 若要复刻 jubail2 的 10卡数组流程

需要先补齐三类文件：
1. `scripts/run_tracking_array.sh`
2. `scripts/run_augmentation_array.sh`
3. `scripts/video_splits/part_00.txt ... part_09.txt`

补齐后再执行标准两阶段提交：
1. `sbatch --array=0-9 run_tracking_array.sh`
2. `sbatch --dependency=afterok:<track_jobid> run_augmentation_array.sh`

## 8. 常见问题

1. `python: command not found`
- 直接用绝对解释器路径 `/scratch/zl4487/miniconda3/envs/GUAVA/bin/python`。

2. 背景变纯色
- 使用当前仓库里已修复的 alpha 合成逻辑，不依赖 `--bg_color 1.0`。

3. 掉帧排查
- 对齐三方帧数：原始 mp4、tracked `frames_num`、增强 mp4。

