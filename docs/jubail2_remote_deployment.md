# jubail2 远程部署与运行指南（GUAVA + EHM）

本文档记录当前在 `jubail2` 上的实际部署路径、SLURM 提交流程与排障要点。

## 1. 机器与路径

登录链路：
1. 本地 -> `chatsign@10.224.35.17`
2. 跳转 -> `jubail2`

项目路径：
- `/scratch/yf23/zhewen/guava-aug`

核心目录（phoenix14t）：
- 训练/测试视频：`/scratch/yf23/zhewen/guava-aug/assets/phoenix14t/vid/videos_mp4/train`
- tracked 输出：`/scratch/yf23/zhewen/guava-aug/assets/phoenix14t/tracked_nohand`
- augmentation 输出：`/scratch/yf23/zhewen/guava-aug/assets/phoenix14t/augmented_nohand`

核心目录（how2sign）：
- 原始视频：`/scratch/yf23/zhewen/guava-aug/assets/how2sign/original_mp4` (20,000 个)
- tracked 输出：`/scratch/yf23/zhewen/guava-aug/assets/how2sign/tracked_nohand`
- 视频分片：`/scratch/yf23/zhewen/guava-aug/scripts/how2sign_splits/part_00.txt ... part_09.txt`

日志目录：`/scratch/yf23/zhewen/guava-aug/logs`

跳板机落地目录（用于从 `jubail2` 拉回 tracked）：
- `~/zhewen/tracked_nohand`
- 绝对路径（当前账号）：`/Users/chatsign/zhewen/tracked_nohand`
- 同步日志：`~/zhewen/rsync_tracked_nohand.log`

## 2. 环境与 SLURM

SLURM 命令路径：
```bash
export PATH=/opt/slurm/20.11.4-13/bin:$PATH
```

远程 python（当前脚本使用）：
- `/scratch/yf23/miniconda3/envs/GUAVA/bin/python`

## 3. 远程脚本

- 跟踪数组（phoenix14t）：`scripts/run_tracking_array.sh`
- 跟踪数组（how2sign）：`scripts/run_how2sign_tracking_array.sh`
- 增强数组：`scripts/run_augmentation_array.sh`
- 视频分片（phoenix14t）：`scripts/video_splits/part_00.txt ... part_09.txt`
- 视频分片（how2sign）：`scripts/how2sign_splits/part_00.txt ... part_09.txt`
- 分片生成脚本：`scripts/split_videos.py`

## 4. 提交流程（推荐）

### 4.1 先做分片（若视频列表更新）

```bash
cd /scratch/yf23/zhewen/guava-aug
/scratch/yf23/miniconda3/envs/GUAVA/bin/python scripts/split_videos.py \
  --video_dir /scratch/yf23/zhewen/guava-aug/assets/phoenix14t/vid/videos_mp4/train \
  --output_dir /scratch/yf23/zhewen/guava-aug/scripts/video_splits \
  --n_parts 10
```

### 4.2 提交 tracker（10 卡）

```bash
cd /scratch/yf23/zhewen/guava-aug
TRACK_JOB=$(sbatch --parsable --array=0-9 scripts/run_tracking_array.sh)
echo "$TRACK_JOB"
```

### 4.3 提交 guava aug（依赖 tracker 成功）

```bash
cd /scratch/yf23/zhewen/guava-aug
AUG_JOB=$(sbatch --parsable --dependency=afterok:${TRACK_JOB} scripts/run_augmentation_array.sh)
echo "$AUG_JOB"
```

## 5. 监控命令

```bash
# 队列
squeue -u yf23 -o "%.18i %.35j %.8T %.10M %.6D %R"

# 作业状态历史
sacct -j <job_id> --format=JobID,State,Elapsed,ExitCode -n

# 产出计数
find /scratch/yf23/zhewen/guava-aug/assets/phoenix14t/tracked_nohand -name videos_info.json | wc -l
find /scratch/yf23/zhewen/guava-aug/assets/phoenix14t/augmented_nohand -name '*_fixed_viewpoint_video.mp4' | wc -l
```

### 5.1 跳板机同步监控

```bash
# 在跳板机查看已同步 tracked 数量
find ~/zhewen/tracked_nohand -name videos_info.json | wc -l

# 查看同步进程
ps -ef | grep rsync | grep tracked_nohand

# 查看同步日志
tail -f ~/zhewen/rsync_tracked_nohand.log
```

## 6. 配额与资源约束

1. GPU 数量约束
- 同账号通常最多同时占用 10 张卡。
- 因此 `ehm_track(10卡)` 期间，`guava_aug(10卡)` 会保持 `PENDING (Dependency)` 或资源等待。

2. inode 配额约束（重点）
- 曾出现 `Disk quota exceeded`，但磁盘空间足够，实际是文件数（inode）超限。
- 检查命令：
```bash
lfs quota -h -u yf23 /scratch
```

3. 建议清理目录
- 历史旧产物：`assets/phoenix14t/tracked`、`assets/phoenix14t/augmented`
- 不要误删当前正在写入的 `tracked_nohand`、`augmented_nohand`。

## 7. 失败重跑策略

假设 `array 1,8` 失败：

```bash
# 仅补跑失败分片
RETRY_JOB=$(sbatch --parsable --array=1,8 scripts/run_tracking_array.sh)

# 取消旧的 aug 依赖作业并重提（依赖原作业完成 + 补跑成功）
scancel <old_aug_jobid>
NEW_AUG_JOB=$(sbatch --parsable --dependency=afterany:<orig_track_jobid>,afterok:${RETRY_JOB} scripts/run_augmentation_array.sh)
```

## 8. 常见问题与修复

1. `python: command not found`
- 在 `run_tracking_array.sh` / `run_augmentation_array.sh` 使用绝对 python 路径。
- `tracking_video.py` 子进程调用建议用 `sys.executable`。

2. 跳过逻辑误判
- 当前 tracker 脚本是“输出目录非空就跳过”。
- 半成品目录可能被误判完成，建议异常样本先删目录再重跑。

3. 背景问题
- 当前 GUAVA 固定视角增强已采用 alpha 合成保留背景。
- 不建议再通过 `--bg_color 1.0` 强制白底。

4. 掉帧问题
- 需要做样本核对：原始 mp4 帧数 vs tracked `frames_num` vs augmented mp4 帧数。

## 9. 当前远程脚本行为说明

`run_tracking_array.sh`：
- 每个 array task 读取 `part_XX.txt`。
- 逐视频执行：
  - 输入：`assets/phoenix14t/vid/videos_mp4/train/${video}.mp4`
  - 输出：`assets/phoenix14t/tracked_nohand/${video}`
  - 参数：`--not_check_hand -n 1 -v 0`

`run_augmentation_array.sh`：
- 每个 array task 读取 `part_XX.txt`。
- 调用 `scripts/run_guava_augmentation.py`。
- 当前配置默认只跑 `--difficulties easy`（如需 `medium/hard` 可改脚本参数）。

`run_how2sign_tracking_array.sh`：
- 每个 array task 读取 `how2sign_splits/part_XX.txt`。
- 逐视频执行：
  - 输入：`assets/how2sign/original_mp4/${video}.mp4`
  - 输出：`assets/how2sign/tracked_nohand/${video}`
  - 参数：`--not_check_hand -n 1 -v 0`

## 10. How2Sign EHM Tracking

### 10.1 数据规模
- 20,000 个 mp4 视频（~20GB）
- 分为 10 个 part，每 part 2,000 个视频

### 10.2 提交流程

```bash
# 1. 生成分片
cd /scratch/yf23/zhewen/guava-aug
/scratch/yf23/miniconda3/envs/GUAVA/bin/python scripts/split_videos.py \
  --video_dir /scratch/yf23/zhewen/guava-aug/assets/how2sign/original_mp4 \
  --output_dir /scratch/yf23/zhewen/guava-aug/scripts/how2sign_splits \
  --n_parts 10

# 2. 提交 tracking（10 卡）
export PATH=/opt/slurm/default/bin:$PATH
sbatch scripts/run_how2sign_tracking_array.sh
```

### 10.3 监控

```bash
# 队列状态
squeue -u yf23

# 已完成 tracking 数量
ls /scratch/yf23/zhewen/guava-aug/assets/how2sign/tracked_nohand/ | wc -l

# 检查特定 task 日志
tail -50 logs/h2s_track_<jobid>_<taskid>.out
```

### 10.4 失败重跑

```bash
# 仅补跑失败分片（例如 task 2, 5）
export PATH=/opt/slurm/default/bin:$PATH
sbatch --array=2,5 scripts/run_how2sign_tracking_array.sh
```
