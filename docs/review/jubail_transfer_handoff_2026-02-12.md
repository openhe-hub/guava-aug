# Jubail Transfer Handoff (2026-02-12)

## Goal
- Sync Phoenix tracked outputs from `jubail2` to `jubail3`.
- After sync is complete, submit 4-GPU cross-act job (`guava_cross`) on `jubail3`.

## Verified Current State
- `jubail2` tracked data is complete (`6715` `videos_info.json`).
- Source path on `jubail2`:
  - `/scratch/yf23/zhewen/guava-aug/assets/phoenix14t/tracked_nohand`
- Jump host local mirror is complete (`6715`):
  - `~/zhewen/tracked_nohand`
- `jubail3` is only partially synced (last reliable read was around `809/6715` before sync stop).

## Main Issues Observed
- Frequent SSH resets from jump host to Jubail:
  - `kex_exchange_identification: Connection reset by peer`
  - mostly against `10.230.42.101:22`
- High-frequency SSH/rsync retries make stability worse.

## Known Pitfalls Already Identified
- Jump-host `rsync` is older; unsupported options caused failures:
  - `--append-verify`
  - `--info=...`
- In bash loops, `ssh` without `-n` can consume loop stdin and corrupt traversal logic.
- Complex heredoc-in-heredoc remote script generation can break quoting.

## Important: Current Process State
- All transfer/upload processes started by previous agent were stopped on request.
- No active `copy_jump_to_j3*`, `orchestrate*`, `rsync tracked_nohand`, or `hf upload` tasks should remain.

## Useful Existing Scripts on Jump Host (`~/zhewen`)
- `setup_cross_on_j3.sh`
  - Prepares split files and SLURM scripts on `jubail3`, then runs `sbatch`.
- `copy_jump_to_j3_rsync_min_v2.sh`
  - Per-directory rsync; worked but sensitive to connection resets.
- `copy_jump_to_j3_throttled.sh`
  - Lower-frequency sync strategy to reduce SSH pressure.

## What the Next Agent Should Do
1. Verify low-frequency connectivity from jump host to `jubail3`.
2. Resume `jump -> jubail3` sync (prefer throttled, low-pressure approach).
3. Monitor until `j3_count == 6715`.
4. Run `~/zhewen/setup_cross_on_j3.sh` (if no `guava_cross` already queued/running).
5. Return:
  - submitted `JOBID`
  - `squeue` status
  - log paths

## Related Docs
- `docs/guava_ehm_usage.md`
- `docs/jubail2_remote_deployment.md`
- `docs/jubail3_remote_deployment.md`
