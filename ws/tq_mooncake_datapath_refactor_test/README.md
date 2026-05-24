# MooncakeStoreClient data-path refactor test

Goal: validate the refactored `deps/TransferQueue/transfer_queue/storage/clients/mooncake_client.py`
unified data-path (every value packed into a single CPU uint8 buffer, one
register + one `batch_upsert_from` per batch; symmetric on get).

## Differences vs `ws/tq_breakdown_test/`

- `use_gdr=false` hard-coded (the refactor only writes to CPU buffers)
- No `VERL_TQ_MC_GET_RETRIES` / `VERL_TQ_MC_GET_RETRY_DELAY_S` exported —
  the bytes-pool exhaustion they worked around no longer exists
- `experiment_name=grpo-tq-mooncake-datapath-refactor` (separate wandb / ckpt)

## Files

| file | purpose |
|---|---|
| `acc_text_30b_vl_onethinker_tq_mooncake.sh` | launch script (30B-VL + onethinker, megatron + vLLM) |
| `plot_datapath_perf_3way.py` | 3-way perf plot vs `run_logs_baseline` + `run_logs_tq` |
| `run_logs/` | per-launch log dumps |

## Launch (10 steps)

```bash
# 1. ray cluster (2 nodes)
bash submit_ray.sh 2

# 2. wait for the slurm job to be running and Ray head to be up
#    squeue -u $USER  → look for "ray" job in R state
#    cn "ray status"  → confirm head + worker registered

# 3. submit the training run inside the container on the compute node
VERL=/lustre/fsw/coreai_devtech_hugectr/pinjiex/workspace/rl/verl
TS=$(date +%Y%m%d_%H%M%S)
LOG=$VERL/ws/tq_mooncake_datapath_refactor_test/run_logs/launch_10step_${TS}.log
cn "cd $VERL && TOTAL_TRAINING_STEPS=10 \
    nohup bash ws/tq_mooncake_datapath_refactor_test/acc_text_30b_vl_onethinker_tq_mooncake.sh \
    > $LOG 2>&1 < /dev/null & disown"
```

## Success criteria

- 10 steps complete; trainer exits 0
- No `AssertionError`, no `BATCH_ASSERT=FAIL`, no `still_empty` in log
- No `batch_upsert_from retry` / `batch_get_into retry` WARN/ERROR lines
- mooncake_master log has near-zero `Failed to allocate buffer`

## Plot

After the 10-step run finishes, edit `LOG_PATH_REFACTOR` at the top of
`plot_datapath_perf_3way.py` to point at the new log file and run:

```bash
python3 ws/tq_mooncake_datapath_refactor_test/plot_datapath_perf_3way.py
```

The plot has 3 panels (reward / per-step wall time / phase breakdown) and 3 lines:
baseline (no TQ), TQ + SimpleStorage, TQ + Mooncake (data-path refactor).
