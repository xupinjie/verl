#!/bin/bash
set -u
TS=$(date +%Y%m%d_%H%M%S)
LOG_ROOT=/home/scratch.pinjiex_gpu/workspace/rl/verl/ws/long_run_test_24h/run_logs
LAUNCH_LOG=$LOG_ROOT/launch_${TS}.log
exec > "$LAUNCH_LOG" 2>&1
echo "[inner_launch] $(date) starting on $(hostname)"
echo "[inner_launch] LAUNCH_LOG=$LAUNCH_LOG"

unset ROCR_VISIBLE_DEVICES
export MC_TCP_ENABLE_CONNECTION_POOL=1
export MC_STORE_MEMCPY=1
export PATH=/root/.local/bin:$PATH
ulimit -n 1048576

cd /home/scratch.pinjiex_gpu/workspace/rl/verl
echo "[inner_launch] $(date) about to exec training script"
exec bash ws/long_run_test_24h/run_30b_vl_tq_mooncake_24h.sh
