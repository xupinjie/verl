#!/bin/bash
# 24h watchdog for the long-run mooncake training.
# Polls main_ppo_sync every WATCH_INTERVAL seconds; if dead, restarts the
# training (which uses trainer.resume_mode=auto, so it picks up from the
# last save_freq=50 checkpoint).
#
# Usage:
#   nohup bash ws/long_run_test_24h/watchdog.sh > $WATCHLOG 2>&1 < /dev/null &

set -u

VERL=/home/scratch.pinjiex_gpu/workspace/rl/verl
LOG_ROOT=$VERL/ws/long_run_test_24h/run_logs
WATCHLOG=$LOG_ROOT/watchdog_$(date +%Y%m%d_%H%M%S).log
TRAIN_SCRIPT=ws/long_run_test_24h/run_30b_vl_tq_mooncake_24h.sh

# the enroot container PID — pinned at script start, must be alive throughout
ENROOT_PID=${ENROOT_PID:-3649242}

WATCH_INTERVAL=${WATCH_INTERVAL:-60}
RESTART_DELAY=${RESTART_DELAY:-30}
MAX_RESTARTS=${MAX_RESTARTS:-100}

restarts=0
last_restart_ts=0

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

is_training_alive() {
    pgrep -f "main_ppo_sync" >/dev/null
}

current_step() {
    # find latest train log, grep latest step
    local last_log=$(ls -t $LOG_ROOT/train_*.log 2>/dev/null | head -1)
    if [[ -n "$last_log" ]]; then
        grep -oE "step:[0-9]+|Training Progress: +[0-9]+/" "$last_log" 2>/dev/null \
            | tail -1
    fi
}

latest_checkpoint() {
    local ckpt_root=$VERL/checkpoint/pinjiex_longrun_24h_30b_vl_tq_mooncake/grpo-tq-mooncake-geo3k-30b-vl-longrun-24h
    if [[ -f $ckpt_root/latest_checkpointed_iteration.txt ]]; then
        cat $ckpt_root/latest_checkpointed_iteration.txt
    else
        echo "none"
    fi
}

# verl's per-process ckpt rotation doesn't persist across restarts, so old
# global_step_* dirs accumulate after any crash+restart. This reaper does
# what verl's max_ckpt_to_keep should have done across the whole disk:
#   1) Delete any global_step_N where N > latest_checkpointed_iteration.txt
#      (these are partial writes from a crashed save — verl never blessed them).
#   2) Of the surviving ckpts, keep the most recent KEEP_CKPT_COUNT, delete rest.
# SAFETY: never deletes the dir pointed to by latest_checkpointed_iteration.txt.
KEEP_CKPT_COUNT=${KEEP_CKPT_COUNT:-1}
reap_old_checkpoints() {
    local ckpt_root=$VERL/checkpoint/pinjiex_longrun_24h_30b_vl_tq_mooncake/grpo-tq-mooncake-geo3k-30b-vl-longrun-24h
    local latest_file=$ckpt_root/latest_checkpointed_iteration.txt
    if [[ ! -d $ckpt_root ]]; then return; fi

    local blessed=""
    if [[ -f $latest_file ]]; then
        blessed=$(cat $latest_file 2>/dev/null | tr -d '[:space:]')
    fi
    if [[ -z "$blessed" ]]; then
        log "reaper: no latest_checkpointed_iteration.txt yet, keeping all ckpts (none to safely reap)"
        return
    fi

    # 1) delete incomplete ckpts (step > blessed)
    local removed_incomplete=0
    for d in "$ckpt_root"/global_step_*; do
        [[ -d "$d" ]] || continue
        local step=$(basename "$d" | sed 's/^global_step_//')
        [[ "$step" =~ ^[0-9]+$ ]] || continue
        if (( step > blessed )); then
            log "reaper: deleting incomplete ckpt $(basename "$d") (step $step > blessed $blessed)"
            rm -rf "$d"
            removed_incomplete=$((removed_incomplete + 1))
        fi
    done

    # 2) of remaining (step <= blessed), keep the latest KEEP_CKPT_COUNT
    local kept=0
    local removed_old=0
    # sort by numeric step DESC, blessed is always included implicitly because step <= blessed
    while IFS= read -r d; do
        [[ -d "$d" ]] || continue
        local step=$(basename "$d" | sed 's/^global_step_//')
        [[ "$step" =~ ^[0-9]+$ ]] || continue
        if (( kept < KEEP_CKPT_COUNT )); then
            # keep — but double-check we never delete the blessed one accidentally
            kept=$((kept + 1))
        else
            # NEVER delete the blessed one even if KEEP_CKPT_COUNT excludes it
            if (( step == blessed )); then
                log "reaper: refusing to delete blessed ckpt step=$blessed (would orphan latest pointer)"
                continue
            fi
            log "reaper: deleting old ckpt $(basename "$d") (step $step, keeping $KEEP_CKPT_COUNT newest)"
            rm -rf "$d"
            removed_old=$((removed_old + 1))
        fi
    done < <(ls -d "$ckpt_root"/global_step_* 2>/dev/null \
                | awk -F'global_step_' '{print $2, $0}' \
                | sort -k1 -n -r \
                | awk '{print $2}')

    log "reaper: blessed=$blessed, removed $removed_incomplete incomplete + $removed_old old, kept $kept"
    df -h "$ckpt_root" 2>/dev/null | tail -1 | awk '{print "  disk: " $3 " used / " $4 " avail (" $5 ")"}'
}

cleanup_residuals() {
    log "cleanup: killing leftover train procs"
    pkill -9 -f "main_ppo_sync" 2>/dev/null || true
    pkill -9 -f "ray::WorkerDict" 2>/dev/null || true
    pkill -9 -f "ray::TaskRunner" 2>/dev/null || true
    pkill -9 -f "AgentLoopWorker" 2>/dev/null || true
    pkill -9 -f "vLLMHttpServer" 2>/dev/null || true
    pkill -9 -f "EngineCore" 2>/dev/null || true
    pkill -9 -f "[m]ooncake_master" 2>/dev/null || true
    sleep 5
    # nuke any leftover ray actors
    ps -ef | grep -E "ray::Worker|ray::TaskRunner|ray::Async" | grep -v grep \
        | awk '{print $2}' | xargs -r kill -9 2>/dev/null || true
    sleep 3
}

launch_training() {
    local launch_ts=$(date +%Y%m%d_%H%M%S)
    local launch_log=$LOG_ROOT/launch_${launch_ts}.log
    log "launching training, log=$launch_log"
    enroot exec $ENROOT_PID bash -c "
        unset ROCR_VISIBLE_DEVICES
        export MC_TCP_ENABLE_CONNECTION_POOL=1
        export MC_STORE_MEMCPY=1
        ulimit -n 1048576
        cd $VERL
        nohup bash $TRAIN_SCRIPT > $launch_log 2>&1 < /dev/null &
        disown
    "
    sleep 30  # give it 30s to start
}

restart_ray_if_needed() {
    # If Ray cluster died (rare but possible), bring it back with the right env
    if ! enroot exec $ENROOT_PID ray status >/dev/null 2>&1; then
        log "ray cluster appears dead, restarting"
        local head_ip=$(hostname -I | awk '{print $1}')
        enroot exec $ENROOT_PID bash -c "
            unset ROCR_VISIBLE_DEVICES
            export MC_TCP_ENABLE_CONNECTION_POOL=1
            export MC_STORE_MEMCPY=1
            export PATH=/root/.local/bin:\$PATH
            ulimit -n 1048576
            ray start --head --node-ip-address=$head_ip --port=6379 --num-gpus=8 \
                > $LOG_ROOT/ray_restart_$(date +%Y%m%d_%H%M%S).log 2>&1
            sleep 4
        "
    fi
}

log "watchdog starting, ENROOT_PID=$ENROOT_PID, interval=${WATCH_INTERVAL}s"
log "current step: $(current_step), latest ckpt: $(latest_checkpoint)"

while true; do
    if ! is_training_alive; then
        now=$(date +%s)
        gap=$((now - last_restart_ts))
        if [[ $restarts -ge $MAX_RESTARTS ]]; then
            log "FATAL: exceeded MAX_RESTARTS=$MAX_RESTARTS, giving up"
            exit 1
        fi
        # crash-loop guard: if we restarted < 60s ago, wait longer
        if [[ $gap -lt 60 ]]; then
            log "crash too soon after last restart (${gap}s), backing off"
            sleep $((RESTART_DELAY * 2))
        fi
        log "training is DEAD (restart #$((restarts+1))). last ckpt: $(latest_checkpoint)"
        cleanup_residuals
        reap_old_checkpoints
        restart_ray_if_needed
        launch_training
        restarts=$((restarts + 1))
        last_restart_ts=$(date +%s)
        log "training relaunched (total restarts: $restarts)"
    else
        # alive — log progress periodically
        log "alive. step: $(current_step), ckpt: $(latest_checkpoint), restarts: $restarts"
    fi
    sleep $WATCH_INTERVAL
done
