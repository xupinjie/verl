set -x

# mooncake binaries (mooncake_master, mooncake_http_metadata_server) are pip-installed
# under --user into /root/.local/bin which is not on the container's default PATH.
export PATH=/root/.local/bin:$PATH

unset ROCR_VISIBLE_DEVICES
ARNOLD_WORKER_NUM=${ARNOLD_WORKER_NUM:-2}
ARNOLD_WORKER_GPU=${ARNOLD_WORKER_GPU:-8}

HDFS_ROOT=${HDFS_ROOT:-$PWD}
DATA_ROOT=${DATA_ROOT:-$PWD}

# ===================================== Algorithm =====================================
adv_estimator=grpo
loss_mode=vanilla

# reference policy
use_kl_in_reward=False
kl_coef=0.001
use_kl_loss=False
kl_loss_coef=0.001

clip_ratio_low=0.2
clip_ratio_high=0.28

actor_lr=1e-6
critic_lr=2e-6
gae_gamma=1.0
gae_lam=0.95
critic_warmup=0

# rollout correction
rollout_is="sequence"                     # Self-normalized sequence-level IS
rollout_is_threshold=2.0                  # Upper threshold for IS weights
rollout_is_batch_normalize="true"         # Self-normalization (mean=1.0)

# ===================================== Data/Model =====================================
TASK_NAME=${TASK_NAME:-dapo}
if [[ $TASK_NAME == "gsm8k" ]]; then
    train_files=/data/verl/gsm8k/train.parquet
    test_files=/data/verl/gsm8k/test.parquet
    actor_model_path=/data/verl/Qwen2.5-1.5B-Instruct
    apply_rope_fusion=True
elif [[ $TASK_NAME == "geo3k" ]]; then
    train_files=/data/verl/geometry3k/train.parquet
    test_files=/data/verl/geometry3k/test.parquet
    actor_model_path=/data/verl/Qwen3-VL-2B-Instruct
    apply_rope_fusion=False
elif [[ $TASK_NAME == "dapo" ]]; then
    train_files=/data/verl/dapo-math-17k/train.parquet
    test_files=/data/verl/aime-2024/train.parquet
    actor_model_path=/data/verl/Qwen3-8B-Base
    max_prompt_length=$((1024 * 2))
    max_response_length=$((1024 * 2))
    apply_rope_fusion=True
else
    echo "TASK_NAME $TASK_NAME not supported"
    exit 1
fi

critic_model_path=$actor_model_path

max_prompt_length=${max_prompt_length:-$((1024 * 1))}
max_response_length=${max_response_length:-$((1024 * 2))}
train_batch_size=128
ppo_mini_batch_size=32
n_resp_per_prompt=8
n_resp_per_prompt_val=1

# ===================================== Training =====================================
backend=${BACKEND:-megatron} # fsdp, fsdp2, megatron

actor_max_token_len_per_gpu=$(((max_prompt_length + max_response_length) * 2))
critic_max_token_len_per_gpu=$(((max_prompt_length + max_response_length) * 4))

USP_SIZE=2
ACTOR_FSDP_CONFIG="
    actor_rollout_ref.actor.fsdp_config.strategy=$backend \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=$USP_SIZE"

TP_SIZE=2
CP_SIZE=1
PP_SIZE=1
VPP_SIZE=null
EP_SIZE=1
ETP_SIZE=1
ACTOR_MEGATRON_CONFIG="
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.actor.megatron.context_parallel_size=$CP_SIZE \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$PP_SIZE \
    actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=$VPP_SIZE \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=$EP_SIZE \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=$ETP_SIZE \
    actor_rollout_ref.actor.megatron.param_offload=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
    actor_rollout_ref.actor.megatron.grad_offload=True \
    actor_rollout_ref.actor.megatron.optimizer_offload=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=$apply_rope_fusion \
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True \
    actor_rollout_ref.actor.megatron.use_mbridge=True"

ACTOR_CONFIG="
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.model.path=$actor_model_path \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode}
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$actor_max_token_len_per_gpu"

CIRITC_CONFIG="
    critic.optim.lr=$critic_lr \
    critic.model.path=$critic_model_path \
    critic.model.use_remove_padding=True \
    critic.ppo_max_token_len_per_gpu=$critic_max_token_len_per_gpu"

CRITIC_FSDP_CONFIG="${ACTOR_FSDP_CONFIG//actor_rollout_ref.actor/critic.model}"
CRITIC_MEGATRON_CONFIG="${ACTOR_MEGATRON_CONFIG//actor_rollout_ref.actor/critic}"

if [[ $backend == "megatron" ]]; then
    CONFIG_NAME=ppo_megatron_trainer
    ACTOR_CONFIG="$ACTOR_CONFIG $ACTOR_MEGATRON_CONFIG"
    if [[ $adv_estimator == "gae" ]]; then
        CIRITC_CONFIG="$CIRITC_CONFIG $CRITIC_MEGATRON_CONFIG"
    else
        CIRITC_CONFIG=""
    fi
else # fsdp, fsdp2
    CONFIG_NAME=ppo_trainer
    ACTOR_CONFIG="$ACTOR_CONFIG $ACTOR_FSDP_CONFIG"
    if [[ $adv_estimator == "gae" ]]; then
        CIRITC_CONFIG="$CIRITC_CONFIG $CRITIC_FSDP_CONFIG"
    else
        CIRITC_CONFIG=""
    fi
fi

# ===================================== Inference =====================================
rollout_name=vllm
infer_tp=2
infer_dp=1
infer_ep=1

ROLLOUT_CONFIG="
    actor_rollout_ref.rollout.name=$rollout_name \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
    actor_rollout_ref.rollout.data_parallel_size=$infer_dp \
    actor_rollout_ref.rollout.expert_parallel_size=$infer_ep \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val"

# ===================================== Checkpoint / Long-run =====================================
SAVE_FREQ=${SAVE_FREQ:-25}
MAX_CKPT_KEEP=${MAX_CKPT_KEEP:-1}
TEST_FREQ=${TEST_FREQ:-25}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-15}

# ===================================== TransferQueue backend =====================================
# USE_MOONCAKE=1 -> MooncakeStore (RDMA);  USE_MOONCAKE=0 -> SimpleStorage (default fallback)
# Only override fields that differ from TransferQueue's package default config.yaml
# (metadata_server=localhost:50050, master_server_address=localhost:50051,
#  local_hostname="", auto_init=true, use_gdr=false are all defaults already).
USE_MOONCAKE=${USE_MOONCAKE:-1}
if [[ "$USE_MOONCAKE" == "1" ]]; then
    # The launcher always runs on the slurm head (cn drops us into node 0). Bind
    # mooncake_master here so its HTTP/RPC sockets are reachable at $HEAD_IP from
    # all Ray workers regardless of which node Ray happens to schedule TaskRunner on.
    # We disable TQ's auto_init so it doesn't spawn its own master on whichever
    # node TaskRunner lands; instead it will just connect to the master we started.
    HEAD_IP=${HEAD_IP:-$(hostname -I | awk '{print $1}')}
    MOONCAKE_SEGMENT_SIZE=${MOONCAKE_SEGMENT_SIZE:-8589934592}    # 8 GB
    MOONCAKE_BUFFER_SIZE=${MOONCAKE_BUFFER_SIZE:-2147483648}      # 2 GB
    MOONCAKE_MASTER_LOG=${default_local_dir:-/tmp}/mooncake_master_$(date +%Y%m%d_%H%M%S).log
    mkdir -p "$(dirname "$MOONCAKE_MASTER_LOG")"
    pkill -f "[m]ooncake_master" 2>/dev/null
    sleep 1
    setsid mooncake_master \
        -default_kv_lease_ttl=999999 \
        -default_kv_soft_pin_ttl=999999 \
        --eviction_high_watermark_ratio=1.0 \
        --eviction_ratio=0.0 \
        --enable_http_metadata_server=true \
        --allow_evict_soft_pinned_objects=false \
        --http_metadata_server_host=$HEAD_IP \
        --http_metadata_server_port=50050 \
        --rpc_port=50051 \
        > "$MOONCAKE_MASTER_LOG" 2>&1 < /dev/null &
    MOONCAKE_MASTER_PID=$!
    echo "Started mooncake_master PID=$MOONCAKE_MASTER_PID, log=$MOONCAKE_MASTER_LOG"
    sleep 3
    if ! kill -0 $MOONCAKE_MASTER_PID 2>/dev/null; then
        echo "ERROR: mooncake_master died immediately, log:" >&2
        cat "$MOONCAKE_MASTER_LOG" >&2 || true
        exit 1
    fi
    trap "kill -9 $MOONCAKE_MASTER_PID 2>/dev/null; pkill -f '[m]ooncake_master' 2>/dev/null" EXIT

    TQ_BACKEND_CONFIG="
        transfer_queue.backend.storage_backend=MooncakeStore \
        +transfer_queue.backend.MooncakeStore.auto_init=false \
        +transfer_queue.backend.MooncakeStore.metadata_server=${HEAD_IP}:50050 \
        +transfer_queue.backend.MooncakeStore.master_server_address=${HEAD_IP}:50051 \
        +transfer_queue.backend.MooncakeStore.protocol=rdma \
        +transfer_queue.backend.MooncakeStore.global_segment_size=${MOONCAKE_SEGMENT_SIZE} \
        +transfer_queue.backend.MooncakeStore.local_buffer_size=${MOONCAKE_BUFFER_SIZE}"
else
    TQ_BACKEND_CONFIG=""
fi

# wandb
project_name=${PROJECT_NAME:-pinjiex_${TASK_NAME}_longrun_0508}
experiment_name=${EXPERIMENT_NAME:-qwen3-8B-$adv_estimator-$backend-$rollout_name-tq}
default_local_dir=${DEFAULT_LOCAL_DIR:-$DATA_ROOT/checkpoint/$project_name/$experiment_name}

# wandb resume: continue the same run if checkpoint exists, otherwise start fresh
if ls "$default_local_dir"/global_step_* 1>/dev/null 2>&1; then
    export WANDB_RESUME=allow
    export WANDB_RUN_ID=$(echo -n "$project_name/$experiment_name" | md5sum | cut -c1-8)
    echo "Checkpoint found in $default_local_dir, resuming wandb run (id=$WANDB_RUN_ID)"
else
    unset WANDB_RESUME
    unset WANDB_RUN_ID
    echo "No checkpoint found, starting fresh wandb run"
fi

mkdir -p "$default_local_dir"
python3 -m verl.trainer.main_ppo_sync \
    --config-path=./config \
    --config-name=$CONFIG_NAME \
    algorithm.adv_estimator=$adv_estimator \
    algorithm.use_kl_in_reward=$use_kl_in_reward \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    algorithm.gamma=$gae_gamma \
    algorithm.lam=$gae_lam \
    algorithm.rollout_correction.rollout_is=$rollout_is \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.return_raw_chat=True \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    trainer.use_legacy_worker_impl=disable \
    trainer.critic_warmup=$critic_warmup \
    trainer.logger=['console'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.default_local_dir=$default_local_dir \
    trainer.n_gpus_per_node=$ARNOLD_WORKER_GPU \
    trainer.nnodes=$ARNOLD_WORKER_NUM \
    trainer.val_before_train=False \
    trainer.val_only=False \
    trainer.log_val_generations=100 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.max_actor_ckpt_to_keep=$MAX_CKPT_KEEP \
    trainer.max_critic_ckpt_to_keep=$MAX_CKPT_KEEP \
    trainer.resume_mode=auto \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_epochs=$TOTAL_EPOCHS \
    $ACTOR_CONFIG \
    $CIRITC_CONFIG \
    $ROLLOUT_CONFIG \
    $TQ_BACKEND_CONFIG \
    ${TOTAL_TRAINING_STEPS:+trainer.total_training_steps=$TOTAL_TRAINING_STEPS} \
    2>&1 | tee "${default_local_dir}/train_$(date +%Y%m%d_%H%M%S).log"
