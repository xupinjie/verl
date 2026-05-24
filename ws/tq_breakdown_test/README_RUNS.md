# 30B-VL MoE GDR test — log convention

跟 `ws/long_run_test/` 同样的目录组织,方便后续复用历史 log 不用每次重跑。

## 启动脚本

| 文件 | TQ 后端 | use_gdr | 用途 |
|---|---|---|---|
| `acc_text_30b_vl_onethinker_baseline.sh` | n/a(Ray Object Store) | n/a | legacy 路径 |
| `acc_text_30b_vl_onethinker_tq.sh` | SimpleStorage(进程内) | n/a | 原 TQ 路径 |
| `acc_text_30b_vl_onethinker_tq_gdr.sh` | **MooncakeStore over RDMA** | `USE_GDR=${USE_GDR:-1}` 切换 | GDR 对照及其无 GDR 兄弟 |

GDR 脚本 `USE_GDR=1`(默认)= TQ-GDR;`USE_GDR=0` = TQ + Mooncake **without** GDR——
让两条路径用同一份脚本跑出来,perf/accuracy 直接对得上。

## Log 目录约定

```
ws/tq_breakdown_test/
├── run_logs_baseline/      legacy 路径,每次 launch 一份 launch_YYYYMMDD_HHMMSS.log
├── run_logs_tq/            TQ + SimpleStorage
├── run_logs_tq_mooncake/   TQ + MooncakeStore(no GDR)
├── run_logs_tq_gdr/        TQ + MooncakeStore + GDR
```

## 启动模板(从 jump 节点)

```bash
VERL=/lustre/fsw/coreai_devtech_hugectr/pinjiex/workspace/rl/verl
TS=$(date +%Y%m%d_%H%M%S)

# GDR run
LOG=$VERL/ws/tq_breakdown_test/run_logs_tq_gdr/launch_${TS}.log
cn "cd $VERL && VERL_TQ_GDR_KEEP_CUDA=1 USE_GDR=1 TOTAL_TRAINING_STEPS=10 \
    nohup bash ws/tq_breakdown_test/acc_text_30b_vl_onethinker_tq_gdr.sh \
    > $LOG 2>&1 < /dev/null & disown"

# Mooncake-no-GDR run (same script, USE_GDR=0)
LOG=$VERL/ws/tq_breakdown_test/run_logs_tq_mooncake/launch_${TS}.log
cn "cd $VERL && USE_GDR=0 TOTAL_TRAINING_STEPS=10 \
    nohup bash ws/tq_breakdown_test/acc_text_30b_vl_onethinker_tq_gdr.sh \
    > $LOG 2>&1 < /dev/null & disown"
```
