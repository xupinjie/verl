# TransferQueue 在 verl 上的加速来源分析

日期: 2026-05-07
作者: pinjiex (with AI assistance)
对应分支: `xpj/timer @ 4cd9e983`

## 1. 问题

接入 TransferQueue (TQ) 后,verl 的训练 step 有约 30% 的端到端加速。**加速的具体来源是什么?在哪个 phase、哪个具体环节?**

社区的 [PR #5958](https://github.com/verl-project/verl/pull/5958) 提供了一个单向 dispatch 延迟的指标,但只测 controller→worker 的正向传输,不能解答双向数据搬运、也不能解释 gen 阶段的收益。我们补充了一套覆盖更全的 timing 仪器,在所有关键 phase 同时上报 `compute_time` / `transfer_time` / `rpc_total`,并把 gen 进一步拆成 4 段:
`rollout_time` / `postprocess_time` / `concat_time` / `transfer_time`。

## 2. 实验配置

| 项 | 值 |
|---|---|
| 算法 | GRPO (no critic, no KL ref) |
| 数据 | OneThinker train.parquet |
| 模型 | Qwen3-VL-30B-A3B-Instruct |
| 训练 | Megatron (TP=4, CP=2, PP=1, EP=8, ETP=1) + USP=2 |
| 推理 | vLLM (TP=2, n=8 responses/prompt) |
| 硬件 | 2 nodes × 8 H100 (80GB) |
| Batch | train_batch_size=128, ppo_mini_batch_size=32 |
| Steps | 3 per run |

对比的两条路径:

- **baseline**: `verl.trainer.main_ppo`,经 `RayPPOTrainer`,数据走 Ray Object Store。
- **TQ**: `verl.trainer.main_ppo_sync`,经 `PPOTrainer`(TQ 版),数据走 TransferQueue。

## 3. 主要结果

### 3.1 端到端 wall-clock(每 step 各 phase 之和)

| step | baseline | TQ | 节省 |
|---|---|---|---|
| 1 | 542.4s | 376.1s | −31% |
| 2 | 458.1s | 317.4s | −31% |
| 3 | 644.1s | 382.2s | −41% |

→ 早先的 timeline 图 [`ws/timeline_baseline_vs_tq.png`](timeline_baseline_vs_tq.png)。

### 3.2 各 phase 的 compute / transfer 拆分(3-step 平均)

| phase | metric | baseline (s) | TQ (s) | Δ |
|---|---|---|---|---|
| **gen** | phase_total | 255.94 | 188.69 | −67.25 |
| | (transfer 部分) | 38.82 | 3.68 | −35.14 |
| **old_log_prob** | phase_total | 116.44 | 65.78 | −50.66 |
| | rpc / compute_rpc | 108.11 | 65.43 | −42.68 |
| | transfer | **44.04** | **0.26** | −43.78 (−99%) |
| **update_actor** | phase_total | 158.39 | 103.57 | −54.82 |
| | rpc_total | 147.78 | 103.57 | −44.21 |
| | compute_time | 101.59 | 91.90 | −9.69 |
| | transfer | 46.18 | 11.67 | −34.51 (−75%) |

### 3.3 Gen 4-段拆分(per step)

`![gen breakdown timeline](timeline_genbreak.png)`

| step | run | rollout | postprocess | concat | transfer | phase_total |
|---|---|---|---|---|---|---|
| 1 | baseline | 193.94 | 2.17 | 15.94 | 54.54 | 242.84 |
| 1 | TQ       | 184.58 | n/a    | n/a    | 0.53     | 185.02 |
| 2 | baseline | 144.80 | 2.08 | 15.57 | 29.25 | 186.95 |
| 2 | TQ       | 140.84 | n/a    | n/a    | 0.23     | 142.02 |
| 3 | baseline | 196.48 | 2.12 | 15.58 | 40.94 | 257.15 |
| 3 | TQ       | 198.62 | n/a    | n/a    | 7.04     | 200.04 |

## 4. 解读

### 4.1 TQ 不影响 ML 计算

每个 step 的 `rollout_time`(纯 forward inference)在 baseline 与 TQ 间差异 ≤ 5%,主要由 sampling 不同导致 response 长度略不同,**有正有负**(step 3 TQ 反而比 baseline 慢 2 秒)。TQ 不是、也不可能是 ML 计算上的优化。

### 4.2 TQ 的全部收益在数据流

把 baseline 比 TQ 多花的时间归类:

**gen 阶段**(每 step 节省 ~50–60s):
完全来自 TQ 的「streaming write-back」消除了 baseline 必须做的 post-rollout 集中收尾——具体是 worker 内部的 `_postprocess`(每 step ~2s,小)、controller 端跨 worker 的 `DataProto.concat`(每 step ~15.6s,稳定)、以及 Ray Object Store 的序列化往返(每 step 30–55s,随 batch 大小放大)。在 TQ 路径下,每个 prompt 完成 rollout 后立即把结果写到 TQ 共享存储,这部分写回与还在跑的其他 prompt 的 rollout 重叠,只有最末尾 ~0.5–7s 的 tail 没盖住(`gen op=transfer_time`)。

**old_log_prob 阶段**(每 step 节省 ~50s):
最干净的 TQ 收益。baseline 把整个 batch 序列化进 Ray Object Store 发给 worker,worker 算完再序列化回控制端,双向各一次大数据搬运。TQ 路径下控制端发的是 `KVBatchMeta`(只是若干 keys 的 metadata),worker 直接从 TQ 取数据(zero-copy / 共享内存),算完直接 `kv_batch_put` 写回。`tq_get + tq_put` 三步合计 0.24–0.30s,**与 token 数无关**。这个稳定值是 KV 共享存储 metadata 操作的极限。

**update_actor 阶段**(每 step 节省 ~50s):
TQ 把整个 batch 通过 `tqbridge` 装饰器透明转发,数据搬运零开销;但 worker 端需要的整张 batch 仍要通过 RPC 传一次(从控制端发 KVBatchMeta + worker 自取),所以 TQ transfer 还有 ~10–14s,主要是 ppo_epoch 内多次 mini-batch dispatch 的 RPC 残留。

### 4.3 收益随 batch 大小放大

step 3 的 `total_num_tokens` 比 step 1 多 35%(1.72M vs 1.28M),对应 baseline 的 `transfer_time` 也从 ~30s 涨到 ~60s 区间;TQ 的 transfer 几乎不动。这表明 **TQ 的相对收益会随 token 数增长**。在更大的 model / 更长 response 的训练中,加速比应该 ≥ 我们这里测到的 30%。

## 5. 测量方法

每条 `[TRANSFER_TIMING]` 形如:

```
[TRANSFER_TIMING] step=N phase=PHASE op=OP elapsed_s=X bytes=B mode={tq,baseline}
```

仪器位置:

| 文件 | 仪器 |
|---|---|
| [`verl/utils/profiler/performance.py`](../verl/utils/profiler/performance.py) | `TransferTimeLogger` Ray actor + `log_worker_compute` (worker fire-and-forget) / `flush_worker_compute` (controller max) |
| [`verl/utils/transferqueue_utils.py`](../verl/utils/transferqueue_utils.py) | `tqbridge` 装饰器在 TQ 与 baseline 两条路径都上报 worker compute_time |
| [`verl/trainer/ppo/ray_trainer.py`](../verl/trainer/ppo/ray_trainer.py) | baseline 各 phase 测 `rpc_total`,与 `flush_worker_compute()` 配合得 `transfer_time` |
| [`verl/trainer/main_ppo_sync.py`](../verl/trainer/main_ppo_sync.py) | TQ 路径同上,加上 `tq_get` / `tq_put` 字节量与 gen 的 `rollout_time` |
| [`verl/experimental/agent_loop/agent_loop.py`](../verl/experimental/agent_loop/agent_loop.py) | gen 4-段拆分 (`rollout_time` / `postprocess_time` / `concat_time` / `transfer_time`) |

解析 + 绘图脚本: [`scripts/plot_timeline.py`](../scripts/plot_timeline.py)。

## 6. 已知方差

- **每 run 仅 3 step**,sampling 噪声占主导。结论方向稳定(三 step 趋势一致),具体数值需要更多 step 才能给出置信区间。
- **`flush_worker_compute()` 当前对所有 task 取全局 max**,跨 phase 上报有理论竞态;实测下未观察到异常,但建议改成 per-task flush(留作后续 cleanup)。
- **gen 的 4 段不严格相加 = phase_total**:每段都是「跨 worker 的 max」,而不同 worker 的最慢瞬间不一定是同一时刻;偏差在 5% 以内,不影响定性结论。

## 7. 结论

TransferQueue 在 verl 上的 30% 端到端加速由两个主要架构改造贡献:

1. **Zero-copy 共享存储替代 Ray Object Store**(`old_log_prob` / `update_actor` / `compute_values`):大数据 batch 不再走 Ray plasma 的 pickle/unpickle 来回,worker 直接在共享地址空间读写。这一项在 `old_log_prob` 上节省 99% 的传输时间。

2. **Streaming write-back 替代集中收尾**(`gen`):per-prompt 完成后立刻 push 到 TQ,与还在跑的其他 prompt 的 rollout 重叠,消除了 baseline 必须做的 worker 内 concat、controller 跨 worker concat、以及 Ray return 序列化(共 ~46–72s/step)。

ML 计算时间(`rollout_time` / 各 worker 的 `compute_time`)**未受影响**,这与 TQ 是数据层中间件的设计预期一致。
