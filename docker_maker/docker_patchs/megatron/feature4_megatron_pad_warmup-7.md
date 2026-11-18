# Pad-Warmup（feature4）：用首个 batch 预留峰值显存，消除变长 batch 的碎片 OOM

> **基线说明**：本环境的 `ms-swift` 是 4.3.1，但 pad-warmup 是 **feature4** 版本
> （`swift/megatron/trainers/pad_warmup.py` 为 feature4 新增文件；`base.py`、`template/base.py`
> 也已由 feature4 改过）。本文描述 **feature4 当前实现**，不是早期的 feature3
> （feature3 只支持非 padding_free、长度取 `min(cutoff, max_len(all_samples))`）。
>
> **数据版本**：本文所有实测数据为 **2026-09-24 复测**，环境为重构后的 `train_megatron.sh`
> + 新镜像（8 卡 XPU/P800，Qwen3.6-35B-A3B MoE，EP=8 / TP=PP=CP=1，alltoall dispatcher，
> `seq_length=20480`、`micro_batch_size=2`、`global_batch_size=16`、`padding_free=true`）。
> **旧文档记录的绝对数值已不可比**，仅在 §8.4 保留一处历史对照。
>
> **阅读顺序**：第 1~3 章是"结论 + 用法"；第 4~7 章是"技术细节"；第 8 章是当前已知问题；
> **第 9 章是架构评审的缺口清单与修复优先级（待办项的唯一事实来源）**。排查手法放附录。

## 1. 解决什么问题

变长 batch 训练时，caching allocator 会交错分配大量不同形状的显存块。等到某个峰值形状的张量
（比如 fused cross entropy 里的 fp32 logits `[seq_len, vocab/tp]`）需要一整块连续显存时，
空闲显存已被切得很碎——总空闲远大于需求，却拿不出连续的一块：

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 18.42 GiB. GPU 4 has a total
capacity of 96.00 GiB of which 44.29 GiB is free. Of the allocated memory 37.47 GiB is allocated
by PyTorch, and 8.57 GiB is reserved but unallocated.
```

这类 OOM 的特征：**总容量没满、空闲很多、但连续块不够**。

**解法**（同 LLaMA-Factory 的 `pad_first_batch_to`）：让**第一个 batch 直接按峰值形状 padding**，
allocator 一开始就把大块显存预留出来，之后所有 batch（需求 ≤ 它）都复用这些块，不再切碎。

## 2. 整体方案

### 2.1 一句话

训练开始前预先算出**每张卡本次训练会遇到的最大 micro-batch 长度**，把该卡的**第一个 batch
补到这个长度**，让 allocator 在 step0 就一次性开出峰值形状的显存块；此后所有 batch 都 ≤ 它，
直接复用，池子不再增长。

### 2.2 执行流程

```
1. 扫描数据集，得到每条样本的 token 长度
   （列式去重计划 → tail-first 顺序 → 分片 fork 池解码；磁盘缓存第二次起直接命中，见 §5）
2. 按长度覆盖情况选峰值模式（§9.7 第三批）：
   - 全扫完（exact）：重放 MegatronPretrainingRandomSampler，取「本 dp_rank × 所有 epoch」
     的最大 micro-batch 长度        ← 精确值，不是估算，见 §4
   - ≤5% 缺失（substituted）：缺失行按 max(有效) 顶替后重放，膨胀有界
   - 更多缺失（tail-estimate）：频率修正的 Monte Carlo 估计 ≈ 真值，见 §9.7
3. 预热集合通信后端（XCCL/BKCL）的通信子   ← 否则 step1 会 OOM，见 §7
4. settle：直接调 `init_state_fn` 把 optimizer 状态分配出来（fresh run 下它们本要到首个
   step() 才落地），然后 gc + empty_cache 清场   ← 否则常驻分配在 carve 之后落地、
   把切好的峰值块钉碎，见 §9.8
5. 第一个 batch 用一条 dummy 样本补到该长度（padding_free 下是 packed 行里的独立一段）
6. 运行期闭环校验：任一实际 batch 超过 warm-up 长度即 WARN（§9.7 第三批）
```

整个 run 每卡只做一次（§6）。

### 2.3 dummy 段方案（padding_free 也支持）

用一条 **dummy 样本**把 packed 行补到目标长度，`packing_row` 给它 `position_ids = arange(T)`，
使其成为 `cu_seqlens` 里**独立的一段**（分段边界是 `position_ids == 0`）。dummy 恒为
`labels=-100` / `loss_scale=0`，因果隔离且零 loss/梯度贡献。dummy 的 token 从真实 batch 采样
（不是单一 pad id），避免把 MoE router 的负载均衡带偏；且只从**纯文本 token** 采样——
多模态占位 token 的数量必须与 ViT 输出的 embedding 数严格一致，采到会越界访存。

### 2.4 效果（3 epoch，8 卡；峰值水位 GiB）

口径是 `max_memory_reserved`（**峰值水位**），即决定会不会 OOM 的那个量（见 §2.5）。
每个 run 实测**从第 1~2 步起即定格不动**，因此"最终峰值"就是稳态值。

| rank | 本卡峰值 | OFF step0 | OFF 最终峰值 | **ON step0** | ON 最终峰值 | ON step0 − OFF 峰值 | ON 峰值 − OFF 峰值 |
|---|---|---|---|---|---|---|---|
| 0 | 19146 | 49.46 | 67.48 | **56.79** | 56.79 | −10.69 ✓ | −10.69 ✓ |
| 1 | 16895 | 50.70 | 66.47 | **54.08** | 54.20 | −12.39 ✓ | −12.27 ✓ |
| 2 | 19007 | 49.31 | 86.91 | **63.62** | 63.63 | −23.29 ✓ | −23.28 ✓ |
| 3 | 24773 | 38.87 | 83.64 | **62.96** | 62.97 | −20.68 ✓ | −20.67 ✓ |
| 4 | 19908 | 44.20 | 85.24 | **62.70** | 62.74 | −22.54 ✓ | −22.50 ✓ |
| 5 | 18041 | 43.42 | 89.19 | **58.29** | 58.30 | −30.90 ✓ | −30.89 ✓ |
| 6 | 18677 | 54.22 | 91.30 | **58.38** | 58.39 | −32.92 ✓ | −32.91 ✓ |
| 7 | 15778 | 44.95 | 85.85 | **56.91** | 56.92 | −28.94 ✓ | −28.93 ✓ |

**8 卡合计：OFF 656.08 GiB → ON 473.94 GiB。ON step0 比 OFF 峰值低 182.35 GiB，
ON 全程峰值低 182.14 GiB。8/8 卡全部下降，无一变高。**

读数要点：

- `ON step0 − OFF step0` **全为正**（+3.38 ~ +24.09）——这是设计使然：把峰值提前到第一步支付。
- 该看的是最后两列：**提前支付的代价 < 事后被碎片顶高的代价**，8/8 卡成立。
- **ON 全程无跳变**（reserved 定格）；OFF 是阶梯式爬升，例如 rank3
  `38.87 → 48.96 → 60.15 → 83.64`（第 5 步一次 +23.49），rank6 爬到 91.30 GiB，
  距 96 GiB 上限只剩约 5 GiB。
- 复测中未出现任何一张卡 ON 高于 OFF 的情况（旧文档曾有一处 rank3 例外，见 §8.4）。

### 2.5 评判口径：看峰值水位，不是稳态 reserved

决定会不会 OOM 的是**峰值水位**（`max_memory_reserved`，池子瞬时冲到的最高点），
**不是稳态 reserved**。ON 的稳态通常高于 OFF（把所有中间 buffer 冻结在峰值尺寸），
但**峰值水位更低**——ON 把同一份需求从"尖峰"摊平成"常驻"。

复测中两者相等（ON 从第 1 步起定格），所以 §2.4 的表里"ON step0"与"ON 最终峰值"几乎相同，
这正是 pad-warmup 生效的标志。

### 2.6 v6 方案总览：为什么扫描更快，峰值却基本不多

一句话：**峰值只由长度分布的上尾决定——快，来自删掉与上尾无关的工作；
不多，来自凡是影响上尾的环节都保留精确值或做有界修正。**

每个加速点对应的"精度无损"理由：

| 加速点 | 收益 | 为什么不损精度 |
|---|---|---|
| 媒体载荷去重（§5.4） | 3395 → 485 条解码，6.6× | 去重删的是**信息冗余**不是信息：同行媒体逐 token 相同，代表行保留精确 encode 值，重复行只差自身文本（误差 ≤2 token 的 BPE 边界噪声，见 §5.4 实测） |
| 列式计划构建（§9.7 第四批） | 5 万行计划 0.28s（15×） | 只是读取方式换成 arrow 向量化，产出计划与行式**逐位一致**（demo 逐场景断言） |
| 单线程 fork 池（§5.4） | 1.77×（接近线性） | 纯并行度修复，不改任何长度语义 |
| 分片扫描（§5.5） | 墙钟 ÷ 节点数 | 各 rank 扫描结果经 crc 协商的计划聚合，交换的是**完整长度值**——合并后数组与单卡全量扫描逐位一致（demo B 验证） |
| 磁盘缓存（§5.1） | 第二次起 0.05s | 纯函数结果落盘，复测两次长度数组 `max_abs_diff = 0` |

**"不多"由一条四层紧度链保证**（欠预留是延迟死、过松是立即死，目标 `w − L*` 最小且 ≥ 0）：

1. **全扫完（exact）**：重放真实 sampler 是精确值而非估算——demo3 用 288 种
   （数据集 × 种子 × 配置）组合验证与真实训练逐 batch 一致（§4.2 / §9.7 第二批）。
2. **少量缺失（≤5%，substituted）**：缺失行按 `max(有效长度)` 顶替，膨胀有界且方向
   是"松"不是"欠"（§9.7 第三批）。
3. **大量缺失（tail-estimate）**：tail-first 排序保证任意截断点上尾已扫
   （50 万行扫 9.7% 即覆盖 100% 上尾）；截断子集的**尾部富集偏差**由频率修正消除——
   极端场景估计 = 真值的 100.0%（未修正会到 149.7%，顶替法会到 750%，见 §9.7）。
4. **运行期兜底**：`_check_overshoot` 对每个实际 batch 校验，超过预留即 WARN——
   一切"静默低估"路径的最后一道防线（§9.7 第三批）。

净效果：3395 条视频数据集扫描 496s → 46s（再 ÷ 节点数），50 万行规模下计划构建 0.28s；
峰值误差全扫时 ≤2 token，极端截断时方向略松、幅度 ≤ 几个百分点。

---

## 3. 配置、验证与生效确认

### 3.1 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `SWIFT_PAD_WARMUP` | **`0`（关闭）** | 设为 `1` 开启 |
| `SWIFT_PAD_WARMUP_LENGTH` | 无 | 显式指定目标长度，跳过扫描与重放 |
| `SWIFT_PAD_WARMUP_CACHE` | `1` | `0` 关闭长度扫描的磁盘缓存 |
| `SWIFT_PAD_WARMUP_CACHE_DIR` | `<cwd>/.pad_warmup_cache` | 换缓存目录（含 one-shot flag）；指向共享 FS 时一次扫描全集群复用 |

> 部署脚本 `env.sh` 里默认开启（`SWIFT_PAD_WARMUP=${SWIFT_PAD_WARMUP:-1}`）。
> 注意 `train_megatron.sh` 会**重新 `source env.sh`**，所以做 OFF 基线必须确认外层设的 `0`
> 没有被覆盖（见 §3.4）。

**自动关闭**（无需配置）：`context_parallel_size > 1`；`template.mode` 非 `transformers/train`
（RLHF/PRM 按 `packed_seq_params.seq_lens` 索引分段 logits，多一段会错位）；长度不可得；
`streaming=True`；扫描失败率 > 5%。

### 3.2 交付物与 patch 验证

| 文件 | 内容 |
|---|---|
| `feature4_megatron_pad_warmup.patch` | **当前版本（v7）**：功能补丁（v7 = v6 + §9.8 settle-then-carve + §9.9 dispatcher 路由状态释放，`pad_warmup.py` 1991 行；v6 = v5 + §9.7 的 P0 修复、复查修复、第二批正确性修复、第三批 tail-first 重设计与第四批列式计划构建（含峰值估计频率修正）） |
| `feature4_megatron_pad_warmup-6.patch` | v6 封存版（`pad_warmup.py` 1879 行，§9.7 全部内容），仅作历史基线 |
| `feature4_megatron_pad_warmup-5.patch` | v5 封存版（1192 行，含 §5.4 扫描加速与 §5.5 分片扫描），仅作历史基线，勿用于新部署 |
| `feature4_demo/pad_warmup.py` | 当前 `pad_warmup.py` 单文件（patch 的构建源，单一事实源；直接阅读/diff 用它） |
| `feature4_demo/pad_warmup_v6.py` | v6 封存的 `pad_warmup.py` 单文件（1879 行，历史基线） |
| `feature4_demo/pad_warmup_v5.py` | v5 封存的 `pad_warmup.py` 单文件（1034 行，历史基线） |
| `feature4_demo/demo_dedup_key.py` | 本地 demo：去重键健壮性（§5.6） |
| `feature4_demo/demo_sharded_scan.py` | 本地 demo：多进程真实集合通信下的分片扫描效果（§5.6，v6 起含场景 F/G/H/I/K，§9.7） |
| `feature4_demo/demo_replay_vectorized.py` | 本地 demo：向量化 replay 与真实 sampler 逐 batch 比特一致（§9.7 第二批） |
| `feature4_demo/demo_tail_first.py` | 本地 demo：tail-first 排序、三档峰值模式、闭环校验（§9.7 第三批） |
| `feature4_demo/demo_columnar_plan.py` | 容器内 demo（真实 datasets/pyarrow）：列式计划构建、decode=False 零解码读取、去重恢复（§9.7 第四批） |
| `feature4_demo/demo_tail_first_scale.py` | 本地 demo：50 万行成簇重尾数据集下 tail-first 的 anytime 收益量化（§9.7 第四批复查） |
| `feature4_demo/demo_settle.py` | 本地 demo：settle-then-carve 的 optimizer 状态落地、幂等、降级（§9.8） |
| `feature4_megatron_pad_warmup.md` | 本文档 |
| `PAD_WARMUP_HANDOVER.md` | 交接/复测记录（含更完整的原始数据） |

patch 覆盖 `swift/megatron/trainers/base.py`（2 hunk，v7 起调用点传 `optimizer=self.optimizer`）、
`swift/megatron/trainers/pad_warmup.py`（新增，v7 为 1991 行）、`swift/template/base.py`（3 hunk）。
下方验证脚本对 v5/v6/v7 同样适用，历史版本把文件名换成对应的 `-N.patch` 即可。

扫描加速（§5.4）与分片扫描（§5.5）**全部内聚在 `pad_warmup.py` 内部**：不额外改动
`base.py` / `template/base.py`。去重、分片与并行度都是默认行为，**功能代码中不预留
排障后门环境变量**。

> **`feature4_xccl_timeout_fix.patch` 已废弃，不要再叠加应用**：它"用文件锁做跨 rank
> rendezvous"的思路已被 §5.5 的分片扫描取代。文件锁/文件轮询都有一个共同缺陷：
> 对端进程挂掉时，其余 rank **无声地永远卡住，连卡在哪个锁上都看不到**；
> 分片扫描用集合通信交换结果，任何一端崩溃都会让 collective 立即报错暴露。

**验证方式**（2026-09-24 复跑通过，3/3 逐字节一致）：

```bash
mkdir -p /tmp/v/swift/megatron/trainers /tmp/v/swift/template
SP=$(python -c 'import swift, os; print(os.path.dirname(os.path.dirname(swift.__file__)))')
cp $SP/swift/megatron/trainers/base.py $SP/swift/megatron/trainers/pad_warmup.py /tmp/v/swift/megatron/trainers/
cp $SP/swift/template/base.py /tmp/v/swift/template/
cd /tmp/v && patch -R -p1 < .../feature4_megatron_pad_warmup.patch \
  && patch -p1 < .../feature4_megatron_pad_warmup.patch
for f in swift/megatron/trainers/base.py swift/megatron/trainers/pad_warmup.py swift/template/base.py; do
  diff -q /tmp/v/$f $SP/$f >/dev/null && echo "OK $f" || echo "DIFF $f"; done
```

### 3.3 确认生效的日志（复测实录）

```
[pad_warmup] Scanning 156 samples (num_proc=32)...
[pad_warmup] Scan finished: 156 rows, 156 valid, 0 failed, mean=5662, max=13878, took 1.72s.
[pad_warmup] Replayed 3 epoch(s) for dp_rank=[0]: peak sum of 2 consecutive sample(s) = 19146.
[pad_warmup] Warmed up 7 communicator(s) before the peak-shaped batch: default(8), ...
[pad_warmup] warmup: pad the first 1 batch(es) per dataloader worker to 19146 (padding_free=True).
[pad_warmup] Padding the first batch to 19146 (pid 25362).
```

关键看四件事：**扫描耗时**、**重放出的峰值**、**通信子预热个数**、**实际 pad 的长度**。
命中缓存时前两行替换为 `Using the cached sample lengths: 156 rows, 0 failed (0.03s, ...)`。

视频数据集的扫描会走去重路径，第一行变成 `485/3395` 这种形式（见 §5.4）：

```
[pad_warmup] Scanning 485/3395 samples (num_proc=64, dedup by media payload)...
[pad_warmup] Scan finished: 3395 rows, 3395 valid, 0 failed, mean=5922, max=8117, took 46.30s.
```

> 注意：训练日志里的 `memory(GiB)` 是聚合后的值，**不能用来横向比较各卡**。

### 3.4 注意事项

1. **`SWIFT_PAD_WARMUP=0` 可能被静默改回 1**：`train_megatron.sh` 会重新 `source env.sh`，
   若 `env.sh` 里写死 `export SWIFT_PAD_WARMUP=1`，外层设的 `0` 会被覆盖掉。
   **关掉功能后务必 grep 日志确认没有 `pad_warmup` 字样**。建议 `env.sh` 写成
   `export SWIFT_PAD_WARMUP=${SWIFT_PAD_WARMUP:-1}`——默认开，但允许外部显式覆盖。
2. **长度缓存与 one-shot flag 都落在 `<cwd>/.pad_warmup_cache/`**，是一次性产物。
   换数据集 / 模型 / 模板 / 多模态像素配置后会失效并触发重扫（见 §5.2）；
   分片扫描（§5.5）后**每个 rank 都会写缓存**，节点本地盘场景下各节点各留一份，
   下次启动全员命中；把 `SWIFT_PAD_WARMUP_CACHE_DIR` 指向共享 FS 则一次扫描全集群复用。

---

## 4. padding 长度怎么选

### 4.1 为什么不用 `Σ top-micro_batch_size`

旧实现取 `min(mbs * max_length, Σ 数据集最长的 mbs 条样本长度)`，即**所有 `C(n, mbs)` 种组合里
的最大值**——等价于假设"最长的几条样本恰好落在同一个 micro-batch"，随机 shuffle 下概率约
`1/C(n, mbs)`，基本不可能。本数据集（156 条，mbs=2）该值 = 25314，而典型 batch 只有 11323，
**是它的 2.24 倍**。按它预留会把 step0 直接打成整 run 的峰值。

### 4.2 重放 sampler（精确，非估计）

`MegatronPretrainingRandomSampler` 的 shuffle 是**确定性**的（种子就是 epoch）：

```python
g = torch.Generator(); g.manual_seed(self.epoch)      # seed = epoch
random_idx = torch.randperm(bucket_size, generator=g).tolist()
```

所以可以精确重放每个 epoch、每个 dp_rank 的 micro-batch 划分，直接取最大值。`replay_peak_batch_length()`
干的就是这个——**不需要任何分位数 / 覆盖率 / 魔数**，天然适配任意长度分布、`mbs`、dp 大小、
epoch 数、`data_sharding`、`group_by_length`。

复测结果与实测逐卡吻合到 1~2 个 token（差值是段对齐造成的）。离线直接调
`replay_peak_batch_length()` 对每个 `dp_rank` 各重放一次即可复算（做法见 §3.4），
本数据集 3 epoch 的结果：

```
rank0..7: 19146 16895 19007 24773 19908 18041 18677 15778
```

> `_REPLAY_MAX_SAMPLES = 50_000_000`：`n_epochs × 样本数` 超限时退化成只重放 epoch 0 并告警，
> 此时预留会**低于后面 epoch 的真实峰值**——即"预留不足 → 峰值 batch 拿不到连续块"的 OOM 机理。

### 4.3 只按「本卡」峰值（不是全局峰值）

每张卡有自己的设备，所以只该覆盖**它自己会遇到的 batch**。复测中各卡自己的峰值只有全局峰值的
64%~100%（平均 76%）。

**但复测也证明这个前提是近似的**：r2/r3/r4 的本卡 warm-up 长度在 3ep 与 20ep 下**完全相同**
（19007 / 24773 / 19908），reserved 却分别 +2.36 / +3.10 / +0.22 GiB。即**本卡 reserved 不是
本卡长度的函数**——EP=8 + alltoall 下存在跨卡耦合，详见 §8.2 / §8.3。

### 4.4 版本演进

| 版本 | 做法 | 结果 |
|---|---|---|
| **v0** | 原始 feature4：`Σ top-mbs`、全局峰值、按 worker 计数 | step0 XCCL `-705` 崩（目标值 25314 比典型 batch 大 2.24 倍） |
| **v1** | v0 + 通信子预热 | step0 过，但第 15 步碎片 OOM（18.42 GiB）——只解决时序，没解决"预留过大" |
| **v2** | 改重放 sampler + 全局峰值 + 每 worker 计数 | step0 降了，但每 epoch 被再预热一次（曾在第 10 步造出 23.42 GiB 人为峰值） |
| **v3** | 重放 + **按本卡峰值** + **全局只预热一次** + 逐次日志 | 复测 8/8 卡优于 OFF，无跳变，0 OOM |
| **v4** | v3 + 扫描加速（按媒体载荷去重 + 单线程多进程，见 §5.4） | 扫描 496.4s → 46.3s，长度误差 ≤2 token |
| **v5** | v4 + **分片扫描**（去重后的媒体行交错分片到各卡并行解码，一次小规模 all_gather 交换结果；缓存全 rank 直读直写，见 §5.5） | 首轮扫描墙钟 ≈ 总解码量 ÷（节点数 × 单节点预算）；集合通信内最长等待 = 分片完成时间差，机制级消除 702 |
| **v6** | v5 + **分布式兜底**（独立 gloo PG / 扫描 deadline / 单样本超时 / 启用协商）+ replay 向量化（epoch 均匀采样）+ **tail-first anytime 扫描**（三档峰值模式 + 闭环校验）+ **列式计划构建**（含 `decode=False` 零解码读取，恢复 HF Image/Video 去重），见 §9.7 | 扫描成为 anytime 算法：扫不完也能拿到 ≈真值 的峰值上界（50 万行截到 30% 仍 100.0%）；计划构建 5 万行 0.28s（15×）；见 §2.6 |
| **v7（当前）** | v6 + **settle-then-carve**：carve 前直接调 `init_state_fn` 落地 optimizer 状态（fresh run 下它们本要到首个 `step()` 才分配）+ gc/empty_cache 清场（§9.8）+ **MoE dispatcher 路由状态释放**（上游 #8297/#8231 backport，§9.9） | 峰值轨迹在"常驻分配已落地、堆已清净"的状态上雕刻；carve 之后也不再有跨步钉住的路由张量在块内打洞 |

配套清理：显存预算相关的 `_fits_in_headroom` / `_peak_bytes_per_token` / `_vocab_size` /
`_PEAK_BYTES_FACTOR` 已删除。

---

## 5. 长度扫描与磁盘缓存

### 5.1 为什么需要缓存

确定 warm-up 长度有两类代价：重放 sampler 是纯整数运算（微秒级，无需缓存）；
**逐条 `template.encode` 求长度很重**——多模态要读文件、抽帧、缩放，单条可达百毫秒级。
后者是**纯函数**（同一份数据 + 同一套配置 ⇒ 结果逐字节相同），所以整体落盘复用。

复测：156 条首次扫描 1.72 s，命中缓存 0.03 s，两次重放出的峰值都是 19146。

> 视频数据集上单条 encode 可达秒级（实测 962 ms/条，其中解码抽帧占 86%），一份 3395 条的数据集
> 扫一遍要 8 分钟。§5.4 讲的就是怎么把它压到 46 s。

### 5.2 缓存 key 的判定原则

> **凡会改变"某条样本被 tokenize 后有多长"的因素都要进 key；只影响重放组织的不能进。**

二者不对称：误进 key 的代价只是多扫一次，漏进 key 的代价是"用了错误长度 → 预留不足 → OOM"。
所以**偏向保守**，并用版本安全网兜底。

**必须进 key**：

| 类别 | 字段 |
|---|---|
| 样本内容与子集 | `dataset`（含 `#N`）、本地文件指纹（`size`+`mtime_ns`）、`split_dataset_ratio`、`columns`、`custom_dataset_info`、`dataset_shuffle` |
| tokenizer / processor | `model`、`model_revision` |
| 模板与截断 | `template` `system` `max_length` `truncation_strategy` `max_pixels` `agent_template` `norm_bbox` `use_chat_template` `padding_free` `template_backend` `response_prefix` `enable_thinking` `preserve_thinking` `add_non_thinking_prefix` |
| 多模态展开 | `MAX_PIXELS` `MIN_PIXELS` `VIDEO_MAX_PIXELS` `VIDEO_MIN_PIXELS` `VIDEO_TOTAL_PIXELS` `FPS` `FPS_MAX_FRAMES` |
| 安全网 | swift 版本、template/processor 类名 |

**不能进 key（只影响重放组织）**：`train_iters` `num_train_epochs` `global_batch_size`
`micro_batch_size` DP 大小/秩 `data_sharding` `train_dataloader_shuffle` `group_by_length`
`seq_length` `dataset_num_proc` `output_dir` `SWIFT_PAD_WARMUP_LENGTH`。

> 安全网的作用：swift 版本与 processor 类名进 key，框架升级后旧条目自动失效；即使将来新增了
> 某个影响长度的参数没被列出，也只会退化成"重扫一次"，不会算出错误长度。
>
> **实践提醒**：改 `FPS_MAX_FRAMES` / `VIDEO_MAX_PIXELS` 会**让整份缓存失效并触发全数据集重扫**，
> 视频数据集扫一次很贵，建议一次调到位。

### 5.3 两条不变量与失败样本

- **不变量 1 —— 一条样本一个槽位**：`encode` 失败的样本**保留槽位填 `0`，绝不丢弃**。
  否则数组变短后 `lengths[i]` 描述的就不是第 `i` 条样本，重放会组合出错误的峰值（可能偏小）。
  配套地 `replay_peak_batch_length` 会用 `len(hf_ds)` 校验 `len(lengths)`，不一致直接拒绝并关闭。
- **不变量 2 —— 缓存条目必须与数据集对得上**：读取时校验条数，不符即作废重扫；写入用
  `os.replace` **原子替换**（临时文件名带 pid），进程被杀或并发跑都不会留下半截文件。

失败样本的处理原则是**宁可不预留，也不预留错**：

| 情况 | 处理 |
|---|---|
| 失败率 > 5% | 放弃 pad-warmup，回退基线 |
| 失败率 ≤ 5%，且设了 `max_length` | 重放时该样本按 `max_length` 计入 —— 只会**高估**不会低估 |
| 失败率 ≤ 5%，且无 `max_length` | 放弃 pad-warmup（没有可用上界） |

**明确不缓存/不走扫描**：`streaming=True`（`IterableDataset` 不可重复迭代）；
数据集自带 `lengths`/`length` 列（直接读取，连扫描都不用做）；`packing=True`
（行是多样本拼接，直接用 `max_length`）。

> 不要用"只扫前 N 条再估算"来提速：那会系统性低估峰值，正好落在预留不足的 OOM 机理上。
> 要提速就按 §5.4 的做法——**同样扫全量，但每条只解码一次**。

### 5.4 扫描加速：按媒体载荷去重 + 单线程多进程

扫描耗时 ≈ `样本数 × 单条 encode 成本 / 有效并行度`。样本数只是乘数里的小项，**单条成本才是杠杆**：

| 项 | 数值（实测，host） |
|---|---|
| 单条 `template.encode`（含视频） | **962 ms**（中位 877，max 1992），n=24 |
| 其中视频解码 + 抽帧 | **823 ms（占 86%）**，20~60 帧 |
| 单条 encode（同一数据集，**去掉 videos**） | **0.93 ms** |
| 纯 tokenizer 吞吐 | 0.117 ms/次（8554 次/s） |

这就解释了"**为什么样本更多的数据集反而扫得更快**"：视频单条 = 纯文本单条的 **1030 倍**，
3395 条视频 ≈ **350 万条纯文本**的扫描成本。冷 IO 与热 page cache 实测几乎一样
（490.4 s vs 493.9 s），说明瓶颈是 CPU 解码，不是读盘。

两处改动，全部在 `pad_warmup.py` 内，**没有新增任何环境变量**（默认全开，无开关）：

| 改动 | 做法 | 单独收益 |
|---|---|---|
| **去重** | `_scan_dedup_plan()` 按 `(videos, images, audios)` 分组，同组只解码代表行；其余行由 `_expand_dedup()` 用「代表长度 + 自身文本长度差」还原 | 3395 → 485 条，**6.6×** |
| **并行度** | `_scan_num_proc()`：`min(32, dataset_num_proc×4)` → `min(64, ×16)`；`_scan_worker_init()` 把每个 worker 固定为 1 线程 | **1.77×**，load 340 → 167 |

去重为什么安全：样本长度 = `media_tokens + text_tokens + 模板常数`，同组 media 部分完全相同
（同一文件、同一套像素/帧配置），所以重复行只差自己的文本，按文本差修正即可；**代表行保留精确
encode 值**，代表行失败则整组记 0（保守方向，走 §5.3 的失败率逻辑）。

去重键的构造对真实数据形态是健壮的（`_freeze_media` 递归冻结）：媒体列不只是字符串或
字符串列表——HF datasets 会把媒体特征解码成 dict（`{'path': ..., 'bytes': ...}`），还有按
对话轮次嵌套的 list-list、tuple/set、数字混合等。旧实现 `tuple(value)` 遇到 dict / 嵌套 list
直接抛 `unhashable type`，整份去重计划作废退回全量扫描。现在递归冻结为可哈希规范形式
（dict 按键排序），这些形态都能**正常去重**；单行取键异常只影响该行本身。

**叶子只接受 str 与数字，其余一律不去重**（这是刻意的保守设计）：`str(obj)` 对别的类型
并不构成身份——`PIL.Image` 的 repr 里是**会被复用的内存地址**
（`<PIL.Image.Image ... at 0x7f...>`，行被释放后地址重分配 → 两张不同的图可能撞成同一键），
numpy 数组的 repr 是**截断**的（`[1 2 3 ... 998 999]`）。任何一种误合并都会静默算错峰值；
拒绝去重只是慢一点。数字带类型标签入键，所以 `2` 与 `'2'` 不会碰撞。

并行度为什么这么改：原先多进程 × 多线程互相超订——16 个 worker 各自开满线程把 384 核打满
（load 340），却只拿到 **6.7×**（效率 42%）。改成"多进程 + 每进程 1 线程"后接近线性。

**A/B（同一进程内，`SWIFT_PAD_WARMUP_CACHE=0`，3395 条）**：

| 模式 | 扫描条数 | 耗时 | mean / max |
|---|---|---|---|
| 原实现 | 3395 | 496.4 s | 5921.6 / 8117 |
| 现在 | **485** | **46.3 s**（复跑 80.2 s，含冷启动） | 5921.6 / 8117 |

**精度**：逐行 diff `max_abs = 2 token`、`mean_abs = 0.04`、**96.1% 的行完全一致**；
`mean / median / max`（5921.6 / 7316 / 8117）三者相同 → 重放出的 warm-up 长度不受影响。

**环境实测**（2026-09-28，8 卡 P800 + 真实训练任务，`SWIFT_PAD_WARMUP_CACHE=0` 强制重扫）：

```
[pad_warmup] Scanning 485/3328 samples (num_proc=64, dedup by media payload)...
[pad_warmup] Scan finished: 3328 rows, 3328 valid, 0 failed, mean=16339, max=49472, took 79.69s.
```

与 host 独占实测的 80.2 s 吻合（这份 run 的 `VIDEO_MAX_PIXELS=401408`，所以 mean/max 与上面
`262144` 那组不同）。缓存路径也复测过：首次 86.84 s 写入，第二次命中 **0.05 s**，两次长度数组
**逐条完全一致**（`max_abs_diff = 0`）。

> 副作用：扫描从 8 分钟压到 80 秒后，**"其他 rank 在 `dist.broadcast` 上干等 → XCCL watchdog
> `702` launch timeout"** 这个现象不再出现（本次跑到 it 3 全程 0 次 702、0 OOM）。
> 但这只是"干等时间变短"的缓解——数据集大到扫描仍需十几分钟时仍会撞 watchdog。
> 机制级根治已由 v5 的分片扫描完成（§5.5），原文件锁方案（`feature4_xccl_timeout_fix.patch`）
> 同时被取代。

### 5.5 分片扫描：多卡并行 + 一次 all_gather 聚合（v5）

动机有二：**超时根治**（单卡扫描时其余 N-1 卡全程阻塞在 broadcast 里，慢操作 × 集合通信的耦合
正是 702 的根因）与**首轮加速**（数万条唯一视频时去重无收益，单节点 CPU 就是天花板——分片把
墙钟压到 `总解码量 ÷（节点数 × 单节点预算）`）。

做法（全部在 `pad_warmup.py` 内）：

1. **先查缓存**：每个 rank 直读磁盘缓存，用一次 1 元素 `all_gather` 交换命中标志
   （毫秒级，人人瞬时到达）。全部命中 → 各用本地缓存，**零后续通信**；部分命中 →
   由最小此类 rank broadcast 给其余（源只是读本地文件，集合通信内无慢等待），
   且**缺条目的节点收到后落盘**，使下次启动收敛为"全员命中、零通信"；
   全不命中 → 进入下面的分片扫描。
2. **去重计划先行**：每个 rank 本地跑 `_scan_dedup_plan()`（只读行元数据，秒级），分片在
   **唯一媒体粒度**上做——同一视频不会被多张卡重复解码，§5.4 的去重收益完整保留。
3. **分片前协商计划**（`_agree_dedup_plan`，2 个整数的 all_gather）：分片布局
   （`counts`、每个聚合值归属哪一行）全部由 `todo` 推导，所以各 rank 必须持有**逐位相同**的
   计划。正常情况下它是数据集的确定性函数，但 `_dedup_plan_or_all` 在读行失败时会退化成
   `range(n)`，而这可能只发生在**一张卡**上——那样分片形状就不一致，结果 all_gather 直接
   挂住或中途 abort。协商发现不一致时，**所有 rank 一起**退回全量扫描（慢但形状一致）。
4. **交错分片**：rank r 扫 `todo[r::world]`。等行数 ≠ 等时间（视频时长差异大），交错让每片
   都混有长短视频，把分片完成时间差（聚合时的真实等待）压到最小。每 rank 的 fork 池缩为
   `预算 ÷ local_world_size`（`_scan_num_proc_sharded`）：单节点总算力不变，多节点线性扩展。
5. **结果交换只有一次 all_gather**：各 rank 把本分片的长度值等长 padding 后聚合
   （通信量 ≈ n×8B，5 万样本约 400KB），每 rank 本地合并 + `_expand_dedup` 还原全量数组。
   **集合通信内的最长等待 = 分片完成时间差**，不再是全量扫描时长。
6. **缓存每节点写一次**（`_is_local_main()`，即 `LOCAL_RANK==0`）：所有 rank 都重建出同一个
   数组，让 `world_size` 个进程并发写同一个多 MB 文件纯属浪费（50 万样本的 JSON 约 3MB，
   8 机 64 卡就是 64 次并发写）。每节点一个写者同时覆盖两种布局：节点本地盘各留一份，
   共享盘只是几次冗余但原子且内容相同的写入。下次启动全员命中、扫描与通信全免。

**明确不用文件锁/文件轮询做 rendezvous**：对端进程挂掉时，其余 rank 会无声地永远卡住，
连卡在哪个锁上都看不到；走集合通信则任何一端崩溃都会让 collective 立即报错、现场清晰。
某 rank 扫描整体失败时贡献全 0 分片（不拆散聚合），失败率超 5% 走 §5.3 的保守关闭逻辑。
旧的"单卡扫描 + broadcast"路径已从功能代码中删除（不留排障后门），A/B 对照可由
`feature4_demo/demo_sharded_scan.py` 的场景 A 用内部函数模拟复现。

> 注意：分片的加速比来自**节点数**（CPU/带宽），同节点内的 rank 共享同一份 CPU 预算；
> 多机共享 FS（如 /cfs）下建议把 `SWIFT_PAD_WARMUP_CACHE_DIR` 指过去，一次扫描全集群复用。

### 5.6 本地 demo 验证（2026-11-03）

两个 demo 直接加载**补丁里的** `pad_warmup.py`（单一事实源），无需训练环境。

**demo1 `demo_dedup_key.py`**（无需 torch）：18 项去重键检查全过——str / list-of-str /
dict（乱序同键）/ list-dict / list-list / tuple+数字 / 多列组合都能正常去重；
bytes、PIL 风格对象（地址会复用）、ndarray 风格对象（repr 截断）一律不去重；
`2` 与 `'2'` 不碰撞；无媒体或空列不去重；端到端 30 行 dict 视频 → 10 代表 + 2 异形行。

**demo2 `demo_sharded_scan.py`**（torch CPU + gloo 多进程真实集合通信）：
6600 行 = 1600 个唯一视频 × 4 QA + 200 纯文本，每视频模拟解码 15 ms。

> 注意：扫描的产出是**全量逐样本长度数组**（日志里的 `max=` 只是最长单样本的自检统计），
> **不是 padding 目标长度**。padding 目标长度由各卡把该数组喂给
> `replay_peak_batch_length(..., dp_rank=本卡)` 重放自己的 sampler 得出（§4.2 / §4.3），
> 这里验证的只是"扫描结果在所有 rank 上逐位一致"这一前提。
world=4 的完整场景矩阵：

| 场景 | 各 rank 耗时(s) | 集合通信内等待(s) | n_failed | 判定 |
|---|---|---|---|---|
| A 旧版 rank0 扫描 + broadcast | [1.91, 1.90, 1.90, 1.91] | [0.01, **1.90, 1.90, 1.90**] | 0 | ✅ 与参考逐条一致 |
| **B 分片扫描 + all_gather** | [0.73, 0.73, 0.73, 0.72] | [0.02, 0.02, 0.01, 0.04] | 0 | ✅ 与参考逐条一致 |
| B2 全员缓存命中 | ~0.005 | **0（无通信）** | 0 | ✅ |
| D rank1 视频解码全部失败（注入） | [0.70, 0.71, 0.71, 0.71] | [0.01, 0.47, 0.01, 0.05] | **1600** | ✅ 各 rank 收敛同一数组，无卡死 |
| E rank2 去重计划退化（注入） | [1.95, 1.95, 1.95, 1.96] | [0.01, 0.01, 0.00, 0.00] | 0 | ✅ 四卡一起退回全量，**不挂死**，与参考一致 |
| C 仅 rank0 有缓存 → 广播 | ~0.012 | ~0.004 | 0 | ✅ 与参考逐条一致 |
| C2 紧接 C 再跑一次 | ~0.005 | **0（全员命中）** | 0 | ✅ 验证缓存收敛 |

world 扩展性（同机 8 核，每 rank 16 个 sleep worker）：

| world | A 旧版墙钟 | B 分片墙钟 | 加速比 | B 的通信内最长等待 |
|---|---|---|---|---|
| 2 | 1.94 s | 1.06 s | 1.8× | 9 ms |
| 4 | 1.91 s | 0.73 s | 2.6× | 37 ms |
| 8 | 1.97 s | 0.78 s | 2.5× | 213 ms |

读法：

- **A 复现了 702 机理**：所有非 0 rank 在 broadcast 里干等**整个扫描时长**（~1.9s）；
  真实视频数据集上这是几十分钟量级，必然撞上 XCCL watchdog。
- **B 把通信内等待压到分片完成时间差**（毫秒级），wall-clock 同步下降。本机 8 核限制下
  w=4/w=8 同为 ~0.75s（128 个 sleeper 抢 8 核）；多机时每 rank 独占一个节点的 CPU 预算，
  加速比随节点数近似线性。
- **D 验证保守降级**：某 rank 分片全灭 → 该分片贡献 0（400 代表行 + 1200 重复行 = 1600，
  25% > 5% 阈值，真实流程会走 §5.3 关闭 pad-warmup），聚合不拆散、各 rank 收敛一致；
  失败的 rank1 反而最早扫完（全失败无解码），在 all_gather 里等最慢分片 0.47s——
  等的是"分片差"，不是"全量扫描"。
- **E 验证计划协商**（review 时补的用例）：人为让 rank2 的去重计划退化成 `range(n)`，
  四个 rank 都打印 `The ranks disagree on the dedup plan ([(1800, ...), (6600, ...)])`
  并**一起**退回全量扫描（1.96s，失去去重加速但形状一致），结果与参考逐条一致。
  **没有这一步协商，这个场景就是 all_gather 形状不匹配 → 多卡挂死。**
- **C2 验证缓存收敛**（review 时补的用例）：C 里缺条目的 rank 在收到广播后落盘，
  紧接着再跑一次就是"全员命中、零通信"——节点本地盘的集群第二次启动起不再有任何
  扫描或通信开销。

复现：`cd feature4_demo && python3 demo_dedup_key.py && python3 demo_sharded_scan.py 4 1600`
（demo2 需要 `pip install torch --index-url https://download.pytorch.org/whl/cpu`）。

> `feature4_demo/pad_warmup.py` 就是当前实现本身（`build_feature4_patch.py` 以它为源构建
> patch，往返提取逐字节一致）；`pad_warmup_v5.py` 是 v5 历史基线。

> 扫描时的 `mmco: unref short failure` 刷屏是**日志观感问题，不影响性能也不影响长度**，
> 不在本 patch 范围内，见附录 D。

---

## 6. 只预热一次（全局）

`PadWarmupCollator` 跑在 dataloader worker 里，而 `persistent_workers=False` 时
**每个 epoch 都会 fork 出新 worker**，拿到父进程里 `num_batches=1` 的副本——per-process 计数会
导致每个 epoch 的首个 batch 都被再 pad 一次（v2 的 bug，曾在第 10 步制造 23.42 GiB 人为峰值）。

现在除 per-process 计数外，还用 `O_CREAT|O_EXCL` 原子抢占 flag 文件
（`./.pad_warmup_cache/warmed_<hash(output_dir,local_rank)>.flag`），保证整个 run 每卡只 pad 一次。

**复测实证**：eval 阶段三个新 fork 的 worker 全部打印 `Warm-up already done, skip padding (pid ...)`
——train 先 pad，eval 的全部新 worker 都被 flag 拦住。

train / val 共用同一个 `data_collator`，所以谁先跑到就谁来 pad；若某个流程先跑验证，
这次 pad 会落在 eval batch 上，训练的第一个 batch 就不再 pad。**功能仍然只做一次**，
但"把峰值放到训练第一步"的意图会被挪到 eval，调配置时需留意。

---

## 7. warmup 时的通信子预热（必需）

**现象**：开启 pad-warmup 后在 step 1 崩溃：

```
[ERR][XCCL][kl3_init.cpp:237] param 10 data alloc failed
ValueError: [RUNTIME ERROR]: error code= 2, out of memory (at .../xccl/XCCLUtils.cpp:28)
```

**根因**：集合通信后端（XCCL/BKCL）在**某个 process group 第一次做集合通信时**才建 context，
且走**裸设备 malloc，绕过 torch 缓存分配器**。不开 pad-warmup 时这件事发生在小 batch 的 step 1；
开了之后 step 1 **就是峰值步**，设备已被占满 → 后端拿不到内存。

**即：不是峰值 batch 装不下，是通信子初始化的时机被挤到了显存最紧张的第 1 步。**

**做法**：`warmup_communicators()` 对每个非单卡 process group 各跑一次 1 元素 `all_reduce`，
把分配挪到设备还空的时候。在 `wrap_pad_warmup()` 中无条件调用（try/except 兜底）。

实现上有两个必须注意的点（都已用单测锁住）：

- **动态枚举** `dir(parallel_state)` 中所有 `get_*_group`，而不是手写列表——手写会漏掉
  `get_expert_model_parallel_group`（就是本配置的 EP 组）。
- **去重必须用对象身份 `id(group)`，不能用 rank 集合**——本配置下 `get_data_parallel_group()`
  与 `get_expert_model_parallel_group()` 的 rank 集合都是全部 8 张卡，但它们是**两个独立
  ProcessGroup 对象、各自一份 XCCL context**。按 rank 集合去重会只预热 1 个、漏掉另 1 个
  （实测会退化成只预热 2 个通信子）。
- 跳过 `world_size <= 1`（Megatron 对单卡组短路，不会建 context）。

**复测**：预热 7 个通信子（全部 size=8）：`default`、`get_amax_reduction_group`、
`get_data_parallel_group`、`get_expert_model_parallel_group`、
`get_expert_tensor_and_model_parallel_group`、`get_expert_tensor_model_pipeline_parallel_group`、
`get_intra_distributed_optimizer_instance_group`。整体开销约 2.3 GiB（每个约 330 MiB）。

---

## 8. 已知问题

### 8.1 warm-up 长度随 epoch 数变化（存在，二阶效应）

`MegatronPretrainingRandomSampler` 的 epoch 即 seed，每个 epoch 一个不同排列，所以各 epoch 的
micro-batch 长度峰值不同。重放 N 个 epoch 取最大值，**N 越大观测到的最大值单调不减**。
离线复算（rank0）：1ep 19146 / 20ep 22199 / 40ep 24087 / 60ep 24087。

于是**各卡峰值随 epoch 数变化，ON 的 reserved 也随之变化**。复测（8 卡，GiB）：

| rank | 1ep | 3ep | 20ep | 3ep→20ep Δ | 本卡长度 1ep / 3ep / 20ep |
|---|---|---|---|---|---|
| 0 | 56.65 | 56.79 | 60.16 | +3.37 | 19146 / 19146 / 22199 |
| 1 | 52.97 | 54.20 | 58.15 | +3.95 | 16895 / 16895 / 20685 |
| 2 | 59.11 | 63.63 | 65.99 | +2.36 | 15843 / **19007 / 19007** |
| 3 | 65.31 | 62.97 | 66.07 | +3.10 | 24773 / **24773 / 24773** |
| 4 | 50.92 | 62.74 | 62.96 | +0.22 | 13885 / **19908 / 19908** |
| 5 | 58.15 | 58.30 | 61.53 | +3.23 | 17705 / 18041 / 19514 |
| 6 | 54.43 | 58.39 | 59.36 | +0.97 | 16252 / 18677 / 18985 |
| 7 | 62.14 | 56.92 | 67.46 | **+10.54** | 14778 / 15778 / 22331 |
| **合计** | **459.68** | **473.94** | **501.68** | **+27.74** | |

两点值得注意：

- **非单调**：1ep 反而高于 3ep（rank3 65.31 > 62.97，rank7 62.14 > 56.91）。
  "epoch 越多峰值越高"不成立。
- 这是**二阶效应**：即便按 20ep 预留（最高 67.46），仍远低于 OFF 的 91.30。

### 8.2 根因：MoE all-to-all-v 的接收缓冲（已插桩实证）

在 `mappings.py::_AllToAll.forward` 插桩记录每次 all-to-all 的真实 `output_split_sizes`
（缓冲大小 = `sum(output_split_sizes) × hidden(2048) × 2B`，手法见附录 C）。**step 1（warm-up 步）**：

| rank | 3ep | 20ep | Δ缓冲 | 本卡长度是否变化 |
|---|---|---|---|---|
| 0 | 981 MiB | 1112 MiB | +131 | 19146 → 22199 |
| 1 | 891 | 1021 | +129 | 16895 → 20685 |
| 2 | 1078 | 1234 | **+156** | **不变（19007）** |
| 3 | **820** | **925** | **+105** | **不变（24773）** |
| 4 | 833 | 928 | **+95** | **不变（19908）** |
| 5 | 1002 | 1130 | +128 | 18041 → 19514 |
| 6 | 928 | 1035 | +106 | 18677 → 18985 |
| 7 | 904 | 1014 | +110 | 15778 → 22331 |

**结论**：

1. 缓冲大小由**实际路由分布**决定，不是由 token 总数或行长决定。r2/r3/r4 本卡长度完全不变，
   缓冲照样涨 +156/+105/+95 MiB——**本卡 reserved 不是本卡长度的函数**，跨卡耦合实证存在。
2. **Δ缓冲只有 ~105 MiB，却造成 Δreserved 3.10 GiB** —— 不是线性关系，是 allocator 的
   **阈值效应**：820 MiB 能塞进池里现成的大空闲块，925 MiB 塞不进 ⇒ **必须新开段**，
   新段开了就留在缓存里。这解释了为什么"Σ 行长"与 reserved **不成单调关系**——
   决定结果的是**是否跨过门槛**，不是长度涨了多少。
3. rank3 细看：3ep 最大 `sum` = 209 981（820 MiB），20ep = 236 772（925 MiB）；
   20ep 有 **20/360 次调用**超过 3ep 的历史最大，累计超出 1259 MiB。

> 20ep 的 **925 MiB 与早期栈级别反推的 925 MiB 完全吻合**，归因被独立复现。

### 8.3 跨卡耦合 vs 本卡峰值：设计选择（实测回答）

既然存在跨卡耦合，"只按本卡峰值"还正确吗？`SWIFT_PAD_WARMUP_LENGTH=24773`（8 卡全锁）
正好等价于"全局峰值模式"，可直接对比（GiB）：

| 模式 | r0 | r1 | r2 | r3 | r4 | r5 | r6 | r7 | 合计 |
|---|---|---|---|---|---|---|---|---|---|
| OFF（3ep） | 67.48 | 66.47 | 86.91 | 83.64 | 85.24 | 89.19 | 91.30 | 85.85 | **656.08** |
| **本卡峰值**（3ep） | 56.79 | 54.20 | 63.63 | 62.97 | 62.74 | 58.30 | 58.39 | 56.92 | **473.94** |
| **本卡峰值**（20ep） | 60.16 | 58.15 | 65.99 | 66.07 | 62.96 | 61.53 | 59.36 | 67.46 | **501.68** |
| **全局峰值**（锁 24773） | 64.82 | 64.93 | 79.51 | 65.53 | 80.97 | 71.39 | 74.61 | 75.58 | **577.34** |

同 epoch 档下 **全局峰值比本卡峰值多 75.66 GiB**（与 3ep 比是 +103.40）。涨得最狠的 r2/r4
正是被"为用不到的 batch 买单"的卡。

**结论**：跨卡耦合虽然真实，但**"按本卡峰值"仍是正确设计**，优势明显；直觉上"全局峰值更保险"
是错的。即便全局峰值（577.34）也仍低于 OFF（656.08）——**pad-warmup 的收益主要来自
"把峰值提前并冻结"这件事本身**，长度取哪个只决定收益大小。

两条实践建议：

- **不要为了"更保险"加余量**：§8.2 已证明 reserved 对长度是阈值响应而非线性，把长度调大只会
  让缓冲变大、更容易跨过新开段的门槛。当前"精确重放本卡峰值"就是最优点。
- **改数据集 / epoch 数后要重看峰值**：某张卡的 reserved 可能因**别人**的长度变化而跳一档。

### 8.4 历史：rank3 曾为唯一变高的卡（复测后已消失）

旧文档记录 3ep 下 rank3 是 8 张卡里唯一 ON 高于 OFF 的（`ON step0 − OFF 峰值 = +2.26`、
`ON 峰值 − OFF 峰值 = +4.85`）。**2026-09-24 复测该现象已消失**（→ −20.68 / −20.67），
8/8 卡全部为负。

摆动可分解为两边各走一半：OFF rank3 最终峰值 69.62 → 83.64（+14.02，这次撞到更多"新纪录"
batch），ON rank3 step0 71.88 → 62.96（−8.92），合计 22.94 = 2.26 + 20.68 ✓。

**这印证了旧文档自己的判断**：那个例外从来就不是 pad-warmup 的稳定属性，而是
"OFF 那次跑的 rank3 恰好没怎么被碎片顶高"造成的——**OFF 的最终峰值本身是随机量**
（取决于撞到几个"新纪录" batch），拿它当基准不可靠。

> 评估 pad-warmup 不要拿"某张卡的 ON vs OFF"下结论，单卡基准噪声太大。
> 要看**全卡合计**和**最差余量**。

---

## 9. 架构评审：缺口清单与修复优先级（2026-11-09）

对 v5（`pad_warmup.py` 1034 行 + `trainers/base.py` 接线 + `template/base.py` dummy 追加/剔除）
做的一次全面评审。**结论：主干算法（分片扫描 + 本卡 replay + dummy 段）正确且完整，可以保留；
缺的是分布式一致性兜底、时限保护、可观测性这三类基础设施。**

已交叉核对过真实源码，以下两点确认**无误**，不在缺口内：

* **replay 与真实 sampler 比特级一致。** `MegatronPretrainingRandomSampler.__iter__` 中
  `active_total_samples = total - total % (mbs*dp)`、`epoch = consumed // active`、
  `idx_range_active[dp_rank::dp]`，与 `replay_peak_batch_length` 里 `consumed = epoch * active`
  的推导完全吻合。
* **所有 rank 都会进入扫描。** `setup_training → _prepare_data_iterator → _prepare_dataloader`
  在每个 rank 上都执行（不是 Megatron 原生的"仅 tp_rank0 建 dataloader"），
  默认 world PG 上的集合通信不会因为参与者缺失而死锁；VPP 下重复调用由
  `_pad_warmup_wrapped` 一次性标志拦住，各 rank 行为一致。

### 9.1 卡间时间不一致 → 集合通信超时（最现实的线上故障）

扫描期所有集合通信都在**默认 world PG** 上、用 CUDA tensor，因此受
`--distributed-timeout-minutes`（默认 **10 分钟**）与 NCCL/XCCL watchdog 约束。
超时的暴露面 = 各卡**到达该集合通信的时间差**。三个暴露点：

| # | 暴露点 | 时间差来源 | 量级 |
|---|---|---|---|
| 1 | `_dedup_plan_or_all` → `_agree_dedup_plan` 的 all_gather | 每卡**串行逐行** `hf_ds[i]` 遍历全量 n 行；若 media 列是 HF `Image(decode=True)`/`Video` feature，这一步会**真的解码** | 50 万行 arrow 单行访问 10–50 µs ≈ 10–25 s，且强依赖 page cache/NFS，节点间可差 5×；解码型 feature 则是分钟级 |
| 2 | `_exchange_shards_via_gather` | `todo[r::world]` 只按**条数**交错，成本按**解码耗时**算：视频时长重度倾斜、单节点存储冷、CPU 被抢占、`LOCAL_WORLD_SIZE` 未设导致并发度不齐 | 偏斜 ≈ T·\|r−1\|。T=30min + 20% 不均衡 → **6 min，已贴 10 min 红线**；T=60min 必超时 |
| 3 | `_scan_rows` 的 fork 池**挂死**（不是慢） | `_scan_row` 只 catch 异常、不管挂死；一个损坏/截断的 mp4 让 ffmpeg/decord 卡在 read 上，`pool.map` 永不返回；worker 被 host OOM-killer 杀掉同理 | 无限。全体卡在 watchdog 上，**报错指向集合通信，看不出是某个视频文件** |

顺带的功能后果：白名单化后 `_freeze_media` 对 PIL/ndarray 返回 `None`，
**HF `Image` feature 数据集的去重能力为 0**，退化成全量扫描，拿不到去重的 ~7× 加速。

### 9.2 会"挂死全作业"的问题

| ID | 问题 | 机理 |
|---|---|---|
| **A** | 各 rank 对"是否启用"可能不一致 → `warmup_communicators()` 单边集合通信 | 它对**所有 PG（含 world group）**做 all_reduce，但只在 `length is not None` 时执行。只要一张卡的 `replay_peak_batch_length` 走到任意 `return None`（sampler 构造抛异常、瞬时 IO 错误），其余卡就在 world all_reduce 里等到死。此外 `get_pad_warmup_length` **整体没有 try/except**，任何意外异常会**直接打挂训练**而不是降级关闭 |
| **B** | `n` 跨 rank 不一致时，`_agree_dedup_plan` 的兜底**没闭合** | 协商失败后退回 `list(range(n))` 用的是**本地 n**。若不一致的根因恰是 n 不同（某卡少过滤一条），兜底后 `counts`→`width` 仍不同，`_exchange_shards_via_gather` 依旧形状不匹配 |
| **C** | mrope `position_ids` 维度歧义（**确定性 bug**） | `_pad_warmup_pop_dummy` 先判 `v.shape[0] == n_real+1`，再判 3 维分支。mrope 是 `(C,B,L)`，当 `micro_batch_size=2` 且 C=3（Qwen2-VL）时 `shape=(3,3,L)`，**前一个条件先成立** → 切掉 channel 维 → `(2,3,L)`，rope 算错或 shape 报错 |
| **D** | 一次性 flag 文件跨 run 残留 → 重跑**静默不 warmup** | `_warmup_flag_path` 的 key 只有 `output_dir + local_rank`，无 run 标识、无清理。固定 `output_dir` 的续训/重跑第二次启动即命中旧 flag，整个 run 一次都不 pad，日志只有一行 `Warm-up already done` |

### 9.3 会"静默算错峰值"的问题

| ID | 问题 | 机理与量级 |
|---|---|---|
| **E** | 失败样本用 `max_length` 顶替 → padding_free 下**过度预留，step 0 自己 OOM** | 允许 5% 失败率、`mbs=4` 时，含失败样本的 micro-batch 占比 1−0.95⁴ ≈ **19%**，必然命中。`max_length=32768`、真实均值 2000 时 peak 从 ~8k 跳到 ~38k，pad-warmup 从"防 OOM"变成"造 OOM" |
| **F** | `_REPLAY_MAX_SAMPLES` 兜底只 replay epoch 0 → **低估峰值，原始 bug 回归** | §8.1 的实测证据：60 epoch 下 rank4 峰值 19908 → 24773（**+24%**）。50 万样本 × 101 epoch 即越界 → 静默退化 → 第 30 个 epoch OOM。越是"小数据集长 epoch"（最需要本特性的场景）越容易撞上 |
| **G** | **验证集完全未纳入峰值** | `_prepare_dataloader` 把同一个 `data_collator` 给了 train 和 val，但 `wrap_pad_warmup` 只收到 `train_dataset`。val 走 `MegatronPretrainingSampler`（**顺序不 shuffle**），batch 组合完全不同，一段连续长样本即可超过训练峰值 → 首次 eval 碎片 OOM |
| **H** | 缓存键的三处失效盲区 | ① `_dataset_fingerprint` 的 4096 截断是 `break` **整个 walk**，只指纹了靠前的目录，改后面的文件**缓存不失效** → 用旧长度数组 → 峰值错；② `_MM_LENGTH_ENV_VARS` 漏 `NFRAMES`/`IMAGE_FACTOR`/`MAX_RATIO`，尤其漏 **`VIDEO_READER_BACKEND`**（decord/torchvision/av 抽帧数可不同）；③ 模型侧只用 `model` id + 类名 + swift 版本，本地改 `chat_template.jinja`/`preprocessor_config.json` 不失效 |
| **I** | `_expand_dedup` 的文本 delta 在截断下不成立 | `length = media + text + const` 的线性假设在 `max_length` 截断/`truncation_strategy` 下失效，方向偏大（安全）但会与 E 叠加放大。另 BPE 边界合并有几 token 系统误差 |
| **J** | `_data_parallel_size` 自己算 DP 而非问 mpu | 真实 `_prepare_dataloader` 用 `mpu.get_data_parallel_world_size()`。手算 `world//(tp*pp*cp)` 目前一致，但随 Megatron 版本漂移（`num_distributed_optimizer_instances`、hierarchical DP、EP 布局）可能失真 → replay 的 batch 组合与真实训练完全不同 |
| **K** | `_sample_dummy_text_ids` 用模块级 `random` | 消耗 dataloader worker 的全局随机流，影响同 worker 内其他依赖 `random` 的增强的可复现性 |

### 9.4 功能缺口清单

| # | 缺口 | 影响面 | 建议 |
|---|---|---|---|
| 1 | `context_parallel_size > 1` 直接禁用 | **CP 恰是长序列训练主场景**，也最容易碎片 OOM，等于在最需要的地方缺席 | 支持 `peak / cp_size` 的切分语义，列为明确开发项而非"不支持" |
| 2 | `streaming=True` 禁用 | 大数据集常用 | 支持"前 K 个 batch 观测 + 外推"，或要求显式 `SWIFT_PAD_WARMUP_LENGTH` |
| 3 | RLHF/PRM（`template.mode` 不在 train/transformers）禁用 | DPO/GRPO 无覆盖 | 理由成立（seq_lens 段数偏移），升级为 roadmap 项 |
| 4 | `packing=True` 直接用 `max_length` | `max_length=131072` 而实际 bin 均 20k 时**预留 6 倍，step 0 自身 OOM** | packing 装箱是确定性的，同样可 replay；或取 `min(max_length, 实际最长 bin)` |
| 5 | **无"预留 vs 实际"闭环校验** | 低估时特性静默失效，E/F/H 全部无感 | 前 N 步统计实际 packed 长度 max，超过 warmup 长度即 **WARN 并打印差值**。这是唯一能在线发现峰值算错的手段，成本几乎为零 |
| 6 | `expandable_segments` 可用时未自动退让 | 本特性是其不可用时的替代品，同开只是白占显存 | 检测 `PYTORCH_CUDA_ALLOC_CONF` 含 `expandable_segments:True` 时 info 一行并自动关闭 |
| 7 | 扫描期无进度/心跳日志 | 几十分钟静默，无法区分"在扫"和"挂了" | 每 30s 一行 `scanned x/y, eta z` |
| 8 | 缓存用 JSON | 1000 万样本 ≈ 60MB JSON，`json.load` 数秒且内存放大 10×+ | 换 `np.int32` 的 `.npy`/二进制 + 小 JSON 头 |
| 9 | 无缓存 GC | `.pad_warmup_cache` 随 key 变化无限增长 | 按 mtime 保留最近 N 个 |

### 9.5 修复优先级

| 优先级 | 项 | 理由 |
|---|---|---|
| **P0** | 扫描期集合通信换到**独立 gloo PG**（`dist.new_group(backend='gloo', timeout=timedelta(hours=6))`） | 载荷全是小整数数组（最大 n×8B），无需走 GPU。收益：① 脱离 NCCL/XCCL watchdog，阈值独立可配；② 即使超时失败的是 gloo PG，**不会 abort 已建好的 NCCL 通信器**，可 catch 后降级而非整作业死；③ 不占显存、不与 `warmup_communicators` 的 context 分配打架 |
| **P0** | 每 rank 一个**扫描 deadline**（`min(PG_timeout/3, 显式配置)`），到点未完成的行贡献 `0` | `counts` 由 `todo` 固定、all_gather 等长 padding，**提前收尾不破坏形状**；零值推高 `n_failed` → 触发 `_MAX_SCAN_FAILURE_RATIO` 保守关闭。**用"放弃 pad-warmup"换"不杀作业"** |
| **P0** | **单样本解码超时**：`pool.map` → `imap_unordered` + per-chunk `next(timeout=...)`，超时 `pool.terminate()` 剩余置 0 | 消除 9.1 暴露点 3 的无限挂死；顺带可挂心跳日志 |
| **P0** | 启用标志跨 rank 协商（`all_reduce(MIN)` 的 0/1 flag，任一卡关闭则全体关闭）+ `get_pad_warmup_length` 整体 try/except；`warmup_communicators()` 在两条路径上对称调用 | 修问题 **A**，同时修掉"补丁本身能打挂训练"的路径 |
| **P0** | mrope `position_ids` 的 3 维分支**提到前面**判断（按 key+ndim 分派，不按 shape 猜维度） | 修问题 **C**，`mbs=2` + Qwen2-VL 必中 |
| **P0** | flag key 加入 run 唯一量（rank0 生成时间戳后 broadcast / `TORCHELASTIC_RUN_ID`），并在 `wrap_pad_warmup` 时清本 run 旧 flag | 修问题 **D**，否则重跑/续训静默失效且极难排查 |
| **P1** | 失败样本顶替值改 `max(有效长度)`（**E**）；replay 改"均匀抽 K 个 epoch + numpy 向量化"而非降到 epoch 0（**F**）；`n` 纳入协商且不一致时**全体关闭**而非退回全量（**B**） | 修掉"算错峰值"的三条主路径。向量化后（`randperm` → `np.add.reduceat` 分段求和取 max）full replay 成本可忽略，`_REPLAY_MAX_SAMPLES` 阈值可抬两个数量级 |
| **P1** | 去重计划改**列式访问**（`select_columns` / 整列取），media 列先 `cast_column(col, Image(decode=False))` 或读 arrow 原始 `{'path','bytes'}` | 一改两得：砍掉 9.1 暴露点 1，并**恢复 PIL/HF Image 数据集的去重能力** |
| **P1** | 按 `os.stat(video).st_size` 做 **LPT 贪心分片**替代 `todo[r::world]`；`_scan_worker_init` 里设 `CUDA_VISIBLE_DEVICES=''` | 文件大小与解码耗时相关性足够好，几乎零成本把不均衡度从 3× 压到 1.2×；后者消除 **fork-after-CUDA** landmine（扫描发生在 `setup_model_training()` 之后，主进程 CUDA 已初始化） |
| **P1** | `_scan_num_proc` 增加 host 内存感知（`available_mem / 预估单 worker 峰值` 取 min） | 64 worker 同时解码 4K 视频 + 此刻主机内存已被模型/优化器占高 → OOM-killer 杀 rank → 其他卡挂在集合通信上 |
| **P2** | 指纹改**全量流式 hash**（只累加进 md5，内存 O(1)）+ 补齐 env + 纳入 processor 配置（**H**）；`_expand_dedup` 每值 clamp 到 `max_length`（**I**）；`_data_parallel_size` 优先走 mpu（**J**）；`random.Random(0)` 局部实例（**K**） | 长期正确性 |
| **P2** | "预留 vs 实际"闭环校验（缺口 5）；val 纳入峰值（**G**）；CP 支持；packing replay；缓存二进制化 | 可观测性与适用面 |

### 9.6 一句话总结

主干算法是对的。**但"每一条异常路径都必须是全体一致的降级，而不是某张卡单独 return"
这条分布式铁律目前只做到了一半** —— `_agree_dedup_plan` 说明已经意识到这个问题，
把同样的协商/兜底思路推广到**启用判定、`n`、扫描时限**三处，再加上独立 PG 与 deadline，
这个特性就具备交付给他人在任意集群上使用的健壮度。

### 9.7 v6：P0 修复已落地（2026-11-09）

v5 封存为 `feature4_megatron_pad_warmup-5.patch`；**当前交付为 `feature4_megatron_pad_warmup.patch`（v6）**
（`pad_warmup.py` 1034 → 1245 行；`trainers/base.py` hunk 不变；`template/base.py` 仅
`_pad_warmup_pop_dummy` 一个 hunk 变化）。§9.5 的 6 个 P0 项全部实现：

| P0 项 | v6 实现 | 位置 |
|---|---|---|
| 独立 gloo PG + 长超时 | `_scan_process_group()`：`dist.new_group(backend='gloo', timeout=6h)`，扫描期全部集合通信（flags / broadcast / plan 协商 / 分片交换）改走该 PG 的 **CPU tensor**；gloo 不可用时告警并回退默认 PG | `_SCAN_PG_TIMEOUT_S = 6*3600` |
| 扫描 deadline | `_SCAN_DEADLINE_S = _SCAN_PG_TIMEOUT_S // 3`（2h）；`_scan_rows` 在**结果路径**（每个结果都查）与**空转路径**（60s 轮询）双侧检查，到点 `pool.terminate()`，未完成行记 0 → 触发 `_MAX_SCAN_FAILURE_RATIO` 保守关闭 | `_scan_rows(deadline_s=...)` |
| 单样本解码超时 | 停滞超时 `_SCAN_ROW_TIMEOUT_S = 600s`：任何一个结果间隔超过它就 `pool.terminate()`（损坏视频卡死 ffmpeg/decord、worker 被 OOM-killer 杀掉，表现相同）；另有每 60s 心跳日志 | `_scan_rows` |
| 启用标志跨 rank 协商 | `_agree_enable()`：all_reduce(MIN)，任一卡否决则全体关闭并打分歧告警；**两处调用点**——扫描前（防 env 只设在部分节点 → 少数卡 stranded 在扫描 PG rendezvous）与长度解析后（防单卡失败 → 其余卡死在 `warmup_communicators`）。`get_pad_warmup_length` 整体 try/except 降级为 None | `wrap_pad_warmup` |
| mrope 维度歧义 | `_pad_warmup_pop_dummy` 改为**先按 key+ndim 判 `(C,B,L)` 的 3 维分支**，再判 `shape[0] == n_real+1`；`mbs == C-1`（如 mbs=2 + Qwen2-VL 的 C=3）时不再错切 channel 维 | `template/base.py` hunk |
| flag 跨 run 残留 | `wrap_pad_warmup` 包装时**先删除**本 rank 的 flag 文件，再交给 worker 竞争 claim —— flag 的生命周期严格等于本 run | `_warmup_flag_path` / `wrap_pad_warmup` |

实现中踩到并绕开的一个环境坑，记录备查：**本机 TencentOS Python 3.11.6 的
`Pool.imap_unordered` 在 `chunksize > 1` 时返回裸 generator**（无 `.next(timeout=)`），
上游 3.13+ 也有此行为。因此 `_scan_rows` 固定 `chunksize=1`（换来 IMapIterator 的
timeout 能力），逐行 IPC 开销相对解码可忽略。

**demo 验证**（`demo_sharded_scan.py`，world=4 / 1600 视频 × 4 QA + 200 文本，10/10 通过；
在原 7 场景基础上新增三个 v6 场景）：

| 场景 | 结果 | 说明 |
|---|---|---|
| F_scan_deadline | OK | 强制 4s deadline + 200ms/行慢解码：4 卡全部在 **4.0s 处精确停止**（各完成 305/450 行），all_gather 形状由固定 plan 决定不受影响，全体收敛到**同一个**数组（1720 失败 = 26% > 5% → 生产路径将保守关闭 pad-warmup），集合通信等待 ≤ 63ms |
| G_enable_disagree | OK | rank0 单独投"启用"→ 全体一致**关闭**，并输出分歧告警 |
| H_pg_fallback | OK | 注入"rank1 创建 gloo 组失败"（rendezvous 后加入、本地初始化失败）：**全体**对称回退默认 PG，无人独自等在 gloo 组上 |
| A/B/B2/D/E/C/C2 | OK | 与 v5 一致：分片 2.5×（本机 8 核上限）、缓存全命中零通信、失败注入/计划退化均收敛不挂死 |

**v6 复查修复**（对 v6 初版的代码复查又抓到 4 个问题，均已修掉并含在上述 1245 行内）：

1. **扫描 PG 的回退不对称**（v6 初版新引入，`demo 场景 H` 覆盖）：`new_group` 只在部分 rank
   失败时，失败者回退默认 PG、成功者用 gloo PG → 两边在不同的 PG 上互等，挂死 6h。
   修复：创建后用默认组做一次 MIN 协商，任一失败则**全体**回退默认组。
2. **`SWIFT_PAD_WARMUP_LENGTH` 只设在部分节点会挂死**（v5 遗留）：该 override 是"提前 return"
   分支，不设它的 rank 会进扫描的 rendezvous 空等。修复：override 值折进 `_agree_enable` 的
   tag（all_gather 比对，不一致全体关闭）。
3. **flag key 用 `LOCAL_RANK`，共享缓存目录下跨节点冲突**（v5 遗留）：§3.1 自己建议
   `SWIFT_PAD_WARMUP_CACHE_DIR` 指向共享 FS 复用扫描结果，但那样所有节点的 `local_rank=0`
   共用同一个 flag 文件 → **只有一个节点会 warmup**。修复：改用全局唯一的 `RANK`。
4. **心跳只在空转时打**：稳定产出的长扫描全程静默。修复：结果路径同样每 60s 打进度。

**v6 第二批：正确性修复（Top5 之 1/2/3，2026-11-09）**：

1. **replay 向量化 + epoch 均匀采样**（问题 F）。sampler 链路（`randperm(seed=epoch)` →
   `[dp_rank::dp]` 抽条 → 按 `group_size` 分段求和/取 max）全程 tensor 化
   （`_replay_peak_vectorized`），36M 次遍历 3.08s（纯 Python 循环需分钟级）；
   上限 `_REPLAY_MAX_SAMPLES` 50M → 500M，越界时**均匀抽 ≤16 个 epoch（含末位）**，
   替代 v5 的"只看 epoch 0"（实测峰值随 epoch 涨 ~24%，旧兜底系统性低估）。
   `group_by_length` 布局仍走真实 sampler（`_replay_peak_via_sampler`，罕见布局，
   正确性优先）。**demo3 矩阵验证 288 种组合（n×mbs×dp×shuffle×data_sharding×agg）
   与真实 sampler 逐 batch 完全一致**——并因此抓到并修掉一个真 bug：v6 初版向量化
   在多 rank 间共享一个 generator，而真实 sampler 每 rank 独立 seed，rank>0 的
   permutation 会与实际训练不同（mbs=1 时碰巧不暴露）。
2. **失败样本顶替值 `max_length` → `max(有效长度)`**（问题 E）。有界保守：失败样本按
   最长有效样本计入，peak 不再被顶到 ~`group_size × max_length`（demo3：5781 vs 旧法
   上限 131072），消除"step 0 warmup batch 自己 OOM"的路径。
3. **`n` 纳入协商，兜底闭合**（问题 B）。flags 会合点与 `_agree_dedup_plan` 都带上
   `n`：各卡数据集行数不一致时**全体对称关闭**（返回 None），不再退回本地 `range(n)`
   造成形状不匹配。`demo 场景 I`：rank3 少一行 → 全体 8ms 内一致关闭、零通信挂起。

**demo 状态**：`demo_sharded_scan.py` 11/11（新增 I），`demo_replay_vectorized.py` 5/5。

**v6 第三批：tail-first 重设计——"扫描不完"从死局变成 anytime 算法（2026-11-10）**：

设计语义的最终修正：warmup 长度是**实际峰值 `L*` 的最紧上界**——欠预留是延迟死
（后续 batch 申请不到连续块 → OOM），过松是立即死（step 0 warmup batch 自爆），
两者都是死，目标函数是 `w − L*` 最小且非负。由此推出：峰值估计需要的是**覆盖上尾**，
而不是覆盖全集。五项改动：

1. **tail-first 扫描顺序**（`_order_tail_first`）：按媒体文件字节代理降序（未知大小最前、
   纯文本最后），代理表 `sizes` 随计划一起进 `_agree_dedup_plan` 的 crc 协商。任意
   截断点未扫的都是短样本 → 扫描成为 anytime 算法。解决了三个索引序/随机序的失效场景：
   数据集长度成簇、解码耗时偏置（长视频完成最晚）、稀疏长尾。
2. **三档峰值模式**（`get_pad_warmup_length`）：`exact`（全扫）→ `substituted`
   （≤5% 缺失，`max(有效)` 顶替，膨胀可忽略）→ **`tail-estimate`**（>5% 缺失：
   `_estimate_peak_from_subset` 在已扫子集上 Monte Carlo 模拟批次（种子固定 → 全体
   rank 结果一致），**不顶替**——顶替会让峰值随缺失率线性膨胀 → step 0 自爆）→
   已扫行数 < `max(1000, mbs×10)` 才关闭。**v6 早版"超 5% 就关闭"的硬降级被移除**，
   大 n 数据集扫不完也能拿到有界保护。
3. **闭环校验**（`PadWarmupCollator._check_overshoot`）：每个 batch 量一次实际
   packed 长度（每行一次 `len()`，成本可忽略），超过 warmup 长度即 WARN 一次——
   代理排序骗人、replay 漂移、缓存过期等一切"静默低估"路径的唯一在线兜底。
4. **模式与覆盖率日志**：`Peak mode: exact/substituted/tail-estimate (coverage X%)`，
   每次启动可审计。
5. **docstring 语义更新**：明确"最紧上界，宁松勿欠、松不过头"的双侧约束。

**demo 验证**：`demo_tail_first.py`（新增）4/4——排序正确性（未知大小最前/降序/文本最后）、
三档行为（exact == replay、substituted == 顶替后 replay、tail-estimate 与全数据估计偏差
<10% 且不膨胀、子集过小才关闭）、估计种子确定性、闭环 WARN 恰好一次；
`demo_sharded_scan.py` **12/12**——新增场景 K（复用 F 的 26% 缺失缓存，4 卡得到**同一个**
tail-estimate 长度，日志 `Peak mode: tail-estimate (coverage 74%)`，~0.13s 零通信）。
排障记录：K 初版误用 `DemoTemplate` 导致缓存键的 `template_cls` 不匹配而静默重扫，
印证了缓存键设计的有效性。

**v6 第四批：列式计划构建 + `decode=False`——砍掉 9.1 暴露点 1，恢复 HF Image/Video
数据集的去重能力（2026-11-10）**：

P1 项「去重计划改列式访问」落地。`_scan_dedup_plan` 此前在每卡上**串行逐行** `hf_ds[i]`
遍历全量 n 行（50 万行 arrow 单行访问 10–50µs ≈ 10–25s，节点间可差 5×；若媒体列是
`Image(decode=True)` 特征还会**真解码**整个数据集），且 PIL 对象被 `_freeze_media`
合理拒绝 → 此类数据集去重能力为 0。三项改动：

1. **列式计划构建**（`_media_columns`）：每个媒体列一次整列读取替代 n 次行随机访问。
   plain 字符串列走 arrow `to_pylist`（向量化；实测 5 万行 **0.28s vs 行式 4.5s，15×**）。
   无 `column_names` 的数据集对象（如 list）原样走行式回退，行为与之前逐位一致。
2. **解码型媒体特征的零解码读取**：`Image/Video(decode=True)` 列优先从 arrow 结构体直接
   抽 `path` 子列 + `bytes` 子列的**长度**（`_arrow_media_column`：不解码、不物化内嵌字节，
   `select()/filter()` 视图的 `_indices` 映射被显式应用，计划始终按逻辑行序）；arrow 不可达
   时退到 `cast_column(col, Image(decode=False))` 的纯元数据视图。这同时**恢复了去重**：
   路径即身份（实测 200 行 = 50 唯一图 × 4 QA → 50 代表，零 PIL 解码）。内嵌 bytes（无
   path）的行仍**不去重**（字节长度只当 tail-first 的大小代理，不冒充身份），且 bytes 载荷
   被 `_BYTES_MARKER` 摘除、不进计划内存。空媒体单元（null struct）归一为 None，与 HF 行
   访问的格式化结果一致。
3. **`_expand_dedup` 文本列式化**：文本增量修正整列取 `messages`，不再对重复行逐个
   `hf_ds[i]`（那对 decode=True 特征会把每行媒体再解码一遍）。

跨路径一致性：列式与行式对同一数据产出**相同**的计划（键形、列序、大小语义逐位对齐，
demo 逐场景断言），仅有的语义差异都是有意的升级（decode=True 特征恢复去重；内嵌 bytes 行
的大小代理 0 → 真实字节数）。若个别 rank 因环境差异走了不同路径，`_agree_dedup_plan` 的
crc 协商会检出并让**全体**对称退回全量扫描——慢但形状一致，不会挂死。

**demo 验证**：`demo_columnar_plan.py`（新增，训练镜像内跑真实 datasets 4.8.4 / pyarrow 24，
`PW_SOURCE=patch` 直接 exercising 补丁内容）**28/28**，场景 A–H：字符串列计划与行式参考
逐位一致（零行访问零解码）；`Image(decode=True)` 零解码去重恢复；内嵌 bytes 不去重、
大小=字节数、tail-first 排前；`Sequence(Image)` 按路径列表去重；`select()` 视图逻辑序正确；
纯文本计划 O(1)；端到端 `_scan_lengths` 每个唯一图像恰好解码一次、长度与参考一致、
`_expand_dedup` 零行访问且文本增量正确；5 万行性能对比。排障记录：datasets 4.x 的
`ds[col]` 返回惰性 `Column`，迭代是逐行格式化（50k 行 ~1.2s），plain 列必须走
`data.column(col).to_pylist()` 才拿得到向量化读取——否则列式化的收益被吃掉大半。

**demo 状态**：`demo_dedup_key.py` 18/18（修复 `_scan_dedup_plan` 三元组解包的过时断言）、
`demo_sharded_scan.py` 12/12、`demo_replay_vectorized.py` 5/5、`demo_tail_first.py` 4/4、
`demo_columnar_plan.py` 28/28。patch 已重建（83514→94817 字节，仅 `pad_warmup.py` hunk
变化），往返提取与 `feature4_demo/pad_warmup.py` 逐字节一致，对 pristine ms-swift 4.3.1
（pip wheel）`patch -p1` 干跑/实应用均通过、三文件编译通过。

**第四批复查：冗余合并（2026-11-10）**。全面核对后结论：无死代码（每个函数都有调用点，
4 个环境变量全部在用），合并了四组结构性重复，1878 → 1857 行——`_row_media_size` 收编为
`_strings_size` 的薄封装（顺带消掉一个无效的 `seen` 标志）；`_row_media_key` /
`_columnar_media_key` 共用 `_media_key(value_of)`；`_arrow_media_column` /
`_arrow_plain_column` 共用 `_arrow_column`（table/`_indices` 处理只写一遍）；
`_arrow_struct_media` 与 `_normalize_media_value` 的 path/bytes 决策树共用 `_media_cell`。
同时删除与 `pad_warmup.py` 逐字节相同的 `pad_warmup_v6.py`（88KB 文件级重复）。
合并后全量回归不变（18/18、12/12、5/5、4/4、28/28），patch 往返一致。

**tail-first 的大规模实测**（`demo_tail_first_scale.py`，新增 13/13）。此前只有小组件级
验证，这次构造了专门击穿索引序的场景量化收益：**50 万行**、lognormal 主体 + 0.4% 重尾
（8–40×）、大小代理带噪（log 域 ρ≈0.85，非完美先知）、64 worker、**deadline 卡在墙钟
30%**（扫描注定不完），两种布局（`clustered`：长度成簇、长样本在索引末尾——索引序的
最坏情况；`interleaved`：洗牌对照），三种顺序走同一套生产函数（`_order_tail_first` +
`_estimate_peak_from_subset`）：

| 布局 | 顺序 | 上尾覆盖（top-2000） | 峰值估计/真值 | 延迟死亡率（后续 batch 超预留比例） |
|---|---|---|---|---|
| clustered | index | **0.0%** | **4.3%** | **84.4%**（必然延期 OOM） |
| clustered | random | 30.2% | 77.0% | 0.01% |
| clustered | **tail-first** | **100%** | **100.0%** | **0.00%** |
| interleaved | index / random | ~30% | 68–77% | ≤0.01% |
| interleaved | **tail-first** | **100%** | **100.0%** | **0.00%** |

Part 2 用**真实 fork 池路径**（`_scan_rows` + 真实文件 + 真实 sleep + 真实 deadline，
4000 行）复现了同一结论：索引序完成 2422 行但 top-100 覆盖 0%（估计为真值 3.3%）；
tail-first 只完成 383 行——**deadline 预算优先喂给昂贵的尾部**正是设计意图——top-100
覆盖 100%，估计为真值的 107%。

**尾部富集问题与频率修正**（2026-11-11）：demo6 初版暴露 tail-first 截断后的子集是
**尾部富集**的——50 万行截到 4.9 万行时，尾行浓度从 0.4% 变成 4.1%（10.3×），均匀
Monte Carlo 会把"多条尾行撞进同一 batch"的稀有事件概率放大 ~100 倍，峰值被推高到真值
的 149.7%。**过松与欠预留同罪**（step-0 warmup batch 自爆），这个幅度不可接受。修复：
`_estimate_peak_from_subset` 改为**按全量分布抽样**——每个 batch 位置以 `|子集|/n_total`
的概率抽真实已扫行，否则用"已扫最短带"（截断处的上邻，轻微高估=安全方向）建模未扫行；
固定种子不变，跨 rank 一致性不受影响。修正后同一极端场景估计 = **真值的 100.0%**
（真实 fork 池路径 107%），上限断言收紧到 <1.10×。注意顶替法（`max(有效)` 补齐）在 90%
缺失时会估到 `group_size × max` ≈ 真值的 750%，仍是被否决的对照。

剩余待办按 §9.5 的 P1/P2 表继续（LPT 分片、fork 池屏蔽 CUDA、`_scan_num_proc` 内存感知、
缓存指纹流式 hash、val 集纳入峰值、CP 支持等）。

### 9.8 v7：settle-then-carve——把 carve 挪到常驻分配落地之后（2026-11-11）

**动机**：v6 在 step1 carve 峰值块，但部分**常驻**分配此时还不存在，它们在首个 step 前后
陆续落地，落点由当时的 free block 布局决定——落进切好的峰值块就把它永久钉碎
（"预热出来的大块被小块占用，无法正常整块复用"）。

**源码事实**（镜像 `qwen-p800-megatron:20261009-p1`，megatron-core 0.18.0 + ms-swift 4.3.1）：

- Adam 的 `exp_avg/exp_avg_sq` 在 fresh run 下由内部优化器在**首个 `step()`** 才分配；
  megatron-core 自带的 `init_state_fn`（`zeros_like`，带 `len(state[p])==0` 守卫）
  只在 ckpt 加载路径被调用（`state_dict/sharded_state_dict` 的 `is_loading=True` 分支）。
  量级 2×fp32 参数 ÷ dp_size，是首个 step 常驻增量的大头。
- 通信子 context（~2.3 GiB）已由 v6 的 `warmup_communicators()` 提前落地。
- `swift/utils/torch_utils.py::gc_collect()`（`gc.collect() + empty_cache()`）是本仓库
  既有模式（`save_checkpoint` 每次保存前都调）。
- 首个 step 的其余增量（cudnn/TE workspace、autograd 结构、MoE alltoall 缓冲）是
  shape 相关的**瞬时**分配，不需要 settle。

**做法**（增量极小，`base.py` 一行 + `pad_warmup.py` 一个函数）：

1. `_settle_allocator_state(optimizer)`：遍历（可能 Chained 的）optimizer，对每个子
   optimizer 直接调 `init_state_fn(inner, config)` 把状态分配出来——与首个 `step()`
   做的事完全相同，但**不跑 step**：不消耗 RNG（dropout）、不让 Adam 动量吸收垃圾梯度、
   不动权重。`len(state)==0` 守卫使 resume-from-ckpt 场景幂等（已加载的状态不被触碰）。
   随后 `gc.collect() + empty_cache()` 释放所有空闲段，打印 allocator 摘要日志。
2. `wrap_pad_warmup` 在 `warmup_communicators()` 之后调用它（`base.py` 调用点加传
   `optimizer=self.optimizer`）。时序变为：扫描/replay → 通信子预热 → settle 清场 →
   step1 由 collator pad → **峰值轨迹在干净堆上围着稳定的常驻布局雕刻**。
3. 两个动作都是**纯本地操作，零新增集合通信**；全程 try/except，单卡失败只降级为
   v6 行为，绝不打挂训练。

**单文件替换的兼容性**：在已打过旧版 patch 的环境上，**只替换 `pad_warmup.py` 也能安全
运行**（`optimizer` 参数默认 `None` 向后兼容）——得到 v6 全部 + 清场 + §9.9 的
dispatcher 释放，但 optimizer 状态落地不生效，且会打一条 WARNING 提示。要完整 v7 需
`base.py` 的一行（调用点传 `optimizer=self.optimizer`），即打整 patch 或直接改该行。

**被否决的备选**（依据见 `feature4_megatron_pad_warmup-7.md` §4）：真实/合成 settle step
（RNG 偏移、Adam 动量被垃圾梯度污染、grad_norm skip 导致状态不分配、first-step 前钩子
状态机交互——处理链长且脆）；单块 `torch.empty(peak_bytes)`（不复制峰值轨迹，小块照样
把大段切碎）。

**demo 验证**：`demo_settle.py`（新增，torch CPU）5/5——fresh optimizer 状态落地且全零、
RNG 不消耗；ChainedOptimizer（dense+expert）全部子优化器落地；已填充状态幂等不动
（resume 场景）；单个子优化器失败只丢自己的 settle（告警 + 其余照常）；`None`/非 megatron
optimizer 降级为仅清场。全量回归（镜像内）：18/18、12/12、5/5、4/4、28/28、13/13、5/5。
patch 已重建（94817 → 99257 字节，`pad_warmup.py` 与 `trainers/base.py` 两个 hunk 变化），
往返提取逐字节一致，pristine 4.3.1 干跑/实应用/三文件编译通过。

**镜像内端到端验证**（`qwen-p800-megatron:20261009-p1`，2026-11-11）：三文件覆盖安装后编译
通过；`wrap_pad_warmup` 经真实 megatron-core 类（ChainedOptimizer + megatron 形态的子优化器
布局）走通全链路——`SWIFT_PAD_WARMUP=1`、200 行带 lengths 列的数据集、真实扫描/replay/
settle/collator 路径，首个 batch `pad_to_length=1953` 与 `replay_peak_batch_length` 独立
重算值逐位一致，settle 落地 3 个参数状态；幂等性（预填状态不被触碰）与 RNG 零消耗均验证。
注：`FP32Optimizer.__init__` 内部分配 `device='cuda'` 张量，无卡宿主机上无法实例化真身，
镜像验证用同属性布局的 namespace 代替（遍历逻辑只依赖 `.optimizer/.init_state_fn/.config`
属性名，demo_settle 亦覆盖）。

**待环境验证**（-7 设计文档 §6，需 P800 实机）：探针确认 step1 常驻增量 ≈ 2×fp32参数/dp +
2.3 GiB（若有大额漏网项，settle 升级为真实 step 的后备方案）；v6 vs v7 A/B（3ep/20ep，
跟踪每 N 步最大空闲块尺寸，观察预热块是否仍被切碎）。

### 9.9 v7 续：MoE dispatcher 路由状态释放（backport 上游 #8297/#8231，2026-11-11）

**问题**：`MoEAlltoAllTokenDispatcher.dispatch_preprocess` 把 `self.probs`（router 输出，
`[num_tokens, num_experts]`）和 `self.routing_map` 存到模块属性且**从不清理**。属性引用
把张量连同 **router 的 autograd 子图**一起钉住，直到下一步同层 forward 覆盖——每层
20-40MB（40960 tokens × 256 experts × fp32/bf16）、全套 GiB 级跨步常驻，且每步
"先分配新的、后释放旧的"导致位置漂移，在 carve 好的峰值块上持续打洞。settle（§9.8）
管不了它：gc/empty_cache 释放不了有引用的块。

**上游已修复**：NVIDIA/Megatron-LM #8297（2026-10-29 合并，`combine_postprocess` 末尾
`self.probs = None`）和 #8231（2026-10-16 合并，释放 routing tensors），均为
community-request 驱动；本镜像 megatron-core 0.18.0（2026-10-09 构建）早于修复两周。
上游注释原文：*Retaining them here pins the recomputed MoE forward graph until the next
iteration.*——full recompute 下被钉住的是整条 router 计算图，不止张量本身。

**本 patch 的 backport（全部在 pad_warmup.py 内，不新增 patch 文件）**：
`_patch_moe_dispatcher_release()` 对 `MoEAlltoAllTokenDispatcher.combine_postprocess`
做方法包裹——调完原方法后置 `self.probs = None; self.routing_map = None`。安全性（对照
本版本 0.18.0 源码确认）：probs 的最后消费点是 `combine_preprocess`（只用 `.dtype`），
routing_map 的最后消费点是 `combine_postprocess` 内部（unpermute 的 `routing_map=` 参数；
`moe_layer.py:544` 的读取在 combine 之前）；recompute 重放 forward 会重新赋值，天然幂等。
包裹在 `wrap_pad_warmup` 里随 `SWIFT_PAD_WARMUP` 启用，带 `_pad_warmup_release` 幂等标记，
try/except 降级，不新增任何集合通信。

**验证**：demo_settle 第 6 组——fake dispatcher 上 weakref 验证 probs 张量 + autograd
子图完全释放、combine 返回结果不变、重复 patch 幂等、无 megatron 环境优雅降级；镜像内对
**真实** `MoEAlltoAllTokenDispatcher` 类包裹验证通过。显存收益待 P800 实机 A/B（预期：
跨步常驻的 probs/routing_map 集合消失，约 30-50MB × MoE 层数，且消除每步的块内打洞）。

---

# 附录

> 排查手法。代码里已不保留对应钩子，需要时按此手法临时加回。

## 附录 A：用 torch memory snapshot 采集显存现场

日志里的 `memory(GiB)` 只是聚合后的值，看不出段分布和碎片。在 P800 + `torch_xmlir` 上实测可用：

```
_record_memory_history / _snapshot / _dump_snapshot : 均可用
snapshot keys   : ['allocator_settings', 'device_traces', 'external_annotations', 'segments']
segments[i]     : device / total_size / allocated_size / active_size / requested_size / stream / blocks[]
allocator_settings: {'expandable_segments': False, ...}   # XPU 不支持 expandable_segments
device_traces   : XPU 上条数很少，timeline 视图基本不可用
```

采集点放在 `train_step()` 里 `self.optimizer.step()` **之前**；不要全程开（会打满 `max_entries`
并严重拖慢），dump 前先 `enabled=None` 停记录。

**务必按 rank 分别记录**——崩的往往不是 rank0，只看 rank0 会得出完全错误的结论。
pkl 拖到 https://docs.pytorch.org/memory_viz 查看，主要用 `segments` 做离线统计：

```python
segs = snap['segments']
reserved  = sum(s['total_size'] for s in segs)
free_blks = sorted((b['size'] for s in segs for b in s['blocks']
                    if b['state'] != 'active_allocated'), reverse=True)
```

## 附录 B：`_record_memory_history` 的正确调用（曾踩坑）

**必须传字符串**才走 `_record_memory_history_impl`；传 bool 会走 legacy 分支、拿不到任何栈，
据此曾误判"该栈不支持 memory history"。

```python
torch.cuda.memory._record_memory_history(enabled='all', context='all',
                                         stacks='python', max_entries=8000000)
# ... 跑一个 step ...
snap = torch.cuda.memory._snapshot()          # device_traces 里带 frames
torch.cuda.memory._record_memory_history(enabled=None)
```

注意：`context` 只接受 `'state'` / `'all'` / `None`；`max_entries` 会截断，**只在需要的那一步
前后开启**；`stacks='python'` 足够。

## 附录 C：怎么自己复现这些数据（一次性插桩手法）

> 以下是一次性排查动作，**钩子用完即删，不属于功能代码**。需要复现本文数据或排查新问题时按此加回。

**逐卡 reserved**：在 `swift/megatron/trainers/base.py` 的 `train()` 循环里加 env 门控的写文件
钩子，每步记录 `torch.cuda.memory_reserved()` 与 `torch.cuda.max_memory_reserved()`。两点必须注意：

- **只有 rank0 的 stdout 进日志**，所以必须每卡各写自己的文件才能横向比较；
- **用 append 模式**。`open(path,'w')` 会先截断文件，写失败就把原文清空。

**各卡峰值长度**（不开训练也能算）：直接调
`swift.megatron.trainers.pad_warmup.replay_peak_batch_length()`，传入缓存下来的逐样本长度数组，
对每个 `dp_rank` 各算一次。

**MoE 缓冲**：在 `mappings.py::_AllToAll.forward` 的 all2all-v 分支记录
`sum(output_split_sizes)`——§8.2 的数据就是这么来的。

跑 A/B 对照时的操作坑：

- 端口 9901 有 ~90 s 残留，连跑会 `EADDRINUSE`，换端口或等待；
- 1 epoch 只有 9 步，按 20 步等待会挂住；跑满会在结束时存 ~70 GB ckpt，中途杀掉可避免；
- 日志里的 `memory(GiB)` 是 `max_memory_reserved` 且经
  `reduce_max_stat_across_model_parallel_group` 打印，不能用来横向比较各卡。

## 附录 D：`mmco: unref short failure` 刷屏（日志噪音，未纳入 patch）

视频训练时日志里会有大量 `mmco: unref short failure`（伴生 `Missing reference picture`）。
它是 **libavcodec 解 H.264** 时遇到的轻微码流问题：MMCO 指令要 unref 一个不在短时参考列表里的帧，
解码器**忽略该指令继续解码**。

**与 §5.4 的扫描性能优化无关，本 patch 没有处理它**（曾一度把静音逻辑混进扫描优化里，已拆开）。

实测结论：

| 项 | 结果 |
|---|---|
| 正确性 | 两次扫描抽出的帧数完全一致（6424），token 长度不变 |
| 性能 | **<2%**：有日志 55.18 s vs `stderr → /dev/null` 54.43 s，落在重复测量自身 1.4 s 的波动内 |
| 唯一代价 | **日志噪音**：占训练日志 **45% / 61% / 88%** 的字节（bs2 那次 37569 行 ≈ 1.92 MB / 2.18 MB） |

若要处理，两种做法（都不在本 patch 内）：

1. **只过滤日志流**（零风险，改 `run.sh`）——用 `sed` 而不是 `grep -v`，避免 `set -o pipefail`
   下 grep 的退出码干扰：

   ```bash
   bash "${TRAIN_SH}" 2>&1 \
     | sed '/mmco: unref short failure/d; /Missing reference picture/d' \
     | tee "${TRAIN_LOG_FILE}"
   ```

2. **进程内静音**（实测有效）——必须**先 `import decord` 再 `av_log_set_level(-8)`**，因为 decord
   加载时会把级别重置回去（先设置再 import → 仍有 10 行；先 import 再设置 → 0 行）；而且 decord
   用的是自带的 `decord.libs/libavutil-*.so`，不是系统那份：

   ```python
   import ctypes, glob
   for p in glob.glob('<site-packages>/decord.libs/libavutil*.so*'):
       ctypes.CDLL(p).av_log_set_level(-8)   # AV_LOG_QUIET
   ```

   调用点要在解码之前（例如 dataloader worker 的初始化处），且会一并静音 decord/ffmpeg 的其他日志
   ——真正的解码失败仍会以异常抛出，不影响 §5.3 的失败样本处理。
