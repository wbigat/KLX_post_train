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
> **阅读顺序**：第 1~3 章是"结论 + 用法"；第 4~7 章是"技术细节"；第 8 章是当前已知问题。
> 排查手法放附录。

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
1. 扫描数据集，得到每条样本的 token 长度（磁盘缓存，第二次起直接命中，见 §5）
2. 重放 MegatronPretrainingRandomSampler，取「本 dp_rank × 本次训练所有 epoch」的最大
   micro-batch 长度        ← 精确值，不是估算，见 §4
3. 第一个 batch 用一条 dummy 样本补到该长度（padding_free 下是 packed 行里的独立一段）
4. 预热集合通信后端（XCCL/BKCL）的通信子   ← 否则 step1 会 OOM，见 §7
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

---

## 3. 配置、验证与生效确认

### 3.1 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `SWIFT_PAD_WARMUP` | **`0`（关闭）** | 设为 `1` 开启 |
| `SWIFT_PAD_WARMUP_LENGTH` | 无 | 显式指定目标长度，跳过扫描与重放 |
| `SWIFT_PAD_WARMUP_CACHE` | `1` | `0` 关闭长度扫描的磁盘缓存 |
| `SWIFT_PAD_WARMUP_CACHE_DIR` | `<cwd>/.pad_warmup_cache` | 换缓存目录（含 one-shot flag） |

> 部署脚本 `env.sh` 里默认开启（`SWIFT_PAD_WARMUP=${SWIFT_PAD_WARMUP:-1}`）。
> 注意 `train_megatron.sh` 会**重新 `source env.sh`**，所以做 OFF 基线必须确认外层设的 `0`
> 没有被覆盖（见 §3.4）。

**自动关闭**（无需配置）：`context_parallel_size > 1`；`template.mode` 非 `transformers/train`
（RLHF/PRM 按 `packed_seq_params.seq_lens` 索引分段 logits，多一段会错位）；长度不可得；
`streaming=True`；扫描失败率 > 5%。

### 3.2 交付物与 patch 验证

| 文件 | 内容 |
|---|---|
| `feature4_megatron_pad_warmup.patch` | 功能补丁（944 行，含 §5.4 的扫描加速） |
| `feature4_megatron_pad_warmup.md` | 本文档 |
| `PAD_WARMUP_HANDOVER.md` | 交接/复测记录（含更完整的原始数据） |

patch 覆盖 `swift/megatron/trainers/base.py`（2 hunk）、`swift/megatron/trainers/pad_warmup.py`
（新增，786 行）、`swift/template/base.py`（3 hunk）。

扫描加速（§5.4）**全部内聚在 `pad_warmup.py` 内部**：不额外改动 `base.py` / `template/base.py`，
也**没有新增任何环境变量**（去重和并行度都是默认行为，无开关）。

> 日志噪音（`mmco: unref short failure`）**不在本 patch 范围内**——它是纯日志观感问题，
> 与扫描性能无关，单独记在附录 D。

> 环境中现装的 `pad_warmup.py` 另有"用文件锁做跨 rank rendezvous"的改动（不在本 patch 内），
> 所以上面那条 `diff -q` 对 `pad_warmup.py` 会报 DIFF —— 这是预期的。

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
   多机训练时若各节点 cwd 不共享，每个节点会各扫一次。

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
| **v4（当前）** | v3 + 扫描加速（按媒体载荷去重 + 单线程多进程，见 §5.4） | 扫描 496.4s → 46.3s，长度误差 ≤2 token |

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
> 但这是"干等时间变短"带来的缓解，不是机制上的根治——若数据集大到扫描仍需十几分钟，
> 依然可能撞上 watchdog；根治要靠把跨 rank rendezvous 从集合通信换成文件锁（环境里有这版，
> 本 patch 未含）。

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
