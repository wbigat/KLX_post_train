# Pad-Warmup（feature4）：用首个 batch 预留峰值显存，消除变长 batch 的碎片 OOM

> **基线说明**：本环境的 `ms-swift` 是 4.3.1，但 pad-warmup 是 **feature4** 版本
> （`swift/megatron/trainers/pad_warmup.py` 为 feature4 新增文件，不在 wheel 的 RECORD 里；
> `swift/megatron/trainers/base.py`、`swift/template/base.py` 也已由 feature4 改过）。
> 本文描述的是 **feature4 当前实现**，不是早期的 feature3（feature3 只支持非 padding_free、
> 长度取 `min(cutoff, max_len(all_samples))`、padding_free 下自动关闭）。
> 下文所有改动都是在 feature4 之上做的。
>
> **阅读顺序**：第 1~3 章是"结论 + 用法"（要做什么、怎么配、怎么验证生效）；
> 第 4~6 章是"技术细节"（长度怎么选、怎么保证只做一次、为什么要预热通信子）；
> 第 7 章是"已知遗留问题"。排查手法与原始数据放在附录。

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

训练开始前，预先算出**每张卡本次训练会遇到的最大 micro-batch 长度**，然后把该卡的
**第一个 batch 补到这个长度**，让 allocator 在 step0 就一次性开出峰值形状的显存块；
此后所有 batch 都 ≤ 它，直接复用，池子不再增长。

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

feature4 相对 feature3 的关键扩展：用一条 **dummy 样本**把 packed 行补到目标长度，
`packing_row` 给它 `position_ids = arange(T)`，使其成为 `cu_seqlens` 里**独立的一段**
（分段边界是 `position_ids == 0`）。dummy 恒为 `labels=-100` / `loss_scale=0`，
因果隔离且零 loss/梯度贡献。dummy 的 token 从真实 batch 采样（不是单一 pad id），
避免把 MoE router 的负载均衡带偏。

### 2.4 效果（3 epoch，8 卡，本数据集；reserved 口径，GiB）

| rank | 本卡峰值 | OFF step0 | OFF 最终峰值 | **ON step0** | ON 最终峰值 | ON step0 − OFF 峰值 | ON 峰值 − OFF 峰值 |
|---|---|---|---|---|---|---|---|
| 0 | 19146 | 49.94 | 70.54 | **62.62** | 65.22 | −7.92 ✓ | −5.32 ✓ |
| 1 | 16895 | 53.04 | 71.40 | **58.46** | 61.17 | −12.94 ✓ | −10.23 ✓ |
| 2 | 19007 | 51.89 | 84.54 | **69.37** | 71.96 | −15.17 ✓ | −12.58 ✓ |
| 3 | 24773 | 40.16 | 69.62 | **71.88** | 74.47 | **+2.26 ✗** | **+4.85 ✗** |
| 4 | 19908 | 44.93 | 88.55 | **70.05** | 72.69 | −18.50 ✓ | −15.86 ✓ |
| 5 | 18041 | 40.26 | 88.61 | **61.88** | 64.48 | −26.73 ✓ | −24.13 ✓ |
| 6 | 18677 | 51.97 | 83.21 | **60.21** | 62.81 | −23.00 ✓ | −20.40 ✓ |
| 7 | 15778 | 44.63 | 88.11 | **59.65** | 62.24 | −28.46 ✓ | −25.87 ✓ |

**8 卡合计：ON step0 比 OFF 峰值低 130.46 GiB，ON 全程峰值低 109.54 GiB。**

读数要点：

- **ON step0 高于 OFF step0**（62.62 vs 49.94 …）——这是设计使然：把峰值提前到第一步。
- **但 ON step0 仍低于 OFF 在整个 run 里逐步爬到的峰值**（8 卡中 7 卡如此，−8 ~ −28 GiB），
  也就是说"提前支付峰值"的代价比"事后被碎片顶高"更低。
- **ON 全程无跳变**（reserved 恒定），OFF 则会在中途被更大的 batch 顶高（最高到 88.61 GiB，
  rank5 甚至出现过 reserved 88.61 → 释放 32 GiB 后重试成功）。
- 唯一超出的是 **rank3**（step0 +2.26 / 峰值 +4.85），原因与评估见 §8.1。

**epoch 矩阵验证**（3 / 10 / 20 / 40 / 60，全部通过、0 OOM）：

| epoch | 3 | 10 | 20 | 40 | 60 |
|---|---|---|---|---|---|
| rank3 reserved | 74.47 | 77.24 | 82.71 | 81.08 | 80.70 |

## 3. 配置、验证与生效确认

### 3.1 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `SWIFT_PAD_WARMUP` | **`0`（关闭）** | 设为 `1` 开启 |
| `SWIFT_PAD_WARMUP_LENGTH` | 无 | 显式指定目标长度，跳过重放 |
| `SWIFT_PAD_WARMUP_CACHE` | `1` | `0` 关闭长度扫描的磁盘缓存（见 §5） |
| `SWIFT_PAD_WARMUP_CACHE_DIR` | `<cwd>/.pad_warmup_cache` | 换一个缓存目录 |

> 默认关闭：功能需要按峰值形状多预留一部分常驻显存（见 §8），默认不改动基线行为。

**自动关闭**（无需额外配置）的条件：

- `context_parallel_size > 1`；
- `template.mode` 非 `transformers/train`（RLHF/PRM 按 `packed_seq_params.seq_lens`
  索引分段 logits，多一段会错位）；
- 长度不可得（扫描失败且未指定 `SWIFT_PAD_WARMUP_LENGTH`）。

### 3.2 交付物与 patch 验证

| 文件 | 内容 |
|---|---|
| `feature4_megatron_pad_warmup.patch` | 功能补丁（817 行） |
| `feature4_megatron_pad_warmup.md` | 本文档 |

**patch 覆盖范围**：

| 文件 | hunks |
|---|---|
| `swift/megatron/trainers/base.py` | 2（`import wrap_pad_warmup` + 调用点） |
| `swift/megatron/trainers/pad_warmup.py` | 新增文件，663 行（`@@ -0,0 +1,663 @@`） |
| `swift/template/base.py` | 3（dummy 段追加/切除 + `_pad_warmup_append_dummy` / `_pad_warmup_pop_dummy`） |

**验证结果**：

1. **往返一致性**：临时树上反向应用 → 正向应用，三个文件与环境中文件**逐字节一致**
   （说明 patch 恰好等于"基线 → 当前"的差量，无多余也无遗漏）。
2. **引用封闭性**：全包扫描 `pad_warmup|PadWarmup|SWIFT_PAD_WARMUP` 只命中上述 3 个文件；
   被引用的辅助符号（`pad_to_length`、`_sample_dummy_text_ids`、`_pad_warmup_append_dummy`、
   `_pad_warmup_pop_dummy`、`wrap_pad_warmup`、`replay_peak_batch_length`）定义均在 patch 内。
3. **无隐藏依赖**：`batch_sampler.MegatronPretrainingRandomSampler` 在 patch 中是**上下文行
   （未改动）**，属基线已有组件。

**迁移注意**：patch 带时间戳与行号上下文，目标基线一致可直接 `patch -p1`，有偏移需 `--fuzz`
或手工合入。

### 3.3 确认生效的日志

```
[pad_warmup] Scanning 156 samples (num_proc=32)...
[pad_warmup] Scan finished: 156 rows, 156 valid, 0 failed, mean=5662, max=13878, took 1.35s.
[pad_warmup] Replayed 3 epoch(s) for dp_rank=[0]: peak sum of 2 consecutive sample(s) = 19146.
[pad_warmup] warmup: pad the first 1 batch(es) per dataloader worker to 19146 (padding_free=True).
[pad_warmup] Warmed up 7 communicator(s) before the peak-shaped batch: default(8), ...
[pad_warmup] Padding the first batch to 19146 (pid ...).
```

关键看四件事：**扫描耗时**、**重放出的峰值**、**通信子预热个数**、**实际 pad 的长度**。

扫描只在 rank0 做一次，结果缓存到 `./.pad_warmup_cache/lengths_v2_<hash>.json`；
命中缓存时前两行替换为：

```
[pad_warmup] Using the cached sample lengths: 156 rows, 0 failed (0.04s, ...).
```

所以只有首次运行（或数据集/模型/模板/像素配置变化导致缓存 key 变化）才会真正付出扫描成本
（本数据集 156 条、`num_proc=32`，实测 1.35s；第二次 0.04s）。详见 §5。

---

# 技术细节

## 4. padding 长度怎么选

### 4.1 为什么不用 `Σ top-micro_batch_size`

旧实现取 `min(mbs * max_length, Σ 数据集最长的 mbs 条样本长度)`，即**所有 `C(n, mbs)` 种组合里
的最大值**——等价于假设"最长的几条样本恰好落在同一个 micro-batch"，随机 shuffle 下概率约
`1/C(n, mbs)`，基本不可能发生。

实测（156 条样本，mbs=2）：该值 = 25314，而典型 batch 只有 11323，**是它的 2.24 倍**。
按它预留会把 step0 直接打成整 run 的峰值。

### 4.2 重放 sampler（精确，非估计）

`MegatronPretrainingRandomSampler` 的 shuffle 是**确定性**的：

```python
g = torch.Generator(); g.manual_seed(self.epoch)      # 种子就是 epoch
random_idx = torch.randperm(bucket_size, generator=g).tolist()
```

所以可以精确重放每个 epoch、每个 dp_rank 的 micro-batch 划分，直接取最大值。
`replay_peak_batch_length()` 就是干这个的——**不需要任何分位数 / 覆盖率 / 魔数**，
天然适配任意长度分布、`mbs`、dp 大小、epoch 数、`data_sharding`、`group_by_length`。

重放结果与实测逐卡吻合到 1~2 个 token（差值是 2 MiB 段对齐造成的）：

| rank | 重放 epoch0 首个 batch | OFF 快照实测 | 差 |
|---|---|---|---|
| 0 | 14441 | 14442 | -1 |
| 1 | 16477 | 16479 | -2 |
| 3 | 4849 | 4850 | -1 |
| 5 | 7851 | 7852 | -1 |

### 4.3 只按「本卡」峰值（不是全局峰值）

每张卡有自己的设备，所以只该覆盖**它自己会遇到的 batch**。重放显示各卡自己的峰值
只有全局峰值的 64%~100%（平均 76%），用全局峰值会把 7/8 的卡无谓推高：

| rank | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| 本卡真实峰值 | 19146 | 16895 | 19007 | **24773** | 19908 | 18041 | 18677 | 15778 |
| 全局峰值 | 24773 | 24773 | 24773 | 24773 | 24773 | 24773 | 24773 | 24773 |

实现：`get_pad_warmup_length()` 取 `mpu.get_data_parallel_rank()` 传给 `replay_peak_batch_length()`。
各卡长度不同是安全的——dummy 段不参与 loss/梯度，各卡 token 分布本就不同。

### 4.4 版本演进：中间版本踩过的坑

从原始 feature4 到现在，一共试了 4 个版本（3 epoch，8 卡，本数据集）：

| 版本 | 做法 | 结果 | 为什么不行 / 为什么行 |
|---|---|---|---|
| **v0** | 原始 feature4：`Σ top-mbs` 作目标、全局峰值、按 worker 计数、`info_once` | **step0 XCCL `-705` 崩**（rank4 只剩 1.48 GiB） | 目标值 25314 比典型 batch 大 2.24 倍；且 peak 步撞上 XCCL 冷启动 |
| **v1** | v0 + 通信子预热 | step0 过，但**第 15 步 torch 碎片 OOM**（18.42 GiB） | 只解决时序冲突，没解决"预留过大" |
| **v2** | 改重放 sampler（精确）+ 全局峰值 + 每 worker 计数 | step0 降了，但**每 epoch 都被再预热一次** | `persistent_workers=False` → 每 epoch fork 新 worker；曾在第 10 步造出 23.42 GiB 人为峰值 |
| **v3（当前）** | 重放 + **按本卡峰值** + **全局只预热一次** + 逐次日志 + 修掉 epoch 循环 bug | **7/8 卡优于 OFF，无跳变，27/27、0 OOM** | — |

配套代码清理：`_fits_in_headroom` / `_peak_bytes_per_token` / `_vocab_size` /
`_PEAK_BYTES_FACTOR`（显存预算）已**删除**。

## 5. 长度扫描与磁盘缓存（cache 加速扫描）

### 5.1 为什么需要

确定 warm-up 长度要做两件事，代价量级完全不同：

| 步骤 | 代价 | 是否需要缓存 |
|---|---|---|
| 逐条 `template.encode` 求每条样本的 token 长度 | **重**：多模态（视频）要读文件、抽帧、缩放，单条可达百毫秒级 | **需要**，是本节的主题 |
| 重放 sampler 求 micro-batch 峰值 | 纯整数运算，微秒级 | 不需要 |

前者是**纯函数**：

```
L[i] = f( row_i , tokenizer/processor , 模板与截断 , 多模态像素/帧配置 )
```

同一份数据 + 同一套配置 ⇒ 结果逐字节相同，因此可以把 `L` 数组整体落盘复用，第二次起直接读取。

### 5.2 缓存 key 的判定原则

> **凡是会改变"某条样本被 tokenize 后有多长"的因素，都必须进 key；凡是只影响重放组织的，都不能进 key。**

二者不对称：误进 key 的代价只是"多扫一次"，漏进 key 的代价是"用了错误长度 → 预留不足 → OOM"。所以设计时**偏向保守**，并用版本安全网兜底。

**必须进 key：**

| 类别 | 字段 |
|---|---|
| 样本内容与子集 | `dataset`（含 `#N` 采样后缀）、本地数据文件的**内容指纹**（`size` + `mtime_ns`）、`split_dataset_ratio`、`columns`、`custom_dataset_info`、`dataset_shuffle` |
| tokenizer / processor | `model`、`model_revision`（换模型 = 换 tokenizer 与视频抽帧/缩放 processor） |
| 模板与截断 | `template`、`system`、`max_length`、`truncation_strategy`、`max_pixels`、`agent_template`、`norm_bbox`、`use_chat_template`、`padding_free`、`template_backend`、`response_prefix`、`enable_thinking`、`preserve_thinking`、`add_non_thinking_prefix` |
| 多模态展开 | `MAX_PIXELS`、`MIN_PIXELS`、`VIDEO_MAX_PIXELS`、`VIDEO_MIN_PIXELS`、`VIDEO_TOTAL_PIXELS`、`FPS`、`FPS_MAX_FRAMES` |
| 安全网 | swift 版本、`Template` 类名、processor 类名 |

**不能进 key（只影响重放组织）：**

`train_iters`、`num_train_epochs`、`global_batch_size`、`micro_batch_size`、数据并行大小/秩、`data_sharding`、`train_dataloader_shuffle`、`group_by_length`、`seq_length`、`dataset_num_proc`、`output_dir`、`SWIFT_PAD_WARMUP_LENGTH`。

它们只决定"哪些样本被放进同一个 micro-batch、按什么顺序重放"，从不改变单条样本的长度。把它们写进 key 会让"只改 epoch 数或并行度"也必须重扫一遍——正是要避免的开销。

> 安全网的作用：`swift` 版本与 template/processor 类名进 key，使得框架升级后所有旧条目自动失效。这样即使将来新增了某个影响长度的参数而没被列进上表，也只会退化成"重扫一次"，不会算出错误长度。

### 5.3 缓存放在哪

默认落在**当前工作目录**的 `.pad_warmup_cache/`，而不是 `$HOME` 下的全局目录：长度缓存和 one-shot
flag 都是一次性产物，放在工作区里会随工作区一起消失，不会无限累积。可用
`SWIFT_PAD_WARMUP_CACHE_DIR` 换位置。

### 5.4 两条不变量

**不变量 1 —— 一条样本一个槽位。**
`encode` 失败的样本（视频解码失败等）**保留槽位并填 `0`，绝不从数组中丢弃**。否则数组变短后 `lengths[i]` 描述的就不是第 `i` 条样本，重放会把错误样本的长度组合起来，算出的峰值可能偏小。配套地，`replay_peak_batch_length` 会用 `len(hf_ds)` 校验 `len(lengths)`，两者不一致直接拒绝并关闭 pad-warmup。

**不变量 2 —— 缓存条目必须与数据集对得上。**
读取时校验条数，不符即作废重扫；写入用 `os.replace` **原子替换**（临时文件名带 pid），进程被杀或并发跑都不会留下半截文件；每个条目把**产生它的完整 key** 一并存下来，便于事后审计。

### 5.5 失败样本的语义

失败的样本长度未知，处理原则是**宁可不预留，也不预留错**：

| 情况 | 处理 |
|---|---|
| 失败率 > 5% | 放弃 pad-warmup，回退基线 |
| 失败率 ≤ 5%，且设置了 `max_length` | 重放时把该样本按 `max_length` 计入 —— 只会**高估**不会低估 |
| 失败率 ≤ 5%，且无 `max_length` | 放弃 pad-warmup（没有可用上界） |

### 5.6 明确不缓存的场景

| 场景 | 原因 |
|---|---|
| `streaming=True` | `IterableDataset` 未物化、不可重复迭代，不存在稳定的逐样本数组；直接禁用扫描并打日志 |
| 数据集自带 `lengths` / `length` 列 | `lazy_tokenize=False` 预处理的产物，直接读取，连扫描都不用做 |
| 数据集含随机增广 / 随机 prompt | 长度不可复现 |
| `packing=True` | 行是多样本拼接，逐样本长度不再是行长度；此时直接用 `max_length`，不走扫描 |

### 5.7 效果与开关

同一份数据集连续两次运行（156 条，`num_proc=32`）：

```
首次：Scanning 156 samples (num_proc=32)...
      Scan finished: 156 rows, 156 valid, 0 failed, mean=5662, max=13878, took 1.35s.
再次：Using the cached sample lengths: 156 rows, 0 failed (0.04s,
      ./.pad_warmup_cache/lengths_v2_<hash>.json).
```

两次重放出的峰值都是 **19146**，说明缓存只省时间、不改变结果。视频数据集上扫描本身要慢得多（需要解码抽帧），缓存收益也相应更大。

```
SWIFT_PAD_WARMUP_CACHE=0            # 关闭长度缓存
SWIFT_PAD_WARMUP_CACHE_DIR=/tmp/x   # 换一个缓存目录
```

> 不要用"只扫前 N 条再估算"来提速：那会系统性低估峰值，正好落在 §8.3 那个 OOM 机理上。

## 6. 只预热一次（全局）

`PadWarmupCollator` 跑在 dataloader worker 里，而 `dataloader_persistent_workers=False` 时
**每个 epoch 都会 fork 出新 worker**，拿到的是父进程里 `num_batches=1` 的副本——
per-process 计数会导致**每个 epoch 的首个 batch 都被再 pad 一次**（旧实现就是这个 bug，
曾在第 10 步制造出一个 23.42 GiB 的人为峰值）。

现在除 per-process 计数外，还用 `O_CREAT|O_EXCL` 原子抢占一个 flag 文件
（`./.pad_warmup_cache/warmed_<hash(output_dir,local_rank)>.flag`），
保证整个 run 每卡只 pad 一次。实测两个不同 pid，说明 epoch 边界确实换了 worker 并被拦住：

```
[pad_warmup] Padding the first batch to 24773 (pid 1160601)          ← epoch0
[pad_warmup] Warm-up already done, skip padding (pid 1163415)        ← epoch1，被拦
```

**日志不用 `info_once`**，每次触发都打印，方便事后审计是否出现预期外行为。

### 6.1 eval / 验证集不会额外触发

flag 文件是按 `hash(output_dir, local_rank)` 命名的**全局一次性**标记，不区分 train / val：

- train 和 val 的 dataloader 共用同一个 `self.data_collator`，但**整个 run 只会有一次 padding**
  ——谁先跑到就谁来 pad，另一个自动跳过。所以 eval 阶段**不会**再制造一次峰值形状的 batch，
  也就不会在验证时额外抬高显存水位。
- 反之，如果某个流程在训练之前先跑验证（eval 先触发），那这次 pad 会落在 eval batch 上，
  训练的第一个 batch 就不再 pad。**功能仍然只做一次**，但"把峰值放到训练第一步"的意图会被挪到
  eval。当前配置 `eval_steps=1200` 远晚于 step0，不受影响；若调整为先 eval，需留意这一点。
- 由于只 pad 一次，eval 的 batch 按真实长度运行，其显存需求必然 ≤ 峰值预留，不会超出 step0
  已经开出的池子。

## 7. warmup 时的通信子预热

**现象**：开启 pad-warmup 后在 step 1 崩溃。

```
[ERR][XCCL][kl3_init.cpp:237] param 10 data alloc failed
[WARN][XCCL][init.cpp:1059] allocate memory for device param failed
ValueError: [RUNTIME ERROR]: error code= 2, out of memory (at .../xccl/XCCLUtils.cpp:28)
```

**根因**：集合通信后端（XCCL/BKCL）在**某个 process group 第一次做集合通信时**才建 context，
且它走**裸设备 malloc，绕过 torch 缓存分配器**。
不开 pad-warmup 时这件事发生在**小 batch** 的 step 1；开了之后 step 1 **就是峰值步**，
设备已被 torch 缓存分配器占满 → 后端拿不到内存。

即：**不是峰值 batch 装不下，是通信子初始化的时机被挤到了显存最紧张的第 1 步。**

**做法**：`warmup_communicators()` 对每个非单卡 process group 各跑一次 1 元素 `all_reduce`，
把通信子分配挪到设备还空的时候。在 `wrap_pad_warmup()` 中无条件调用（try/except 兜底）。

> 曾一度判定"非必要"并删除，随后在 **60 epoch 验证中崩溃复现**，故重新加入。原因：epoch 数越多
> 各卡峰值普遍变大（rank4 从 13885 → 24773），step0 峰值大幅升高，step1 的 XCCL 初始化再度失败。
> 带它重跑 3 / 10 / 20 / 40 / 60 epoch **全部通过，0 XCCL 崩溃**。

### 7.1 实现审查：完整性与冗余

初版是**手写 9 个 getter 的列表**。审查后发现两处问题，已修：

| 问题 | 初版 | 修正后 |
|---|---|---|
| **遗漏** | 手写列表漏了 `get_expert_model_parallel_group`（就是 EP 组，本配置 size=8） | `dir(parallel_state)` 动态枚举所有 `get_*_group`，新增组自动覆盖 |
| **冗余** | 单卡组（tp/pp/cp 等，本配置 world size=1）也被预热，白占通信子 | 跳过 `world_size <= 1`（Megatron 对单卡组的集合通信有短路，不会建 context） |

**去重必须用对象身份，不能用 rank 集合**——这一点踩过坑：本配置下 `get_data_parallel_group()` 与
`get_expert_model_parallel_group()` 的 rank 集合都是全部 8 张卡，但它们是**两个独立的
ProcessGroup 对象、各自一份 XCCL context**。按 rank 集合去重会只预热 1 个、漏掉另 1 个
（实测会退化成只预热 2 个通信子）。改用 `id(group)` 去重后才正确。

**实测结果**（3 epoch，8 张卡）：

```
[pad_warmup] Warmed up 7 communicator(s) before the peak-shaped batch: default(8),
get_amax_reduction_group(8), get_data_parallel_group(8), get_expert_model_parallel_group(8),
get_expert_tensor_and_model_parallel_group(8), get_expert_tensor_model_pipeline_parallel_group(8),
get_intra_distributed_optimizer_instance_group(8)
```

初版 9 个（含单卡组、漏 EP 组）→ 现在 7 个（全部 size=8，覆盖完整）。
整体开销约 2.3 GiB（每个通信子约 330 MiB），在 step0 峰值之前分配，属于池子的一部分。

**仍存在的"冗余"**：7 个组的 rank 集合两两相同（都是全部 8 张卡），实际被使用的可能只有
2~3 个（dp 用于 grad-norm、EP 用于 MoE all-to-all）。但它们是不同对象、各自可能被不同代码路径
走到，从外部无法安全判定哪些不会被用，故保守全部预热。若要进一步压缩需按实际调用点裁剪，
属于 Megatron 侧改动。

日志会打印每个组的名字与 world size，便于审计；单个组失败只 warning 并继续，不影响其余。

## 8. 已知遗留问题

### 8.1 ON 与 OFF 的显存差异：机制与代价

#### 8.1.1 机制：ON 把"随机爬升"换成"确定性预留"

同一张卡、同一个 24773-token batch（3 epoch，`it=10`，此时两者 packed 长度完全相同）按块大小分档：

| 档 | OFF | ON | ON−OFF |
|---|---|---|---|
| >=20 GiB（fp32 logits） | 22.92 | 22.92 | **0** |
| 5-20 GiB（bf16 logits） | 11.46 | 11.46 | **0** |
| 1-5 GiB | 2.24 | 0 | −2.24 |
| **128 MiB - 1 GiB** | **8.50** | **15.20** | **+6.69** |
| <128 MiB | 24.50 | 24.90 | +0.40 |

逐步对齐（rank3，从 `it=1` 起两者每步 packed 长度完全相同，只有 `it=0` 被 pad 到 24773）：

| it | OFF packed / reserved | ON packed / reserved | 差 |
|---|---|---|---|
| 0 | 4849 / 40.16 | **24773** / 71.88 | +31.72 |
| 4 | **24773** / 68.76 | **24773** / 74.47 | **+5.71** |
| 5~26 | 相同 / **69.62** | 相同 / **74.47** | **+4.85 恒定** |

| | OFF | ON |
|---|---|---|
| logits 大块 | 每遇一个"新纪录" batch 就新分一对，旧的留着 → 攒 **1~6 对** | step0 只分 **1 对**，之后永远复用 |
| 中等块（激活 / MoE / attention workspace） | 按**当前** batch 尺寸 | 按**峰值**尺寸冻结（约 +7 GiB/卡） |
| 峰值 | 随机 69.6~88.6 GiB，取决于撞到几个"新纪录" | 恒定 = base + 1 对 logits + 峰值尺寸中等块 |

对账：rank3 `+6.69 − 2.24 + 0.40 = +4.85` ✓；rank5 `−27.50 + 7.25 − 3.49 − 0.39 = −24.13` ✓。

**一句话**：代价是"把所有中间 buffer 冻结在峰值尺寸"（约 +7 GiB/卡），
收益是"消除 OFF 攒下的多对 logits 大块"（本配置最多省 27 GiB/卡）。

#### 8.1.2 rank3 为什么是唯一净变高的卡

每卡都 pad 到自己的峰值，所以"持有峰值块"解释不了 rank3 的特殊性。真正的原因是
**OFF 的每卡峰值不由 batch 长度决定，而是随机爬升的结果**：

| 本卡峰值 | OFF 峰值 | ON 峰值 | rank |
|---|---|---|---|
| 15778 | **88.11** | 62.24 | rank7 |
| 16895 | 71.40 | 61.17 | rank1 |
| 18041 | **88.61** | 64.48 | rank5 |
| 18677 | 83.21 | 62.81 | rank6 |
| 19007 | 84.54 | 71.96 | rank2 |
| 19146 | 70.54 | 65.22 | rank0 |
| 19908 | **88.55** | 72.69 | rank4 |
| **24773** | **69.62** | 74.47 | rank3 |

本卡峰值最小的 rank7，OFF 峰值却最高（88.11）；本卡峰值最大的 rank3，OFF 峰值却最低（69.62）
—— 两者完全不相关。OFF rank7 是**阶梯式爬升**：
`it=0 44.63 → it=1 59.50 → it=8 73.39 → it=15 88.11`；
而 OFF rank3 在 `it=3` 后就平了（69.62），其 24773 峰值 batch 在 `it=4` 时
reserved 只从 68.95 → 68.76（几乎没涨，全从缓存复用）。

rank3 在第 5 步就撞到 24773 这个最大值、之后再没新纪录 → **OFF 也只攒了一对 logits**
（`[23468, 11734, 2298]`）→ ON 拿不到"消除多对"的收益，+7 GiB 成本就裸露出来。
其他卡的 OFF 攒了 4~6 对，节省盖过了成本。

#### 8.1.3 评判基准：要看余量，不是峰值

拿 OFF 的 rank3 当基准，等于拿 OFF 里**唯一没被碎片顶高的那张卡**当基准：

| rank | OFF 峰值 | OFF 最小 dev_free | ON 峰值 | ON 最小 dev_free |
|---|---|---|---|---|
| 0 | 70.54 | 20.75 | 65.22 | 26.08 |
| 1 | 71.40 | 19.89 | 61.17 | 30.13 |
| 2 | 84.54 | 6.75 | 71.96 | 19.33 |
| 3 | 69.62 | 21.67 | 74.47 | **16.82** |
| 4 | 88.55 | **2.74** ⚠ | 72.69 | 18.60 |
| 5 | 88.61 | **2.69** ⚠ | 64.48 | 26.82 |
| 6 | 83.21 | 8.09 | 62.81 | 28.49 |
| 7 | 88.11 | **3.19** ⚠ | 62.24 | 29.05 |

**OFF 有 3 张卡（4/5/7）dev_free 只剩 2.7~3.2 GiB，随时可能 OOM；ON 最差的一张（rank3）仍有 16.82 GiB。**

另一个直接证据 —— reserved 出现 >5 GiB 跳升的次数：

| rank | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| OFF | 1 | 1 | 3 | 2 | 3 | **5** | 3 | 3 |
| **ON** | **0** | **0** | **0** | **0** | **0** | **0** | **0** | **0** |

#### 8.1.4 已排除的假设

| 假设 | 验证方式 | 结果 |
|---|---|---|
| ON 多保留了完整的 bf16 logits 块 | `it=10` 同卡同 step 分档对比 | **排除**：两者 fp32/bf16 logits 块完全相同（差 0），差异全在 128 MiB - 1 GiB 档 |
| 预热大块被优化器状态拆走占住 | 追踪 step0 最大块在首次 `optimizer.step()` 后的状态 | **排除**：it=0~19 一直是 `1 block / inactive / active=0 MiB` |
| 预热大块被后续小 batch 切碎 | 同上 | **排除**：块始终完整 |
| dummy 集中成一条超长段（19924 token）导致按段长计费更贵 | 把 dummy 拆成多条（按真实样本最大长度切分）后重测 | **排除**：rank3 step0 从 71.88 → **75.32**，反而更差 |

### 8.2 epoch 重放必须覆盖最后一个 epoch

**现象**：rank2 / rank5 在 `it=18 → it=19`、rank6 在 `it=17 → it=18`，reserved 突增
+17.88 / +16.98 / +16.80 GiB。

**根因**：`consumed` 是**跨 epoch 累计的样本计数**，本来就会超过 `total`；重放里多余的
`if consumed >= total: break`（156 条样本、epoch2 时 `consumed=288`）把最后一个 epoch 砍掉了，
于是这三张卡的 target（16293 / 17705 / 18056）低于真实峰值（19007 / 18041 / 18677）。

**修法**：去掉该保护，`consumed` 改为**构造后再赋值**（构造函数里有
`assert consumed_samples < total_samples`）：

```python
sampler = MegatronPretrainingRandomSampler(..., consumed_samples=0, ...)
sampler.consumed_samples = consumed
```

**验证**：修复后各卡 target 全部等于真实峰值，rank2 的 step0 块变成
18006 MiB / 1 block / inactive，全程 reserved 恒定在 71.96 GiB，**跳变消失**，27/27、0 OOM。

### 8.3 评判指标：要看"峰值水位"，不是"稳态 reserved"

决定会不会 OOM 的是**峰值水位**（`max_memory_reserved`，池子瞬时冲到的最高点），
**不是稳态 reserved** —— 用后者评判会得出"ON 更费显存"的相反结论。

三组同口径对比（rank3，1 epoch）：

| 配置 | 稳态 reserved | **峰值水位** | 后续是否爬升 |
|---|---|---|---|
| OFF | 69.62 | **80.69** | 是：42.71→57.77→68.95→69.62，中途尖到 80.69 |
| ON + `empty_cache` | 69.36 | **76.07** | **是**：53.15→64.32→69.36 |
| **ON** | 73.83 | **73.83** | **否，it=1 起定格不动** |

ON 稳态最高，但**峰值水位最低**。OFF 把同一份需求以**尖峰**形式表现（80.69），
ON 把它变成**常驻**（73.83）——**ON 的最大瞬时占用比 OFF 低 6.86 GiB**。
这就是 pad-warmup 的价值：把峰值摊平成常驻。

### 8.4 epoch 矩阵验证（3 / 10 / 20 / 40 / 60）—— 全部通过

带通信子预热，跑到第 12 步（池子在 step0 已定格，12 步足够判定）：

| epoch | 3 | 10 | 20 | 40 | 60 |
|---|---|---|---|---|---|
| rank3 reserved (GiB) | **74.47** | 77.24 | **82.71** | 81.08 | 80.70 |
| XCCL 崩溃 | 0 | 0 | 0 | 0 | 0 |

20 epoch 完整跑（180 步）也曾验证：8 张卡 step0 后只涨 0.04~0.08 GiB，全程只有 2~3 个不同取值，
`max == 稳态`（无尖峰）——**不存在"样本一点点变长、显存一点点被吃掉"的情况**。

各卡 it=1 reserved（20 epoch 完整跑）：

| rank | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| it=1 | 70.40 | 66.96 | 76.44 | 82.67 | 75.58 | 68.54 | 63.28 | 70.77 |
| 第 179 步 | 70.44 | 67.01 | 76.48 | 82.72 | 75.66 | 68.58 | 63.33 | 70.82 |

### 8.5 遗留问题：warm-up 长度恒定时，reserved 仍随 epoch 数变化

**先说长度与 epoch 数的关系**（`MegatronPretrainingRandomSampler.__iter__`）：

```python
self.epoch = self.consumed_samples // active_total_samples
g = torch.Generator(); g.manual_seed(self.epoch)     # seed 就是 epoch
idx_range = torch.randperm(...)[...][self.data_parallel_rank::self.data_parallel_size]
```

epoch 即 seed → 每个 epoch 一个不同排列 → 样本配对方式随 epoch 变 → 各 epoch 的 micro-batch
长度峰值不同。重放 N 个 epoch 取最大值，**N 越大观测到的最大值单调不减**，趋近理论上限
（top-2 样本长度之和 25314）。离线复算与实测日志完全吻合（rank0：1ep 19146 / 20ep 22199 /
40ep 24087）。

> 注意 `_REPLAY_MAX_SAMPLES = 50_000_000`：`n_epochs × 样本数` 超限时退化成只重放 epoch 0
> 并告警，此时预留会**低于后面 epoch 的真实峰值**——即"预留低于真实峰值 → 峰值 batch 拿不到连续块"的那个 OOM 机理。

但**本节的遗留问题与长度无关**：rank3 的峰值长度在 1 / 20 / 40 epoch 下**恒为 24773**
（epoch 0 即最大），然而 reserved 并不恒定：

| epoch | 1 | 20 | 40 |
|---|---|---|---|
| rank3 reserved (GiB) | 73.83 | **82.71** | 81.04 |

**既不恒定，也不单调**（40 epoch 反而比 20 epoch 低 1.63 GiB）。20 epoch 跑两次均为 82.71，可复现。

差异 **100% 落在 step0 forward 里"分配完即释放"的块上**（L10 快照）：

| L10 快照 | 1 epoch | 20 epoch | 差 |
|---|---|---|---|
| reserved | 71.23 | **80.12** | +8.89 |
| **active** | 34.98 | **35.21** | **+0.23** |
| inactive | 36.25 | 44.91 | +8.66 |
| segments | 1072 | 1079 | +7 |

**active 几乎一样** —— 多出来的全部是"已分配又释放、留在缓存里"的空间。
20ep 额外开了 926 / 925 / 872 / 866 / 787 / 786 / 778 MiB 共约 **7.7 GiB 的新段**，
而 1ep 把这些分配复用了已有的大空闲块（1ep 的 9651 MiB 空闲块被切成 9651 + 1429 + 584 使用）。

那批新段的尺寸 = `256 experts × capacity × 2048 × 2B`（capacity 926/925/872…），
形状吻合 **MoE 专家容量缓冲区**；capacity 由 batch 的**实际路由分布**决定，不由 token 总数决定
（完整尺寸反推见 §8.6）。

逐层插桩进一步显示分叉位置（step0 forward，rank3）：

| 层 | 1ep reserved | 20ep reserved | Δ | 1ep max_alloc | 20ep max_alloc | Δ |
|---|---|---|---|---|---|---|
| L0 | 21.06 | 23.29 | +2.23 | 19.21 | 19.81 | **+0.60** |
| L8 | 27.76 | 28.55 | +0.79 | 21.94 | 22.83 | +0.89 |
| L12 | 31.70 | **41.48** | **+9.78** | 23.61 | 24.63 | +1.02 |
| L39 | 36.57 | 45.45 | +8.88 | 31.93 | 32.65 | +0.72 |

即：**live 从 L0 起每层就高 0.6~1.1 GiB，reserved 在 L8→L12 之间暴涨约 +10 GiB。**

### 8.6 根因（栈级别）：MoE all-to-all-v 的接收缓冲

分配栈用 `_record_memory_history` 采集（调用方式见附录 B）。

**分配量按调用点汇总（step0 forward，rank3）**：

| 调用点 | 3ep 分配量 | 20ep 分配量 | Δ | 次数(相同) |
|---|---|---|---|---|
| `grouped_linear.py:291`（TE 分组 GEMM = MoE 专家） | 268.68 GiB | 292.66 GiB | **+23.99** | 5280 |
| `grouped_linear.py:301` | 134.34 | 146.33 | +11.99 | 2160 |
| `permutation.py:646`（TE MoE token permute） | 90.94 | 99.57 | +8.63 | 240 |
| `mappings.py:440`（**all-to-all-v**） | 158.97 | 165.44 | +6.47 | 360 |
| `lora.py:442` / `lora.py:443`（专家层 LoRA） | 129.78 | 136.25 | +6.47 | 480 |

**次数完全相同，只是每次都更大（+9%）**。

其中最关键的一处是 `megatron/core/tensor_parallel/mappings.py:439`：

```python
else:
    # Unequal split (all2all-v)
    output = input.new_empty(size=[sum(output_split_sizes)] + list(input.size()[1:]), ...)
handle = torch.distributed.all_to_all_single(...)
```

即 **MoE 的 all-to-all-v 接收缓冲**，尺寸 = `sum(output_split_sizes) × hidden × 2B`，
由 batch 的**实际路由分布**决定，**不是由 token 总数决定**。

换算有效 permute token 数：

| | 单次尺寸 | = 有效 token 数 | 相对 T=24773 |
|---|---|---|---|
| 3 epoch | 774 MiB | 8 × T | 8.0（= topk） |
| 20 epoch | 925 MiB | ~9.57 × T | **8 × 1.2** |

**结论**：ON 的额外常驻开销、以及它随 epoch 数的增长，全部落在
**MoE dispatch / permute / 分组 GEMM / 专家层 LoRA 这条链路上**；
直接触发者是 `all_to_all_single` 的接收缓冲，其尺寸按路由后的
`sum(output_split_sizes)` 计算，20 epoch 时相当于多带了约 **1.2× 的容量余量**。

而**与 T 直接挂钩**的那些 buffer（如 `permutation.py:236` 的 774 MiB、
`chunk_delta_h.py:89` 的 776 MiB）在两次运行中**完全不变** —— 反证了差异不在 batch 形状。

**块尺寸反推**（用 `requested_size`，`size` 对已合并的空闲块不可靠；
`T=24773, V=248320, H=2048, topk=8`）：

| 块 | 反推形状 | 是什么 |
|---|---|---|
| 23467 MiB | T × 248320 × 4B | fp32 logits |
| 11733 MiB | T × 248320 × 2B | bf16 logits |
| 970 MiB ×3 | 248320 × 2048 × 2B | embedding / lm_head 权重及梯度 |
| 774 MiB | T × 8 × 2048 × 2B | MoE dispatch / permute tokens |
| 925 / 880 / 865 MiB | 256 × capacity × 2048 × 2B | 同上，但 capacity 由路由分布决定（见下） |

### 8.7 消除可能性评估（3ep vs 20ep 的 +8.28 GiB）

**已排除的所有输入侧因素**（全部实测）：

| 项 | 3ep | 20ep | 结论 |
|---|---|---|---|
| warm-up 长度（rank3） | 24773 | 24773 | 相同 |
| batch 序列 | `24773 10643 7568 11881 24773 10584` | 逐项相同 | 相同 |
| **dummy token 内容** | `n=19924 uniq=1097 md5=8528ec84f12aa92e` | **md5 完全相同** | 相同 |
| step0 `pre` / `fwd` live | 14.84 / 50.67 | 14.84 / 50.68 | 相同 |
| `train_iters` 代码通路 | 只进 `lr_decay_iters` / `wd_incr_steps` / 进度条 / `_num_epochs` | — | 无显存通路 |

另做了一次判别：20 epoch + `SWIFT_PAD_WARMUP_LENGTH=24773`（绕过重放、长度仍锁 24773）
→ rank3 = **80.56**，比自然重放的 82.71 低 2.15 GiB，但仍远高于 3 epoch 的 74.47。
即：**重放路径只解释约 2 GiB，剩下约 6 GiB 与长度和数据都无关。**

**但 all-to-all 的 split 分布确实变了**（`n_experts_split=8`）：

| | split_sum 样本 |
|---|---|
| 3ep (rank3) | 150563, 198184, **168235** … |
| 20ep (rank3) | 165152, 198184, **190960**, 181219 … |

即：**输入完全一致，路由结果却不同** —— 差异来自 MoE 路由本身（XPU 上 topk/sort 的
nondeterminism，或 dispatcher 内部按实际路由做的容量 padding）。

**结论**：这 8.28 GiB **不能用"改 warm-up 长度 / 改 dummy 采样"来消除**，因为输入已证明一致。
可行的方向只有两条：

1. **在 MoE dispatcher 侧固定容量**：显式设置 `moe_expert_capacity_factor`（或关闭"按实际路由
   padding 到 max"的行为），让 buffer 尺寸只由 `T` 决定 → 稳定在最小值（774 而非 925）。
   这需要改 Megatron/TE 侧配置，不在 pad-warmup 的职责内。
2. **接受它**：即便 20 epoch 下 ON 是 82.71 GiB，仍与 OFF 的峰值水位 80.69 同量级，
   且 40/60 epoch 回落到 81.08 / 80.70，未出现失控增长；矩阵 3/10/20/40/60 全部通过、0 OOM。

**ON vs OFF 那 4.85 GiB 则不应消除** —— 它是"把 OFF 的尖峰摊平成常驻"，
ON 峰值水位（73.83）反而比 OFF（80.69）低 6.86 GiB（见 §8.3）。

**此问题不在当前 patch 范围内，迁移后依然存在。**

---

# 附录

> 以下记录排查过程中用到的方法与数据，代码里已不保留对应钩子，需要时按此手法临时加回。

## 附录 A：用 torch memory snapshot 采集显存现场

日志里的 `memory(GiB)` 只是 `max_memory_allocated`，看不出段分布和碎片。
在 P800 + `torch_xmlir` 上实测可用：

```
_record_memory_history / _snapshot / _dump_snapshot : 均可用
snapshot keys   : ['allocator_settings', 'device_traces', 'external_annotations', 'segments']
segments[i]     : device / total_size / allocated_size / active_size / requested_size / stream / blocks[]
allocator_settings: {'expandable_segments': False, ...}   # XPU 不支持 expandable_segments
device_traces   : XPU 上条数很少（实测 8 条），timeline 视图基本不可用
```

采集点放在 `train_step()` 里 `self.optimizer.step()` **之前**；不要全程开
`_record_memory_history`（35B 模型一个 step 几百万次分配，会打满 `max_entries` 并严重拖慢），
dump 前先 `enabled=None` 停记录。

**务必按 rank 分别记录**——本次崩的是 rank4，而只看 rank0（dev_free 18 GiB）会得出完全错误的结论。

pkl 拖到 https://docs.pytorch.org/memory_viz 查看。主要用 `segments` 做离线统计：

```python
segs = snap['segments']
reserved  = sum(s['total_size'] for s in segs)
free_blks = sorted((b['size'] for s in segs for b in s['blocks']
                    if b['state'] != 'active_allocated'), reverse=True)
```

本次靠它拿到的关键结论（ON，step0 顶到 25314，rank0）：

```
segments = 1067   reserved = 74.71 GiB   allocated = 31.67 GiB
free(cached) = 193 块共 43.04 GiB，最大空闲块 23980 MiB (23.42 GiB)
```

23.42 GiB ÷ (248320×4B) = 25314 → 反推出 step0 行长就是 25314；同时证明
"torch 自己分配不出来"这个假设不成立（池里明明有 23.42 GiB 空闲大块）。

## 附录 B：`_record_memory_history` 的正确调用（曾踩坑）

**踩坑**：一直传 `enabled=True`（bool），结果走了 legacy 分支、拿不到任何栈，
据此误判"该栈不支持 memory history"。

看 `torch/cuda/memory.py`：

```python
if isinstance(enabled, bool):
    return _record_memory_history_legacy(enabled, *args, **kwargs)   # 只能开关, 不记录栈
else:
    return _record_memory_history_impl(enabled, context, stacks, max_entries, ...)  # 真正实现
```

**必须传字符串**才走 `_record_memory_history_impl`：

```python
torch.cuda.memory._record_memory_history(enabled='all', context='all',
                                         stacks='python', max_entries=8000000)
# ... 跑一个 step ...
snap = torch.cuda.memory._snapshot()          # device_traces 里带 frames
torch.cuda.memory._record_memory_history(enabled=None)
```

注意事项：

- `context` 只接受 `'state'` / `'all'` / `None`，传 `'alloc'` 会报
  `RuntimeError: expected state to be 'state', 'all', or None`。
- `max_entries` 会截断：全程开着会因分配事件过多而打满，**只在需要的那一步前后开启**。
- 实测抓到 111k 事件（`alloc` 37,786 / `segment_alloc` 81），帧信息完整。
- `stacks='python'` 足够；`'all'` 会附 C++ 帧，体积更大。

本次 §8.6 的栈级别根因就是靠它拿到的。
