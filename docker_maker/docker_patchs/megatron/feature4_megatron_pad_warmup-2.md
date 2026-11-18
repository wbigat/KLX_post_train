# Pad-Warmup（feature4）：用首个 batch 预留峰值显存，消除变长 batch 的碎片 OOM

> **基线说明**：本环境的 `ms-swift` 是 4.3.1，但 pad-warmup 是 **feature4** 版本
> （`swift/megatron/trainers/pad_warmup.py` 为 feature4 新增文件，不在 wheel 的 RECORD 里；
> `swift/megatron/trainers/base.py`、`swift/template/base.py` 也已由 feature4 改过）。
> 本文描述的是 **feature4 当前实现**，不是早期的 feature3（feature3 只支持非 padding_free、
> 长度取 `min(cutoff, max_len(all_samples))`、padding_free 下自动关闭）。
> 下文所有改动都是在 feature4 之上做的。

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

## 2. feature4 的 dummy 段方案（padding_free 也支持）

feature4 相对 feature3 的关键扩展：用一条 **dummy 样本**把 packed 行补到目标长度，
`packing_row` 给它 `position_ids = arange(T)`，使其成为 `cu_seqlens` 里**独立的一段**
（分段边界是 `position_ids == 0`）。dummy 恒为 `labels=-100` / `loss_scale=0`，
因果隔离且零 loss/梯度贡献。dummy 的 token 从真实 batch 采样（不是单一 pad id），
避免把 MoE router 的负载均衡带偏。

## 3. 预留长度：重放 sampler，取「本卡真实峰值」

### 3.1 为什么不用 `Σ top-micro_batch_size`

旧实现取的是 `min(mbs * max_length, Σ 数据集最长的 mbs 条样本长度)`，即**所有
`C(n, mbs)` 种组合里的最大值**——它等价于假设"最长的几条样本恰好落在同一个 micro-batch"，
随机 shuffle 下概率约 `1/C(n, mbs)`，基本不可能发生。

实测（156 条样本，mbs=2）：该值 = 25314，而典型 batch 只有 11323，**是它的 2.24 倍**。
按它预留会把 step0 直接打成整 run 的峰值。

### 3.2 重放 sampler（精确，非估计）

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

### 3.3 只按「本卡」峰值（不是全局峰值）

每张卡有自己的设备，所以只该覆盖**它自己会遇到的 batch**。重放显示各卡自己的峰值
只有全局峰值的 64%~100%（平均 76%），用全局峰值会把 7/8 的卡无谓推高：

| rank | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| 本卡真实峰值 | 19146 | 16895 | 19007 | **24773** | 19908 | 18041 | 18677 | 15778 |
| 全局峰值 | 24773 | 24773 | 24773 | 24773 | 24773 | 24773 | 24773 | 24773 |

实现：`get_pad_warmup_length()` 取 `mpu.get_data_parallel_rank()` 传给 `replay_peak_batch_length()`。
各卡长度不同是安全的——dummy 段不参与 loss/梯度，各卡 token 分布本就不同。

### 3.4 实测：OFF vs ON 逐卡显存（reserved 口径，单位 GiB）

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
- 唯一超出的是 **rank3**（step0 +2.26 / 峰值 +4.85），原因见 §5.1。

### 3.5 版本演进：中间版本踩过的坑

从原始 feature4 到现在，一共试了 7 个版本，逐个记录（3 epoch，8 卡，本数据集）：

| 版本 | 做法 | 结果 | 为什么不行 / 为什么行 |
|---|---|---|---|
| **v0** | 原始 feature4：`Σ top-mbs` 作目标、全局峰值、按 worker 计数、`info_once` | **step0 XCCL `-705` 崩**（rank4 只剩 1.48 GiB） | 目标值 25314 是 `C(156,2)` 种组合的最大值，比典型 batch 大 2.24 倍；且 peak 步撞上 XCCL 冷启动 |
| **v1** | v0 + 通信子预热 | step0 过，但**第 15 步 torch 碎片 OOM**（18.42 GiB） | 只解决时序冲突，没解决"预留过大" |
| **v2** | 蒙特卡洛估"期望峰值"（17016） | **第 5 步就 OOM，比 v0 更差** | 17016 有 593 个 pair 会超过它，覆盖率 **0**；欠预留比不预留更糟 |
| **v3** | + 显存余量守卫（不够就关闭） | 能跑完，但**等于把功能关掉** | 掩盖问题而非修复 |
| **v4** | 改重放 sampler（精确）+ 全局峰值 + 每 worker 计数 | step0 降了，但**每 epoch 都被再预热一次** | `persistent_workers=False` → 每 epoch fork 新 worker，拿到 `num_batches=1` 的副本；曾在第 10 步造出 23.42 GiB 人为峰值 |
| **v5** | 重放 + **按本卡峰值** + **全局只预热一次** + 逐次日志 | step0 明显下降，但 **rank2/5/6 在第 20 步跳变 +17 GiB** | 重放代码里 `if consumed >= total: break` 把最后一个 epoch 砍了，这三张卡的真实峰值都在那个 epoch |
| **v6（当前）** | v5 + 修掉 epoch 循环 bug | **7/8 卡优于 OFF，无跳变，27/27、0 OOM** | — |

配套代码清理：`_fits_in_headroom` / `_peak_bytes_per_token` / `_vocab_size` /
`_PEAK_BYTES_FACTOR`（显存预算）已**删除**；`empty_cache` 只做过验证、**未落地**（见附录 C）。

## 4. 只预热一次（全局）

`PadWarmupCollator` 跑在 dataloader worker 里，而 `dataloader_persistent_workers=False` 时
**每个 epoch 都会 fork 出新 worker**，拿到的是父进程里 `num_batches=1` 的副本——
per-process 计数会导致**每个 epoch 的首个 batch 都被再 pad 一次**（旧实现就是这个 bug，
曾在第 10 步制造出一个 23.42 GiB 的人为峰值）。

现在除 per-process 计数外，还用 `O_CREAT|O_EXCL` 原子抢占一个 flag 文件
（`~/.cache/swift/pad_warmup/warmed_<hash(output_dir,local_rank)>.flag`），
保证整个 run 每卡只 pad 一次。实测两个不同 pid，说明 epoch 边界确实换了 worker 并被拦住：

```
[pad_warmup] Padding the first batch to 24773 (pid 1160601)          ← epoch0
[pad_warmup] Warm-up already done, skip padding (pid 1163415)        ← epoch1，被拦
```

**日志不用 `info_once`**，每次触发都打印，方便事后审计是否出现预期外行为。

## 5. 已知残留问题

### 5.1 rank3 比 OFF 峰值高 4.85 GiB —— 逐块对比与"OFF 基线"修正

#### 5.1.1 逐块对比（同一张卡、同一个 24773-token 的 batch）

| | ON step0（预热到 24773） | OFF step5（真实 24773） |
|---|---|---|
| reserved | 71.23 GiB | 68.76 GiB |
| **active（活着）** | **31.43 GiB** | **33.19 GiB** |
| **inactive（空闲缓存）** | **39.80 GiB** | **35.57 GiB** |
| segment 数 | **1072** | **1973** |
| 最大两个空闲块 | 23468 + **11734 MiB** | 23468 + **6793 MiB** |

- ON 的 **active 反而比 OFF 少 1.76 GiB**，多出来的全在 inactive（空闲缓存）+4.23 GiB；
- 具体是 **bf16 logits 那块**：ON 完整保留 11734 MiB，OFF 只剩 6793 MiB（被切小复用掉了）；
- 换来的是**碎片少一半**（1072 vs 1973 个 segment）。

#### 5.1.2 但"为什么只有 rank3 变高"—— 上面的解释不充分，真正原因在 OFF 侧

**每卡都持有自己的峰值形状块**（都 pad 到自己的峰值），所以"持有峰值块"不能解释 rank3 的特殊性。
真正的原因是：**OFF 的每卡峰值根本不由 batch 长度决定，而是随机爬升的结果**。

按本卡峰值排序（GiB）：

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

**本卡峰值最小的 rank7，OFF 峰值却最高（88.11）；本卡峰值最大的 rank3，OFF 峰值却最低（69.62）**
—— 两者完全不相关。

看 OFF rank7 的轨迹就知道它是**阶梯式爬升**，不是由 batch 驱动：

```
it=0 44.63 → it=1 59.50 → it=7 59.54 → it=8 73.39 → it=14 73.39 → it=15 88.11 (dev_free 3.19)
```

而 OFF rank3 的轨迹在 `it=3` 之后就平了（69.62），它的 24773 峰值 batch 在 `it=4` 时
reserved 只从 68.95 → 68.76（几乎没涨，全从缓存复用）。

#### 5.1.3 块级差集：ON 独有的最大一块是完整的 bf16 logits

`>=64MiB` 的块按大小分桶做差（ON step0 vs OFF step5，同为 rank3 / 24773 batch）：

| 块大小 | ON | OFF | |
|---|---|---|---|
| 23468 MiB（fp32 logits） | 1 | 1 | 相同 |
| **11734 MiB（bf16 logits，完整）** | **1** | **0** | **ON 独有** |
| 6793 MiB（bf16 被切后的残余） | 0 | 1 | OFF 独有 |
| 1938 MiB | 0 | 1 | OFF 独有 |
| 776 ×2 / 774 / 387 MiB | 有 | 0 | ON 独有 |
| 871 / 242 / 120 / 95 MiB | 0 | 有 | OFF 独有 |

`>=64MiB` 总量：**ON 52.79 GiB vs OFF 48.06 GiB（差 4.73 GiB）**；
同时 ON 只有 1072 个 segment，OFF 有 1973 个（OFF 多出的 ~900 个小段占约 2.3 GiB）。

即：**ON 把峰值形状块整块闲置持有，OFF 把它们切给别的分配复用了**——
这正是"预留"的代价，也是 ON 后续不再攀升的原因。

#### 5.1.4 已证伪的三个假设

| 假设 | 验证方式 | 结果 |
|---|---|---|
| 预热大块被优化器状态拆走占住 | 追踪 step0 最大块在首次 `optimizer.step()` 后的状态 | **证伪**：it=0~19 一直是 `1 block / inactive / active=0 MiB` |
| 预热大块被后续小 batch 切碎 | 同上 | **证伪**：块始终完整（18006 MiB / 1 block / inactive） |
| dummy 集中成一条超长段（19924 token）导致按段长计费更贵 | 把 dummy 拆成多条（按真实样本最大长度切分）后重测 | **证伪**：rank3 step0 从 71.88 → **75.32**，反而更差（其余卡变化 ±0.5~2.5，无一致改善） |

#### 5.1.5 严格对齐实验（3 epoch，同 step / 同卡 / 同 packed 长度）

逐步对齐 rank3 的 ON 与 OFF —— **从 `it=1` 起两者每步的 packed 长度完全相同**，只有 `it=0` 被 pad 到 24773：

| it | OFF packed / reserved | ON packed / reserved | 差 |
|---|---|---|---|
| 0 | 4849 / 40.16 | **24773** / 71.88 | +31.72 |
| 1 | 10643 / 57.73 | 10643 / 74.43 | +16.70 |
| 3 | 11881 / 68.95 | 11881 / 74.47 | +5.52 |
| 4 | **24773** / 68.76 | **24773** / 74.47 | **+5.71** |
| 5~26 | 相同 / **69.62** | 相同 / **74.47** | **+4.85 恒定** |

注意 OFF 在 `it=3` 池子已 68.95，到 `it=4` 处理 24773 峰值 batch 时**根本没涨**（68.95→68.76）。

**在 `it=10`（同 step、同卡、同 packed）按块大小分档**：

| 档 | OFF | ON | ON−OFF |
|---|---|---|---|
| >=20 GiB（fp32 logits） | 22.92 | 22.92 | **0** |
| 5-20 GiB（bf16 logits） | 11.46 | 11.46 | **0** |
| 1-5 GiB | 2.24 | 0 | −2.24 |
| **128 MiB - 1 GiB** | **8.50** | **15.20** | **+6.69** |
| <128 MiB | 24.50 | 24.90 | +0.40 |

对照 rank5（ON 比 OFF **少** 24.13 GiB）：`5-20GiB: OFF 52.53 vs ON 25.04 → −27.50`；
`128MiB-1GiB: +7.25`。

#### 5.1.6 真正的机制（修正了此前的错误说法）

**ON 多出来的不是 bf16 logits 块**——`it=10` 时两者的 fp32/bf16 logits 块完全相同（差 0）。
多出来的是 **128 MiB - 1 GiB 这一档的中等块**（激活 / MoE dispatch / attention workspace），
**每张卡都有**：rank3 +6.69、rank5 +7.25 GiB。

| | OFF | ON |
|---|---|---|
| logits 大块 | 每遇一个"新纪录" batch 就新分一对，旧的留着 → 攒 **1~6 对** | step0 只分 **1 对**，之后永远复用 |
| 中等块 | 按**当前** batch 尺寸 | 按**峰值**尺寸冻结（约 +7 GiB） |
| 峰值 | 随机 69.6~88.6 GiB，取决于撞到几个"新纪录" | 恒定 = base + 1 对 logits + 峰值尺寸中等块 |

完整对账：
- rank3：`+6.69 − 2.24 + 0.40 = **+4.85**` ✓
- rank5：`−27.50 + 7.25 − 3.49 − 0.39 = **−24.13**` ✓

**rank3 为什么是唯一净变高的卡**：它在第 5 步就撞到 24773 这个最大值、之后再没新纪录，
所以 OFF 也只攒了一对 logits（`[23468, 11734, 2298]`）→ ON 拿不到"消除多对"的收益，
+7 GiB 的成本就裸露出来。其他卡的 OFF 攒了 4~6 对，节省盖过了成本。

**一句话**：pad-warmup 的代价不是"多留一对 logits"，而是**把所有中间 buffer 都冻结在峰值尺寸**
（本配置约 +7 GiB/卡）；它的收益是**消除 OFF 攒下的多对 logits 大块**（本配置最多省 27 GiB/卡）。

拿 OFF 的 rank3 当基准，等于拿 OFF 里**唯一没被碎片顶高的那张卡**当基准。
真正应该看的是**余量**，而不是峰值：

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

ON 所有卡**全程 0 次跳升** —— 这正是预留生效的直接体现：池子在 step0 一次到位，之后不再增长。

**综上**：pad-warmup 把"随机的阶梯式爬升"换成了"确定性的一次性预留"。
rank3 是唯一看起来变高的卡，但它的绝对余量（16.82 GiB）远高于 OFF 最危险的三张卡（2.7~3.2 GiB）。

### 5.2 曾出现过的 ~17 GiB 跳变（rank2/5/6）——根因是重放少算了最后一个 epoch，已修

> 这一节记录一次**已被修复的 bug**及其定位过程，保留下来是因为排查手法可复用。

**现象**：rank2 / rank5 在 `it=18 → it=19`、rank6 在 `it=17 → it=18`，reserved 突增
+17.88 / +16.98 / +16.80 GiB。

**排查一：是不是被优化器占住了？** 否。dump 各卡实际 padding 参数后发现
`total_by_length_field == total_by_input_ids`（补长本身正确），
再看 step0 最大块在第一次 `optimizer.step()` 之后的状况：

| rank | it0 最大 segment | it1 / it2 / it17 / it18 同一 segment | 被 active 占住 |
|---|---|---|---|
| 0 | 18138 MiB | 18138 MiB / 1 block / **inactive** | **0 MiB** |
| 2 | 15434 MiB | 15434 MiB / 1 block / **inactive** | **0 MiB** |
| 4 | 13154 MiB | 13154 MiB / 1 block / **inactive** | **0 MiB** |

优化器那 ~2.55 GiB 分配在别处，**没有占用预热块**；预热块也**从未被切分**，一直完整空闲。

**排查二：那问题在哪？** 对比各卡 `target` 与离线重放的真实峰值：

| rank | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| 代码给的 target | 19146 | 16895 | **16293** | 24773 | 19908 | **17705** | **18056** | 15778 |
| 真实峰值 | 19146 | 16895 | **19007** | 24773 | 19908 | **18041** | **18677** | 15778 |

**恰好就是 2/5/6 三张卡的 target 偏小**——它们的真实峰值落在最后一个 epoch。

**根因**：`replay_peak_batch_length()` 里有个多余的保护：

```python
consumed = epoch * ((total // (group_size * dp)) * (group_size * dp))
if consumed >= total:      # 156 条样本，epoch2 时 consumed=288 >= 156，直接 break
    break
```

`consumed` 是**跨 epoch 累计的样本计数**，本来就会超过 `total`，这个保护把最后一个 epoch 砍掉了。

**修法**：去掉该保护；`consumed` 改为**构造后再赋值**（构造函数里有
`assert consumed_samples < total_samples`）：

```python
sampler = MegatronPretrainingRandomSampler(..., consumed_samples=0, ...)
sampler.consumed_samples = consumed
```

**验证**：修复后各卡 target 全部等于真实峰值（19007 / 18041 / 18677），
rank2 的 step0 块变成 **18006 MiB / 1 block / inactive**（= 19007 的 fp32 logits），
全程 reserved 恒定在 71.96 GiB，**跳变消失**，27/27、0 OOM。

### 5.3 评判指标：要看"峰值水位"，不是"稳态 reserved"

分析时踩过的坑：一开始拿**稳态 reserved** 当指标，得出"ON 比 OFF 多 3.7 GiB、是浪费"的错误结论。
决定会不会 OOM 的是**峰值水位**（`max_memory_reserved`，即池子瞬时冲到的最高点）。

三组同口径对比（rank3，1 epoch）：

| 配置 | 稳态 reserved | **峰值水位** | 后续是否爬升 |
|---|---|---|---|
| OFF | 69.62 | **80.69** | 是：42.71→57.77→68.95→69.62，中途尖到 80.69 |
| ON + `empty_cache` | 69.36 | **76.07** | **是**：53.15→64.32→69.36 |
| **ON** | 73.83 | **73.83** | **否，it=1 起定格不动** |

ON 稳态最高，但**峰值水位最低**。OFF 把同一份需求以**尖峰**形式表现（80.69），
ON 把它变成**常驻**（73.83）——**ON 的最大瞬时占用比 OFF 低 6.86 GiB**。
这就是 pad-warmup 的价值：把峰值摊平成常驻。

### 5.4 泛化验证：20 epoch / 40 epoch，池子是否还会爬

**20 epoch（180 步），0 OOM，8 张卡 step0 后只涨 0.04~0.08 GiB：**

| rank | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| it=1 | 70.40 | 66.96 | 76.44 | 82.67 | 75.58 | 68.54 | 63.28 | 70.77 |
| 第 179 步 | 70.44 | 67.01 | 76.48 | 82.72 | 75.66 | 68.58 | 63.33 | 70.82 |
| **step0 后增长** | +0.04 | +0.05 | +0.04 | **+0.05** | +0.08 | +0.04 | +0.05 | +0.05 |
| 全程不同取值数 | 2 | 3 | 2 | 3 | 2 | 2 | 2 | 3 |

179 步里每张卡只有 2~3 个不同取值，且 `max == 稳态`（无尖峰）。
**不存在"样本一点点变长、显存一点点被吃掉"的情况。**

**40 epoch（360 步）各卡 it=1：**

| rank | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| reserved | 75.04 | 72.02 | **87.00** | 81.04 | 79.56 | 73.49 | 68.93 | 79.05 |

### 5.5 遗留问题：warm-up 长度恒定时，reserved 仍随 epoch 数变化

rank3 的峰值长度在 1 / 20 / 40 epoch 下**恒为 24773**（epoch 0 即最大），但 reserved 并不恒定：

| epoch | 1 | 20 | 40 |
|---|---|---|---|
| rank3 reserved (GiB) | 73.83 | **82.72** | 81.04 |

**既不恒定，也不单调**（40 epoch 反而比 20 epoch 低 1.63 GiB）。20 epoch 跑两次均为 82.67，可复现。

已排除：batch 序列一致、warm-up 长度一致（实测首步 packed 均为 24773）、
step0 `pre` 状态一致（14.84 GiB）、step0 `fwd` 的 live 一致（50.67 vs 50.68）。

差异 **100% 落在 step0 forward 里"分配完即释放"的块上**：

| | 瞬时空闲块（MiB） | 合计 |
|---|---|---|
| 1 epoch | 871, 752, 730, 686 | 20.27 GiB |
| 20 epoch | 926, 925, 872, 866, 787, 786, 784, 778 | **29.15 GiB（+8.88）** |

→ 这批 buffer 的尺寸**不受 batch 形状控制**，而是随 epoch 数变化。量级与形状
（`num_experts=256`、`topk=8`、dispatcher `flex` + `deepep`）指向 **MoE dispatch/combine 缓冲区**，
疑似其容量参数随 `train_iters` / `num_train_epochs` 缩放。

**卡点**：该栈 `torch.cuda.memory._record_memory_history` 实际是 legacy no-op
（`device_traces` 8 个 device 全为 0 条事件，snapshot 里 `frames` 全空），
`torch.profiler(profile_memory=True)` 也不可用 → 拿不到分配栈，无法点名算子。

下一步：在 MoE dispatcher（`flex` / `deepep`）处插桩打印 buffer 尺寸，
或跑 2/5/10 epoch 看 reserved 曲线形状反推缩放量。

**此问题不在当前 patch 范围内，迁移后依然存在。**

## 6. 配置

| 变量 | 默认 | 说明 |
|---|---|---|
| `SWIFT_PAD_WARMUP` | `1` | `0` 关闭（A/B 对比） |
| `SWIFT_PAD_WARMUP_LENGTH` | 无 | 显式指定，跳过重放 |

自动关闭：`context_parallel_size > 1`；`template.mode` 非 `transformers/train`
（RLHF/PRM 按 `packed_seq_params.seq_lens` 索引分段 logits，多一段会错位）；长度不可得时。

## 7. 确认生效的日志

```
[pad_warmup] Scan finished: 156 rows, 156 valid, mean=5662, max=13878.
[pad_warmup] Replayed 3 epoch(s) for dp_rank=[0]: peak sum of 2 consecutive sample(s) = 19146.
[pad_warmup] warmup: pad the first 1 batch(es) per dataloader worker to 19146 (padding_free=True).
[pad_warmup] Padding the first batch to 19146 (pid ...).
```

## 8. 交付物与 patch 验证

| 文件 | 内容 |
|---|---|
| `feature4_megatron_pad_warmup.patch` | 功能补丁（28118 B，538 个 `+` 行） |
| `feature4_megatron_pad_warmup.md` | 本文档（功能说明 + 全部调试分析） |

**patch 覆盖范围**：

| 文件 | hunks |
|---|---|
| `swift/megatron/trainers/base.py` | 2（`import wrap_pad_warmup` + 调用点） |
| `swift/megatron/trainers/pad_warmup.py` | 新增文件，427 行（`@@ -0,0 +1,427 @@`） |
| `swift/template/base.py` | 3（dummy 段追加/切除 + `_pad_warmup_append_dummy` / `_pad_warmup_pop_dummy`） |

**验证结果**：

1. **往返一致性**：在临时树上反向应用 → 正向应用，三个文件与环境中文件**逐字节一致**
   （说明 patch 恰好等于"基线 → 当前"的差量，无多余也无遗漏）。
2. **引用封闭性**：全包扫描 `pad_warmup|PadWarmup|SWIFT_PAD_WARMUP` 只命中上述 3 个文件，
   无第 4 处引用；被引用的辅助符号（`pad_to_length`、`_sample_dummy_text_ids`、
   `_pad_warmup_append_dummy`、`_pad_warmup_pop_dummy`、`wrap_pad_warmup`、
   `replay_peak_batch_length`）定义均在 patch 内。
3. **无隐藏依赖**：`pad_warmup.py` 用到的 `batch_sampler.MegatronPretrainingRandomSampler`
   在 patch 中是**上下文行（未改动）**，属基线已有组件，非本 patch 引入。

**迁移时注意**：patch 带时间戳与行号上下文，目标基线一致可直接 `patch -p1`，有偏移需 `--fuzz` 或手工合入。

---

# 附录

> 以下记录排查过程中用到的方法与数据，代码里已不保留对应钩子，需要时按此手法临时加回。

## 附录 A：通信子预热——**非必要，代码与 patch 中均已移除**

> 本节记录一个**被评估为不必要、已删除**的改动，保留原因：失败现象的定位过程有复用价值。

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

**曾加的改动**：`warmup_communicators()` 对 default / tp / pp / cp / dp / tensor+dp /
model-parallel / expert-tp / expert-dp 各组各跑一次 1 元素 `all_reduce`，
把通信子分配挪到设备还空的时候。实测能让 step 1 从"必崩"变为顺利通过（81 GiB），
通信子合计 ~2.6 GiB。

**移除理由**：改为"长度按本卡峰值"后，step0 余量已足够（最紧的卡 dev_free 16.82 GiB，见 §5.1.6），
不再需要它。**已按要求删除，环境代码与 `feature4_megatron_pad_warmup.patch` 中均无残留。**

注：排查时先排除了 `max_length` 差异（30720 vs 20480）——用 20480 + 开 复跑同样在 step 1 崩，
确认与该参数无关。

## 附录 B：用 torch memory snapshot 采集显存现场

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

## 附录 C：`torch.cuda.empty_cache()` 实验

XCCL 触发不到 torch 的 `free_cached_blocks_and_retry`（那条路径只在 torch 自己的分配失败时走），
所以出现"设备显示还有 44 GiB 空闲却拿不到 330 MiB"——那 44 GiB 里 ~43 GiB 是 torch 缓存的空闲块。

在 `optimizer.step()` 之前调用一次（ON，各卡 24773）：

| rank | reserved 前 → 后 | dev_free 前 → 后 |
|---|---|---|
| 0 | 73.38 → 34.29 GiB | 19.33 → **58.42** GiB |
| 2 | 88.95 → 36.98 GiB | 3.76 → **55.73** GiB |
| 4 | **91.23 → 37.16 GiB** | **1.48 → 55.55** GiB |
| 5 | 77.51 → 34.34 GiB | 15.20 → **58.37** GiB |

效果对照（3 epoch）：

| 组合 | 结果 |
|---|---|
| 不 empty_cache、不通信预热 | step0 XCCL `-705` 崩（rank4 只剩 1.48 GiB） |
| 不 empty_cache + 通信预热 | step0 过，第 15 步 torch 碎片 OOM（18.42 GiB） |
| **每步 backward 后 empty_cache** | **27/27 通过，0 OOM**，8.4~9.4 s/it 与基线同量级 |

它同时解决两件事：① 让 XCCL 拿得到内存；② 每步清空，避免 §5.2 的跳变。
**未写进 `pad_warmup.py`**——"只第 1 步"还是"每步"、是否加空闲阈值条件，尚未定。

### 附录 C.1「只在 warm-up 之后 empty_cache 一次」——已撤回

曾试过一个更"省"的方案：step0（峰值 warm-up）结束后调一次 `empty_cache()`，
把冷启动时开出的碎片还给设备，让池子按实际用到的形状重新成型。

**实测（rank3）**：

```
it=1  reserved=74.43   ← warm-up 结束
      → empty_cache
it=2  reserved=53.15   ← 吐回 21.1 GiB
it=3  reserved=53.15
it=4  reserved=64.32   ← 真实峰值 batch 来了，+11.17
it=5  reserved=69.36   ← 再爬 +5.04
it=6~26 69.36/69.37
max = 76.07
```

**稳态确实更低（69.36 vs 73.83），但方案有漏洞，已撤回：**

1. **丢失了预留**：池子又变成"随样本长度逐步爬升"（53.15 → 64.32 → 69.36）。
   本次只在 it=5 稳住，纯粹因为该数据集最长样本恰好在 it=4；换数据集就会在某个更晚的
   step 突然顶高——正是 §附录 A 里 warm-up 长度不足导致 OOM 的同一机理。
2. **峰值水位反而升高**：74.43 → **76.07**，OOM 风险不降反升。

对照 §5.3 的三组数据：ON 峰值水位 73.83 最低，ON+empty_cache 76.07，OFF 80.69。
**稳态的下降是假收益。**

## 附录 D：块尺寸反推，以及 epoch 数如何影响 warm-up 长度

### D.1 大块尺寸反推

用 `requested_size`（不是 `size`，后者对已合并的空闲块不可靠）反推形状。
模型：`hidden=2048, layers=40, experts=256, topk=8, vocab=248320`，T=24773：

| 块 | 反推形状 | 是什么 |
|---|---|---|
| 23467 MiB | T × 248320 × 4B | fp32 logits |
| 11733 MiB | T × 248320 × 2B | bf16 logits |
| 970 MiB ×3 | 248320 × 2048 × 2B | embedding / lm_head 权重及梯度 |
| 774 MiB | T × 8 × 2048 × 2B | MoE dispatch / permute tokens |

### D.2 epoch 数越多，warm-up 长度越长

`MegatronPretrainingRandomSampler.__iter__` 里：

```python
self.epoch = self.consumed_samples // active_total_samples
...
g = torch.Generator()
g.manual_seed(self.epoch)                    # seed = epoch
idx_range_total = torch.randperm(full_bucket_size, generator=g).tolist()
idx_range = idx_range_total[...][self.data_parallel_rank::self.data_parallel_size]
```

**epoch 即 seed** → 每个 epoch 一个不同的排列 → 样本两两配对方式随 epoch 变 →
各 epoch 的 micro-batch 长度峰值不同。重放 N 个 epoch 取最大值，**N 越大观测到的最大值单调不减**，
趋近理论上限（top-2 样本长度之和 25314）。

离线复算（用缓存的 156 条样本长度）与实测日志完全吻合：

| rank | epoch 0 峰值 | epoch 0~19 | epoch 0~39 | 1ep / 20ep / 40ep 实测 reserved |
|---|---|---|---|---|
| 0 | **19146** | **22199** | **24087** | 63.18 / 70.44 / 75.04 |
| 1 | 16895 | 20685 | 22331 | 58.24 / 67.01 / 72.02 |
| 2 | 15843 | 19007 | 23840 | 68.28 / 76.48 / 87.00 |
| 3 | **24773** | **24773** | **24773** | 73.83 / 82.72 / 81.04 |
| 4 | 13885 | 19908 | 21148 | 58.72 / 75.66 / 79.56 |
| 5 | 17705 | 19514 | 21895 | 63.58 / 68.58 / 73.49 |
| 6 | 16252 | 18985 | 21895 | 58.80 / 63.33 / 68.93 |
| 7 | 14788 | 22331 | 22331 | 62.95 / 70.82 / 79.05 |

（rank0 的 19146 / 22199 / 24087 与三次运行的日志 `Padding the first batch to ...` 完全一致。）

注意 `_REPLAY_MAX_SAMPLES = 50_000_000`：`n_epochs × 样本数` 超限时退化成只重放 epoch 0 并告警，
此时预留会**低于后面 epoch 的真实峰值**——即 §附录 A 那个 OOM 机理。
