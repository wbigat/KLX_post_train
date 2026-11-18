# feature4: megatron SFT pad_warmup（dummy 段方案，padding_free 开/关统一）

## 1. 背景与动机

### 问题

在**不支持 `expandable_segments`** 的硬件（如 P800/XPU）上，PyTorch caching allocator 的空闲段一旦切小就无法原地扩张。batch 的显存形状随数据顺序波动——`padding_free` 把一个 micro-batch 拼成一行（行长 = Σ样本长度）；非 `padding_free` 则按批内最长样本右 pad（形状 = [mbs, L_batch]）——前面跑的短 batch 把显存切成小段，后面来的长 batch 即便总空闲显存足够，也凑不出一块连续内存：

```
torch.OutOfMemoryError: Tried to allocate 15.60 GiB. GPU 3 has a total capacity of
96.00 GiB of which 38.53 GiB is free...
```

> 注：训练脚本里虽有 `PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'`，但 P800 上该特性不生效，碎片问题依旧。

### 解法（warmup，统一 dummy 段思想）

把**第一个 micro-batch 补到峰值形状**，让分配器提前预留出峰值形状的内存段，之后所有 batch（尺寸 ≤ 峰值）都复用这些段，从源头消除碎片。两条路径共用同一个 dummy 构造（真实样本不动、dummy 纯文本采样、`labels=-100`/`loss_scale=0`）：

- **padding_free**：向 batch 追加一个 dummy 样本把整行补到峰值行长 W，dummy 独立成段（逢 0 分段）；
- **非 padding_free**：追加一条长度 W 的 dummy 样本，把整批右 pad 顶到 `[mbs, W]`，collate 完成后把 dummy 行切掉（`_pop_warmup_dummy`），batch 内容不变、仅让分配器看到峰值形状。

## 2. feature3 回顾与局限

feature3（`feature3_megatron_pad_warmup.patch`）采用**样本级右 pad**：

- 在 collator 之前，把 warmup batch 里**每个样本**都右 pad 到 `L1 = min(max_length, max_len(数据集))`
- 用 `_real_seq_len` 记录真实长度，collator 据此构建 4D attention mask 把 pad 列置零
- `position_ids` 在 pad 尾巴上**连续递增**（尾巴并入样本所在段）

**局限：显式禁用了 `padding_free` 和 `packing`**（`get_pad_warmup_cutoff` 与 `_pad_batch_to_length` 双重守卫），因此 padding_free 场景下完全不生效。

## 3. feature4 设计：dummy 段方案（最终架构）

### 核心思路

真实样本一个不动，统一在 `Template.data_collator` 入口处向 batch **追加一个 dummy 样本**（`_pad_warmup_append_dummy`），`_data_collator` 及所有子类模板**零改动**：

```
                data_collator(batch, pad_to_length=W)
                            │
                _pad_warmup_append_dummy(batch, W)   ← 唯一新增动作
                            │
        ┌───────────────────┴───────────────────┐
        │ padding_free(含 packing)               │ 普通 batch
        │ packing_row 把 dummy 拼成【独立新段】   │ 右 pad 把 dummy 顶成【新行】,
        │                                       │ collate 后 _pad_warmup_pop_dummy 切除
        ▼                                       ▼
   cu_seqlens 多一项                      batch 精确回到 [mbs, W]
```

```
原始 batch:  [样本A(3), 样本B(2)]           总长 5
warmup 目标: 10
追加 dummy:  长度 5                          → 总长 10
```

关键在 `position_ids`：dummy 样本的 `position_ids = range(pad)`（0 开头），而 `get_packed_seq_params` 的分段规则是"**逢 0 即新段**"：

```python
# swift/utils/transformers_utils.py
cu_seqlens = torch.cat([indices_q[position_ids_f == 0], total_len])
```

所以 dummy 恰好成为**一个独立新段**（不是 pad 个长度 1 的段）：

```
position_ids = [0,1,2, 0,1,  0,1,2,3,4]
cu_seqlens   = [0, 3, 5, 10]              ← 3 段, 末段(dummy)长 5
```

> ⚠️ 这正是之前"单次请求组合上百个长度 1 的请求"事故的根源：若尾巴 `position_ids` 全填 0，每个 pad token 都会被当成段起点。dummy 段方案靠 `packing_row` 自动生成 `range(pad)`，从机制上杜绝。

### 各字段取值

| 字段 | 值 | 说明 |
|---|---|---|
| `input_ids` | **从真实 batch 随机采样**（非全 0 / 非全 pad_token_id） | embedding 分布接近真实 → MoE router 保持均衡 |
| `labels` | `-100` | ignore_index，不产生 loss |
| `loss_scale` | `0.` | 梯度归零 |
| `position_ids` | `range(pad)` | 由 packing_row 生成，形成独立新段 |
| attention mask | 无 | varlen 分段隔离，不需要 mask |

### 峰值长度 W 的计算

| 场景 | W |
|---|---|
| `packing=True` | `max_length`（packing bin 上限即峰值） |
| `padding_free`（非 packing） | `min(mbs × max_length, Σ top-mbs 样本长度)`，再向上对齐 `get_padding_to(args)` |
| 非 `padding_free` | `min(max_length, max 单样本长度)`（top-1），再向上对齐 |

`Σ top-mbs 样本长度`：任意 batch 都是 mbs 个样本的子集，其总长 ≤ 这个值，是**任意 shuffle 顺序下的严格上界**；非 padding_free 的批形状由批内最长样本决定，上界即数据集最长单样本。`SWIFT_PAD_WARMUP_LENGTH` 可手动覆盖。

### MoE 均衡

dummy token 取自真实 batch（随机采样）而非同一 pad id，embedding 分布接近真实数据，router 输出近似均匀。残余的 aux loss 偏差仅出现在 warmup 的 1~2 个 micro-batch（每 dataloader worker 的首 batch），且 dummy 段比 feature3 的逐样本尾巴**总量更小、集中一处**，真实样本段零污染。

## 4. feature3 vs feature4 对比

| 维度 | feature3（样本级右 pad） | feature4（dummy 段，统一方案） |
|---|---|---|
| 适用场景 | 非 padding_free / 非 packing | **padding_free 开/关均可（含 packing）** |
| 填充位置 | collator 前，逐样本 | pf: packing_row 前整行；非 pf: 追加 dummy 样本顶起右 pad 后切回 |
| 段归属 | 尾巴并入样本段 | pf: 独立新段；非 pf: dummy 行 collate 后移除 |
| 目标长度 | 每样本到 `L1`（偏大，假设全样本同时最长） | `W`（精准上界） |
| 分段载体 | 4D mask + `_real_seq_len` | pf: `cu_seqlens` 自动多一项；非 pf: 不需要 |
| pad token 量 | `Σ(L1-real_i)`，散布所有段 | pf: `W-total` 集中 1 段；非 pf: 仅 1 条 dummy 样本且被切掉 |
| MoE 污染面 | warmup batch 全部样本段 | warmup batch 仅 dummy 段/行 |
| warmup 自身 OOM 风险 | 高（过度预留） | 低 |
| RL 兼容 | `_real_seq_len` 供 CP 定位 | 干扰 `get_last_tokens` 段计数 → **RL 显式禁用** |

## 5. 迁移指导（feature3 → feature4）

在容器内执行：

```bash
# 进入 site-packages
SP=$(pip show ms_swift | sed -n 's/^Location: //p' | tail -1)
cd "$SP"

# 1) 回滚 feature3(若已应用)
patch -p1 -R --dry-run < /workdir/docker_patchs/megatron/feature3_megatron_pad_warmup.patch \
  && patch -p1 -R < /workdir/docker_patchs/megatron/feature3_megatron_pad_warmup.patch

# 2) 应用 feature4
patch -p1 --dry-run < /workdir/docker_patchs/megatron/feature4_megatron_pad_warmup.patch \
  && patch -p1 < /workdir/docker_patchs/megatron/feature4_megatron_pad_warmup.patch

# 3) 校验
python -m py_compile swift/megatron/trainers/pad_warmup.py \
                     swift/megatron/trainers/base.py swift/template/base.py && echo OK
```

> 两个 patch 改的是同一批文件，**互斥**，不能叠加。回滚 feature3 前若提示 `Reversed patch detected`，说明已是干净状态，直接应用 feature4 即可。

## 6. 使用与开关

默认对所有 megatron SFT 训练生效（padding_free 开/关均可），无需改训练脚本：

```bash
# 训练脚本中无需修改, 默认开启
bash swift_megatron_qwen3_5_35b_moe_sft.sh

# A/B 对比: 关闭 warmup(退回原始行为)
SWIFT_PAD_WARMUP=0 bash swift_megatron_qwen3_5_35b_moe_sft.sh

# 手动指定 warmup 峰值长度(跳过数据集扫描)
SWIFT_PAD_WARMUP_LENGTH=8192 bash swift_megatron_qwen3_5_35b_moe_sft.sh
```

日志中出现 `[pad_warmup] warmup: pad the first 1 batch(es) ...` 即生效。

## 7. 已知限制

- **仅 SFT 路径**（`mode in {transformers, train}`）。DPO/GRPO/GKD/PRM 等按 `packed_seq_params.seq_lens` 段数索引 last-token logits，dummy 段会使计数错位 → 这些 trainer 自动禁用并打日志。
- **`context_parallel_size > 1` 不支持**（沿用 feature3 的限制）。
- warmup 只覆盖"数据级"长度波动；若训练中期出现超过 W 的 shape（如 GRPO rollout 变长），需把 W 设成全程最大值。
- 多模态 mrope 路径（Qwen2VL 系模板）已适配：dummy 自动补 `mm_token_type_ids=0`（纯文本段）与 `channel` 占位字段，`packed_seq_params.seq_lens` 仅含真实样本，channel loss/last-token 索引不受 dummy 段影响。

## 7.1 修复记录（2026-09-20/21）

初版 patch 在 `Template.data_collator` 中**无条件**把 `pad_to_length` 透传给 `_data_collator`，而 `Qwen2VLTemplate` 等约 20 个子类以旧签名（仅 `padding_to`）重写该方法 → 任何模板都直接 `TypeError: ... unexpected keyword argument 'pad_to_length'`（即使 warmup 未激活）。

**dummy 的 `input_ids` 仅从纯文本 token 采样**：多模态样本的媒体占位 token（`image_token_id`/`video_token_id`）数量必须与 ViT 输出的 embedding 数严格一致（`masked_scatter`），混入 dummy 会越界访存（`unspecified launch failure`）；`get_rope_index` 也依赖媒体 token 与 grid_thw 一一对应。修复：按 `mm_token_type_ids==0` 过滤文本位，并排除 `tokenizer.all_special_ids` 与模板的媒体 token id。

**最终架构重构（2026-09-21）**：dummy 追加/切除统一收敛到 `Template.data_collator` 单入口（`_pad_warmup_append_dummy` / `_pad_warmup_pop_dummy`），`_data_collator` 及全部子类模板（含 qwen.py）**零改动**，从结构上杜绝签名兼容问题；非 padding_free 的 dummy 行在 collate 后切除，patch 只涉及 3 个文件（`trainers/base.py`、`trainers/pad_warmup.py`、`template/base.py`）。

实测（P800 8 卡，Qwen3.6-35B-A3B LoRA，llava_150k）：
- **padding_free=true，100 step**：OFF 显存 39→68 GiB 持续爬升（249 步时 74 GiB），ON 恒定 52.78 GiB；稳态速度 3.85 vs 3.84 s/it 持平；balance loss 仅 warmup 首步略高（1.41 vs 1.24），随后同区间。
- **padding_free=false，100 step**：OFF 显存 44→88 GiB（55 步即达 92% 容量），ON 恒定 52.26 GiB；balance loss 两条曲线一致（dummy 行被切掉，零污染）。
- **`SWIFT_PAD_WARMUP_LENGTH=1024` 覆盖 + dataloader_num_workers=2，30 step**：覆盖生效（跳过扫描），各 worker 独立 pad 首 batch，显存恒定 64.8 GiB，eval 正常。
- **packing=true**：warmup 分支逻辑已由 CPU 单测覆盖（batch-of-list 展平后 dummy 成段）；本环境 `packing` 与 `lazy_tokenize` 互斥，且关掉 lazy 后 LoRA 初始化存在**与 warmup 无关的既有 bug**（`TEGroupedLinear.weight0`），packing 端到端暂不可跑。
- 每次 run 收尾的 50 步 eval 均正常（eval 路径覆盖）。

## 8. 验证

CPU 级功能验证（无需 XPU）已在容器内通过（`/root/dl/runtime/f4test/test_feature4_unified.py`）：

- dummy 段恰好 1 个（`cu_seqlens=[0,3,5,10]`），非 N 个长度 1 段
- `labels` 尾段全 `-100`、`loss_scale` 全 `0`
- 非 padding_free：dummy 样本把整批右 pad 顶到 W 后切回（`input_ids`/`labels`/`attention_mask`/`position_ids`/`loss_scale`/`channel`/`seq_lens` 均为 `[mbs, W]`），真实样本内容不变；批内已达 W 时不追加；`pad_to_length=None` 时行为与原版一致
- pop 兼容 mrope `position_ids (C,B,L)` 按 dim1 切除，不误切非 per-sample 字段（如 `pixel_values`）
- 两条路径 dummy 均只含纯文本 token（排除媒体占位/特殊 token）
- packing（batch-of-list 展平）路径 dummy 正常成段；非 train mode / 无 `input_ids` 行安全跳过
- 长度解析分支：packing→max_length、pf→`min(mbs×max_length, top-mbs 和)`、非 pf→`min(max_length, top-1)`、CP>1→None、rlhf→None、扫描失败→兜底
