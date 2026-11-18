# 特性二：首个 batch padding 到 cutoff_len（抗显存碎片）

一份自包含的迁移说明。改动只涉及 **2 个文件、5 个位置 + 3 行 import**，全部代码在下面逐字给出，
可直接手工应用到其他设备上的 LLaMA-Factory 副本。

- 目标文件 A：`src/llamafactory/data/collator.py`
- 目标文件 B：`src/llamafactory/train/sft/workflow.py`

---

## 1. 这个特性做什么

训练/评估的**第一个 batch** 强制 padding 到 `cutoff_len`（即 LMF 的最大句长），让 caching allocator
在显存被变长 batch 打碎之前，先把「峰值形状」所需的大段 segment 建立起来。之后所有 batch 走原生
`longest` 策略，其显存需求都 ≤ 首个 batch，可直接复用已缓存的大段。

关键性质：

- **不改训练流程**：不新增 step、不做 dummy forward/backward、不改数据顺序。
- **不改数值**：padding 位置 `attention_mask=0`、`labels=IGNORE_INDEX(-100)`，loss 与梯度和不 padding 时逐位一致。
- **只作用一次**：`pad_first_batch_to` 用完即置 0（消费式），train 还是 eval 谁先跑谁生效（对内存块预留而言等价）。

## 2. 开关

| 环境变量 | 默认 | 说明 |
| --- | --- | --- |
| `LLAMAFACTORY_PAD_FIRST_BATCH` | `1`（开启） | 设为 `0` 关闭，用于 A/B 对比实验 |

```bash
llamafactory-cli train xxx.yaml                                  # 默认开启
LLAMAFACTORY_PAD_FIRST_BATCH=0 llamafactory-cli train xxx.yaml    # 关闭
```

日志（正常情况只有第一条；`dataloader_num_workers>0` 时每个 worker 各打一条，pid 不同，属正常）：

```
[pad_warmup] Padding the first batch to 15360 (pid 344302).
[pad_warmup] cu_seqlens may still be split: 12361 zeros on the text axis of position_ids, expected 1.   # 仅异常时出现
```

---

## 3. 文件 A：`src/llamafactory/data/collator.py`

### A-1. 顶部 import（新增 `math`、`os`）

原：
```python
import copy
import inspect
from dataclasses import dataclass
```

改为：
```python
import copy
import inspect
import math
import os
from dataclasses import dataclass
```

### A-2. 新增模块级 logger

在 `if TYPE_CHECKING:` 块之后（即 `from .template import Template` 之后的空行处）新增：

```python
logger = logging.get_logger(__name__)
```

同时确认文件顶部已 import：
```python
from ..extras import logging
```
（原文件通常只有 `from ..extras.constants import ...`，需要补这一行。）

### A-3. `MultiModalDataCollatorForSeq2Seq` 新增字段与状态

定位 `class MultiModalDataCollatorForSeq2Seq` 的字段区与 `__post_init__`。

原：
```python
    template: Optional["Template"] = None
    processor: Optional["ProcessorMixin"] = None

    def __post_init__(self):
```

改为：
```python
    template: Optional["Template"] = None
    processor: Optional["ProcessorMixin"] = None
    # [TEMPORARY] Pad the FIRST batch to `pad_first_batch_to` tokens (0 disables it), so the caching
    # allocator reserves peak-shaped segments before variable-length batches fragment the memory.
    # Padded positions keep attention_mask=0 and labels=IGNORE_INDEX, so the loss is unaffected.
    pad_first_batch_to: int = 0

    def __post_init__(self):
```

并在 `__post_init__` 末尾（`self.get_rope_func = None` 那个 if/else 之后）追加一行：

```python
        self._padding_first_batch = False
```

### A-4. 新增方法 `_fix_pad_position_ids`

放在 `__call__` 定义之前（例如紧跟 `_compute_rope_position_ids_with_packing` 之后）。
**这是整个特性中最关键的一段，缺了它在 GDN / linear-attention 模型上会直接 OOM，见第 5 节。**

```python
    def _fix_pad_position_ids(self, features: dict[str, "torch.Tensor"]) -> bool:
        r"""[TEMPORARY] Make position_ids keep increasing over the right padding, returns whether applied.

        Some models (e.g. qwen3_5) fill the position_ids of padded positions with 0, while the varlen
        path derives cu_seqlens from `position_ids == 0` (see `prepare_fa_kwargs_from_position_ids`),
        so every padded position would become a new sequence: `NT` of the linear attention kernels
        explodes and a huge fp32 buffer is requested. Same trick as `dummy_image_right_padding_mrope`,
        but continuing the last valid position instead of restarting from 0, so cu_seqlens stays a
        single segment. The largest value written is `seq_len - 1`, exactly the upper bound of a normal
        `arange(seq_len)`, hence no position can go out of range.
        """
        position_ids = features.get("position_ids")
        attention_mask = features.get("attention_mask")
        if not torch.is_tensor(position_ids) or not torch.is_tensor(attention_mask):
            return False

        # only a plain right padding tail is handled
        if position_ids.dim() not in (2, 3) or attention_mask.dim() != 2 or self.tokenizer.padding_side != "right":
            return False

        # [bsz, seq_len] -> [1, bsz, seq_len]; unsqueeze returns a view, so in-place writes are kept
        position_ids = position_ids if position_ids.dim() == 3 else position_ids.unsqueeze(0)
        bsz, seq_len = attention_mask.shape
        if position_ids.size(-2) != bsz or position_ids.size(-1) != seq_len:
            return False

        applied = False
        for i, valid_length in enumerate((attention_mask != 0).sum(dim=-1).tolist()):
            if not 0 < valid_length < seq_len or attention_mask[i, :valid_length].eq(0).any():
                continue  # nothing to pad, or the padding is not a contiguous right tail

            offsets = torch.arange(1, seq_len - valid_length + 1, dtype=position_ids.dtype, device=position_ids.device)
            position_ids[:, i, valid_length:] = position_ids[:, i, valid_length - 1 : valid_length] + offsets
            applied = True

        return applied
```

### A-5. `mm_token_type_ids` 对齐（gemma4 一致性）

定位 `__call__` 内的 `if "mm_token_type_ids" in mm_inputs:` 分支。

原：
```python
        if "mm_token_type_ids" in mm_inputs:  # need tensor-like for gemma4
            mm_token_type_ids = mm_inputs.pop("mm_token_type_ids")
            max_len = max(len(ids) for ids in mm_token_type_ids)
            padded = []
```

改为：
```python
        if "mm_token_type_ids" in mm_inputs:  # need tensor-like for gemma4
            mm_token_type_ids = mm_inputs.pop("mm_token_type_ids")
            max_len = max(len(ids) for ids in mm_token_type_ids)
            if self.pad_first_batch_to:  # not consumed yet, the padding block below runs later
                # [TEMPORARY] this key is padded manually, keep it aligned with the padded input_ids
                max_len = max(max_len, self.pad_first_batch_to)
                if self.pad_to_multiple_of:
                    max_len = math.ceil(max_len / self.pad_to_multiple_of) * self.pad_to_multiple_of

            padded = []
```

> 这个 key 不走 `tokenizer.pad`，是手工 pad 到 batch 内最长，不同步就会和 `input_ids` 长度不一致。

### A-6. padding 主逻辑

定位 `__call__` 中调用父类的那一行（在 `mm_inputs["mm_token_type_ids"] = torch.tensor(...)` 之后）。

原（单独一行）：
```python
        features: dict[str, torch.Tensor] = super().__call__(features)
```

改为：
```python
        # [TEMPORARY] pad the first batch to the peak shape, padded positions stay masked
        self._padding_first_batch = self.pad_first_batch_to > 0
        if self._padding_first_batch:
            origin_padding, origin_max_length = self.padding, self.max_length
            longest = max(len(feature["input_ids"]) for feature in features)  # never shrink a batch
            self.padding, self.max_length = "max_length", max(self.pad_first_batch_to, longest)
            self.pad_first_batch_to = 0  # consumed, the following batches keep the longest padding
            logger.info(f"[pad_warmup] Padding the first batch to {self.max_length} (pid {os.getpid()}).")
            try:
                features: dict[str, torch.Tensor] = super().__call__(features)
            finally:
                self.padding, self.max_length = origin_padding, origin_max_length
        else:
            features: dict[str, torch.Tensor] = super().__call__(features)
```

三个要点：

1. `padding="max_length"` + `max_length` 是 `DataCollatorForSeq2Seq` 的原生能力，`labels` 会被同步 pad 到
   同一长度（内部判据 `self.padding == PaddingStrategy.MAX_LENGTH`，字符串 `"max_length"` 与之相等）。
2. 目标取 `max(pad_first_batch_to, longest)`：`tokenizer.pad` 只填充不截断，注入 fake image token 后样本
   可能超过目标，不兜这一手会拼不成 tensor。
3. `finally` 保证异常时也恢复原策略。

### A-7. 调用 position_ids 修正

定位 `__call__` 中 `if self.get_rope_func is not None:` 块末尾的「transformers 兼容」拼接之后。

原：
```python
            # For transformers compatibility, after https://github.com/huggingface/transformers/issues/39400
            if features["position_ids"].dim() == 3:
                features["position_ids"] = torch.cat(
                    [features["position_ids"][0].unsqueeze(0), features["position_ids"]], dim=0
                )
```

在其后追加：
```python
            # [TEMPORARY] keep cu_seqlens as a single segment for the padded first batch
            if self._padding_first_batch and self._fix_pad_position_ids(features):
                num_zeros = int((features["position_ids"][0] == 0).sum())
                if num_zeros != features["position_ids"].size(-2):  # expect one zero per sample
                    logger.warning(
                        f"[pad_warmup] cu_seqlens may still be split: {num_zeros} zeros on the text axis "
                        f"of position_ids, expected {features['position_ids'].size(-2)}."
                    )
```

---

## 4. 文件 B：`src/llamafactory/train/sft/workflow.py`

### B-1. import

原：
```python
from ...extras.misc import calculate_tps
```
改为：
```python
from ...extras.misc import calculate_tps, is_env_enabled
```

### B-2. collator 接线（唯一改动点）

定位 `data_collator = SFTDataCollatorWith4DAttentionMask(`。

原：
```python
        pad_to_multiple_of=8 if training_args.do_train else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
```

改为：
```python
        pad_to_multiple_of=8 if training_args.do_train else None,  # for shift short attention
        # [TEMPORARY] pad the first batch (train or eval, whichever comes first) to cutoff_len, so the
        # caching allocator reserves peak-shaped segments before variable-length batches fragment the
        # memory. Set LLAMAFACTORY_PAD_FIRST_BATCH=0 to turn it off for A/B comparison. Skipped for
        # generation, where padding to cutoff_len would only blow up the KV cache and slow decoding.
        pad_first_batch_to=(
            data_args.cutoff_len
            if is_env_enabled("LLAMAFACTORY_PAD_FIRST_BATCH", "1") and not training_args.predict_with_generate
            else 0
        ),
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
```

---

## 5. 为什么 position_ids 必须「递增」而不是「随便填个非 0」

这是本特性最容易踩、且后果最严重的一点，实测数据如下。

`transformers` 从 `position_ids` 反推变长序列边界的判据是：

```python
indices_q = (position_ids == 0).nonzero().view(-1)   # prepare_fa_kwargs_from_position_ids
```

而部分模型（实测 `qwen3_5_moe`）把 padding 位置的 `position_ids` 填 **0**：

```
modeling_qwen3_5_moe.py:1455:  position_ids = torch.zeros(
modeling_qwen3_5_moe.py:1629:  position_ids = position_ids.masked_fill(attention_mask == 0, 0)
```

于是每个 padding 位置都被当作一个新序列起点，`cu_seqlens` 从 1 段退化成上万段。在 GDN /
linear-attention 的反向里：

```python
# xpu_fla/python/ops/common/chunk_delta_h.py
chunks_per_seq = (seqlens + chunk_size - 1) // chunk_size   # chunk_size = 64
NT = int(chunks_per_seq.sum())
dh = q.new_empty(B, NT, H, K, V, dtype=torch.float32)
```

`NT` 被放大约 51 倍，`dh` 这一个 fp32 张量从 0.47 GiB 变成 24 GiB，直接 OOM。
实测（T=15360、有效 3000、H=32/K=128/V=128）：

| | cu_seqlens 段数 | NT | `dh` |
| --- | --- | --- | --- |
| 未修正 | 12361 | 12407 | **24.23 GiB** |
| 已修正 | **1** | **240** | **0.47 GiB** |

（24.23 GiB 与线上日志 `Tried to allocate 24.13 GiB` 吻合。）

**「填非 0 常量」是必要但不充分的**：有些实现用「递减」而非 `== 0` 判定边界，常量填充在真实 token 与
padding 的交界处仍会多切 1 段；`prepare_fa_kwargs_from_position_ids` 的注释也提到某些模型的
`position_ids` 并非单调。递增写法代价相同（一次 `arange` 广播），且写入最大值恰为 `seq_len - 1`，
与正常 `arange(seq_len)` 上界一致，**天然不会越界**，无需额外 clamp。

同类问题上游也踩过，`collator.py` 里 dummy image 的右侧 padding 处理即为先例
（注释：`avoid continual cuseqlens breaking varlen attention`，`LlamaFactory#10452`），
区别是它从 0 重新起，会多切 1 段。

## 6. 已知边界与行为

| 场景 | 行为 |
| --- | --- |
| 无 packing | 首个 batch pad 到 `cutoff_len`，完整生效 |
| `packing=true` | 样本长度本就是 `cutoff_len - 1`（`data_args.py` 会减 1），仅补齐数个 token |
| `neat_packing=true` + FA2 | pad 后被 `_unpad_packed_features` 裁掉，实际无效（不报错） |
| `predict_with_generate` | 主动跳过：该场景左 padding，pad 到 `cutoff_len` 只会放大 KV cache、拖慢解码 |
| `dataloader_num_workers>0` | 每个 worker 各持一份 collator 副本，各自 pad 一次首 batch；对预留只有好处 |
| `eval_on_start=true` | eval 会先消费掉这次机会，train 首 batch 不再 pad（eval 峰值低于训练，预留可能不足） |
| 多模态 | 只固定 token 序列维（含展开后的 image token）；`pixel_values` 不经过 pad，vision tower 峰值不受约束，需要靠 `image_max_pixels == image_min_pixels` 固定像素预算才能定形 |

其他注意：

- 若 `cutoff_len` 远大于数据真实长度分布，首个 batch 的显存需求会明显高于后续任何真实 step，
  可能出现「首步 OOM 但实际训练能跑完」——这是 fail-fast 的代价，正确做法是把 `cutoff_len` 设到贴近真实分布。
- 仅接线到 SFT workflow。若要接到 dpo/kto，注意 KTO 的 collator 一次 `__call__` 会走两次父类 `__call__`。
- `pad_to_multiple_of=8` 是仓库原有配置（LongLoRA shift short attention + Tensor Core 对齐），
  最终长度为 `ceil(max(cutoff_len, batch内最长) / 8) * 8`。若将来启用序列并行，对齐需求会升为 `sp_size` 的倍数，
  届时需同步调整该值。

## 7. 迁移后的验证

```python
# 1) 首 batch 被 pad、后续恢复 longest、loss token 数不变
#    期望：shapes 依次为 (1, cutoff_len 对齐值) -> (1, longest) -> (1, longest)
#         attention_mask.sum() 与 (labels != -100).sum() 三次完全相同
#
# 2) position_ids 修正有效（最关键）
from transformers.modeling_flash_attention_utils import prepare_fa_kwargs_from_position_ids as fa
cu = fa(features["position_ids"][0])[0][0]
assert len(cu) - 1 == features["input_ids"].size(0)   # 段数 == batch size
assert int(features["position_ids"].max()) == features["input_ids"].size(1) - 1
```

线上直接看日志即可：出现 `Padding the first batch to N` 且**没有** `cu_seqlens may still be split`
警告，即为正常。
