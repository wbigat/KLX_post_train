import types
import torch
from swift.template.base import Template
from swift.utils.transformers_utils import get_packed_seq_params

print("========== 模式1: padding_free=True (dummy -> 新段) ==========")
fake = types.SimpleNamespace(
    use_megatron=True, padding_free=True, packing=False,
    tokenizer=types.SimpleNamespace(pad_token_id=151643),
    _data_collator_mm_data=lambda row: {})
batch = [
    {"input_ids": [11, 12, 13], "labels": [11, 12, 13], "length": 3, "loss_scale": [1., 1., 1.]},
    {"input_ids": [21, 22], "labels": [21, 22], "length": 2, "loss_scale": [1., 1.]},
]
# 真实样本保持不动
Template._append_warmup_dummy(fake, batch, 10)
assert len(batch) == 3, "应追加 1 个 dummy"
dummy = batch[2]
print("dummy length:", dummy["length"], "(期望 10-5=5)")
assert dummy["length"] == 5
assert dummy["labels"] == [-100] * 5 and dummy["loss_scale"] == [0.] * 5
# 真实样本未被改动
assert batch[0]["input_ids"] == [11, 12, 13] and batch[1]["input_ids"] == [21, 22]
packed = Template.packing_row(fake, batch)
cu = get_packed_seq_params(torch.tensor([packed["position_ids"]]))["cu_seq_lens_q"].tolist()
print("packed 总长:", len(packed["input_ids"]), "| cu_seqlens:", cu)
assert cu == [0, 3, 5, 10]
assert packed["labels"][-5:] == [-100] * 5
print("-> dummy 恰好 1 个新段, OK\n")

print("========== 模式2: padding_free=False (dummy -> 新行, 强迫 pad 到峰值) ==========")
fake2 = types.SimpleNamespace(
    use_megatron=True, padding_free=False, packing=False,
    tokenizer=types.SimpleNamespace(pad_token_id=0))
batch2 = [
    {"input_ids": [11, 12, 13], "labels": [11, 12, 13], "length": 3, "loss_scale": [1., 1., 1.]},
    {"input_ids": [21, 22], "labels": [21, 22], "length": 2, "loss_scale": [1., 1.]},
]
Template._append_warmup_dummy(fake2, batch2, 6)   # 单样本峰值 L=6
assert len(batch2) == 3, "应追加 1 个 dummy"
d = batch2[2]
print("dummy length:", d["length"], "(期望 6, 单样本峰值)")
assert d["length"] == 6
assert d["labels"] == [-100] * 6 and d["loss_scale"] == [0.] * 6
# 真实样本长度保持原样(real_seq_lens 天然正确, 无需 _real_seq_len)
assert [len(b["input_ids"]) for b in batch2[:2]] == [3, 2]
# batch 内最大长度被 dummy 顶到 6 -> 现有 collator 会把所有样本 pad 到 6
assert max(len(b["input_ids"]) for b in batch2) == 6
print("-> dummy 作为最长样本, 真实样本不动, collator 自然 pad 到 6, OK")

print("\n=== 统一 append-dummy 两种模式验证通过 ===")
