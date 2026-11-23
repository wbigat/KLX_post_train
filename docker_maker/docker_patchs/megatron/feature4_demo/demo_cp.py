#!/usr/bin/env python3
"""Demo 8: pad-warmup under context parallelism (v8) -- split behaviour + precision neutrality.

Verifies, with the same math the real CP path uses (mcore_bridge `split_cp_inputs` /
`_handle_megatron_cp` / the masked loss):

1. PADDED-SPACE SIZING: under CP every sample is padded to a 2*cp multiple by the collator,
   so the warm-up length must be resolved over the padded lengths (raw-space peaks would
   under-reserve by up to group_size * (2*cp - 1) tokens, silently);
2. ZIGZAG SPLIT: the dummy segment is split per-segment like any other (rank r takes chunks
   (r, 2cp-1-r)); every rank holds dummy_len/cp dummy tokens, all with labels=-100, and the
   ranks' chunks reconstruct the original packed row exactly;
3. PRECISION NEUTRALITY: causal varlen attention is segment-isolated, so the real tokens'
   outputs are BITWISE identical with and without the dummy segment; the masked loss
   (sum(losses*mask)/mask.sum(), all-reduced over the dp+cp group) gets exactly zero from the
   dummy in both numerator and denominator;
4. the closed-loop overshoot guard compares in the padded space under CP.

Run:  python3 demo_cp.py            (loads pad_warmup.py -- the patch's build source)
      DEMO_PATCH=1 python3 demo_cp.py   (extracts pad_warmup.py from the patch instead)
"""
import importlib.util
import math
import os
import re
import sys
import tempfile
import types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
WARNINGS = []


def _stub_swift():
    if 'swift' in sys.modules:
        return
    swift = types.ModuleType('swift')
    utils = types.ModuleType('swift.utils')

    class _L:
        def info(self, m, *a, **k):
            pass

        def warning(self, m, *a, **k):
            WARNINGS.append(m)

        def info_once(self, m, *a, **k):
            pass

    utils.get_logger = lambda *a, **kw: _L()
    utils.get_env_args = lambda k, t, d: d
    dsu = types.ModuleType('swift.dataset.utils')
    dsu.LazyLLMDataset = type('LazyLLMDataset', (), {})
    sys.modules.update({'swift': swift, 'swift.utils': utils,
                        'swift.dataset': types.ModuleType('swift.dataset'),
                        'swift.dataset.utils': dsu})


_stub_swift()


def load_pad_warmup():
    if os.environ.get('DEMO_PATCH'):
        lines = open(os.path.join(HERE, '..', 'feature4_megatron_pad_warmup-v10.patch')).read().split('\n')
        start = next(i for i, l in enumerate(lines) if l.startswith('@@ -0,0 +1,'))
        count = int(re.match(r'@@ -0,0 \+1,(\d+) @@', lines[start]).group(1))
        src = '\n'.join(l[1:] for l in lines[start + 1:start + 1 + count]) + '\n'
        path = os.path.join(tempfile.mkdtemp(), 'pad_warmup.py')
        open(path, 'w').write(src)
    else:
        path = os.path.join(HERE, 'pad_warmup.py')
    spec = importlib.util.spec_from_file_location('pad_warmup_demo8', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['pad_warmup_demo8'] = mod
    spec.loader.exec_module(mod)
    return mod


def zigzag_split(x, segs, cp, rank):
    """Mirror of mcore_bridge.split_cp_inputs: per segment, view (2*cp, len/(2*cp)) and take
    chunks (rank, 2*cp-1-rank); concat the local chunks of every segment."""
    out = []
    for a, b in segs:
        v = x[a:b].view(2 * cp, -1)
        out += [v[rank], v[2 * cp - 1 - rank]]
    return torch.cat(out)


def varlen_causal_attn(x, segs):
    """Naive per-segment causal attention (fp64, CPU): the reference for segment isolation."""
    d = x.shape[1]
    out = torch.zeros_like(x)
    for a, b in segs:
        s = x[a:b]
        scores = s @ s.T / math.sqrt(d)
        causal = torch.tril(torch.ones(b - a, b - a, dtype=torch.bool))
        out[a:b] = scores.masked_fill(~causal, float('-inf')).softmax(-1) @ s
    return out


def masked_lm_loss(logits, labels):
    """sum(ce * mask) / mask.sum() -- the trainer's loss reduction shape."""
    ce = torch.nn.functional.cross_entropy(logits, labels, reduction='none')
    mask = labels != -100
    return (ce * mask).sum() / mask.sum()


def main():
    pw = load_pad_warmup()
    cp = 2
    m = 2 * cp

    print('== 1. padded-space sizing: warm batch lands exactly on the padded peak ==')
    # Real samples with awkward raw lengths; the collator pads each to a multiple of 2*cp.
    raw_lens = [13, 7, 21, 10]
    padded = [pw._cp_pad_length(v, cp) for v in raw_lens]
    assert padded == [16, 8, 24, 12], padded
    target = 60  # resolved warm-up length (padded-space peak, a multiple of m)
    total_padded = sum(padded)
    dummy_len = target - total_padded  # template._append_warmup_dummy in the padded space
    assert dummy_len == 60 - 60 == 0 or dummy_len > 0
    target = 72  # pick a target strictly above, to exercise a real dummy
    dummy_len = target - total_padded
    assert dummy_len == 12 and dummy_len % m == 0
    assert total_padded + dummy_len == target
    print(f'   raw={raw_lens} padded={padded}, dummy={dummy_len}, packed total == target={target}  OK')

    print('== 2. zigzag split: dummy is an ordinary segment; reconstruction is exact ==')
    g = torch.Generator().manual_seed(0)
    segs = [(0, 16), (16, 24), (24, 48), (48, 60), (60, 72)]  # 4 real + 1 dummy(12)
    input_ids = torch.randint(0, 32000, (72,), generator=g)
    labels = input_ids.clone()
    labels[60:] = -100
    position_ids = torch.cat([torch.arange(b - a) for a, b in segs])
    per_rank = {}
    for rank in range(cp):
        ids_r = zigzag_split(input_ids, segs, cp, rank)
        lab_r = zigzag_split(labels, segs, cp, rank)
        pos_r = zigzag_split(position_ids, segs, cp, rank)
        per_rank[rank] = (ids_r, lab_r, pos_r)
        assert ids_r.numel() == 72 // cp == 36                     # balanced load
        assert (lab_r == -100).sum().item() == 12 // cp == 6       # dummy labels survive the split
        assert pos_r.max().item() < 24                             # per-segment arange preserved
    # exact reconstruction: per segment, chunks [r0c0, r1c0, r1c1, r0c1]
    for a, b in segs:
        seg_len = b - a
        c = seg_len // (2 * cp)
        r0 = per_rank[0][0]
        r1 = per_rank[1][0]
        # locate the segment's local chunks: segments contribute 2 chunks per rank, in order
    # (simpler: rebuild via inverse index math)
    recon = torch.empty(72, dtype=input_ids.dtype)
    off = {0: 0, 1: 0}
    for a, b in segs:
        c = (b - a) // (2 * cp)
        for rank in range(cp):
            ids_r = per_rank[rank][0]
            first = ids_r[off[rank]:off[rank] + c]
            second = ids_r[off[rank] + c:off[rank] + 2 * c]
            recon[a + rank * c:a + (rank + 1) * c] = first
            recon[a + (2 * cp - 1 - rank) * c:a + (2 * cp - rank) * c] = second
            off[rank] += 2 * c
    assert torch.equal(recon, input_ids), 'zigzag split must be exactly invertible'
    print(f'   cp={cp}: each rank holds 36 tokens (6 dummy, labels=-100); reconstruction exact  OK')

    print('== 3. precision neutrality: attention bitwise-isolated, loss exactly unchanged ==')
    torch.manual_seed(0)
    x = torch.randn(72, 8, dtype=torch.float64)
    out_with = varlen_causal_attn(x, segs)
    real_segs = segs[:4]
    out_without = varlen_causal_attn(x[:60], real_segs)
    assert torch.equal(out_with[:60], out_without), 'real tokens must be bitwise unaffected'
    logits = torch.randn(72, 50, dtype=torch.float64)
    cls = torch.randint(0, 50, (72,), dtype=torch.long)
    cls[60:] = -100
    full_loss = masked_lm_loss(logits, cls)
    loss_without = masked_lm_loss(logits[:60], cls[:60])
    # dummy labels are -100: numerator and denominator both get exactly zero (the tiny
    # tolerance only absorbs the different fp summation ORDER over 72 vs 60 elements)
    assert abs(full_loss - loss_without) < 1e-12, (full_loss, loss_without)
    print(f'   real-token attention outputs bitwise equal; masked loss {full_loss:.6f} '
          f'identical with/without dummy  OK')

    print('== 4. overshoot guard compares in the padded space under CP ==')
    WARNINGS.clear()
    col = pw.PadWarmupCollator(lambda batch, **kw: batch, length=100, num_batches=0,
                               flag_path=None, padding_free=True, cp_size=cp)
    col([{'input_ids': [0] * 53}, {'input_ids': [0] * 45}])   # raw 98 <= 100, padded 56+48=104 > 100
    assert len(WARNINGS) == 1 and 'exceeded' in WARNINGS[0], WARNINGS
    WARNINGS.clear()
    col2 = pw.PadWarmupCollator(lambda batch, **kw: batch, length=100, num_batches=0,
                                flag_path=None, padding_free=True, cp_size=cp)
    col2([{'input_ids': [0] * 52}, {'input_ids': [0] * 46}])  # raw 98, padded 52+48=100 <= 100
    assert not WARNINGS, WARNINGS
    print('   warns on padded overshoot that a raw check would miss; silent when padded fits  OK')

    print('== 5. the warm-up target is always 2*cp-aligned (no dummy double-padding) ==')
    # `_align` alone only guarantees get_padding_to (cp, not 2*cp, without fp8/fused): a
    # target of 50 with cp=2 would make `_handle_megatron_cp` re-ceil the dummy and the
    # warm-up batch would overshoot the resolved length by cp tokens. Driven through the
    # packing path (max_length goes straight through `_finalize`; the env override that
    # used to exercise this no longer exists).
    for name in ('swift.megatron', 'swift.megatron.utils'):
        sys.modules[name] = types.ModuleType(name)
    meg_utils = types.ModuleType('swift.megatron.utils.utils')
    meg_utils.get_padding_to = lambda args: 2  # cp=2, plain attention (the risky case)
    sys.modules['swift.megatron.utils.utils'] = meg_utils
    pw.get_env_args = lambda k, t, d: k == 'SWIFT_PAD_WARMUP'
    tpl = types.SimpleNamespace(mode='train')
    args = types.SimpleNamespace(max_length=50, packing=True, padding_free=True,
                                 context_parallel_size=cp)
    length = pw.get_pad_warmup_length(args, None, tpl)
    assert length == 52 and length % (2 * cp) == 0, length
    # cp=1 keeps the target untouched (no phantom padding without CP)
    args2 = types.SimpleNamespace(max_length=50, packing=True, padding_free=True,
                                  context_parallel_size=1)
    assert pw.get_pad_warmup_length(args2, None, tpl) == 50
    print('   packing max_length=50 with cp=2/padding_to=2 -> 52 (2*cp-aligned); '
          'cp=1 -> 50 unchanged  OK')

    print('== 6. real `_append_warmup_dummy` (from the patch): fields, text-only ids, CP sizing ==')
    # Exec the exact method text the patch ships, on a fake template self.
    lines = open(os.path.join(HERE, '..', 'feature4_megatron_pad_warmup-v10.patch')).read().split('\n')
    i0 = next(i for i, l in enumerate(lines) if l.startswith('+    def _sample_dummy_text_ids'))
    i1 = next(i for i in range(i0, len(lines)) if lines[i].startswith('+    def data_collator'))
    src = '\n'.join(l[1:] for l in lines[i0:i1])  # strip the diff '+'
    assert '_append_warmup_dummy' in src and '_sample_dummy_text_ids' in src
    ns = {'torch': torch, 'math': math, 'List': list, 'Dict': dict, 'Any': object,
          'Optional': object}
    from typing import List, Dict, Any, Optional
    ns.update({'List': List, 'Dict': Dict, 'Any': Any, 'Optional': Optional})
    exec('class _FakeTemplate:\n' + src.rstrip() + '\n', ns)

    def _cp_len(self, n):
        c = self.sequence_parallel_size
        return n if (not self.use_megatron or c == 1) else math.ceil(n / (2 * c)) * (2 * c)

    t = ns['_FakeTemplate']()
    t.use_megatron = True
    t.padding_free = True
    t.sequence_parallel_size = cp
    t.tokenizer = types.SimpleNamespace(pad_token_id=0, all_special_ids=[1, 2])
    t.image_token_id = 32000  # media placeholder that must never enter the dummy
    t._get_megatron_cp_length = types.MethodType(_cp_len, t)
    rows = [
        {'input_ids': [1, 5, 6, 7, 32000, 32000, 8, 9, 10, 11, 12, 13, 14],
         'length': 13, 'labels': [1] * 13, 'loss_scale': [1.] * 13,
         'token_type_ids': [0] * 13, 'channel': 'ch0',
         'mm_token_type_ids': torch.tensor([0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]),
         'position_ids': torch.arange(13).expand(3, 13).clone()},
        {'input_ids': [2, 20, 21, 22, 23, 24, 25], 'length': 7, 'labels': [1] * 7,
         'loss_scale': [1.] * 7, 'token_type_ids': [0] * 7, 'channel': 'ch1',
         'mm_token_type_ids': torch.zeros(7, dtype=torch.int64),
         'position_ids': torch.arange(7).expand(3, 7).clone()},
    ]
    batch = [dict(r) for r in rows]
    total_padded = sum(t._get_megatron_cp_length(r['length']) for r in batch)  # 16 + 8 = 24
    target = 64  # 2*cp-aligned
    t._append_warmup_dummy(batch, target)
    assert len(batch) == 3
    dummy = batch[-1]
    assert dummy['length'] == target - total_padded == 40
    assert dummy['length'] % (2 * cp) == 0, 'dummy must need no further cp-padding'
    # every field a real row carries exists on the dummy, with matching lengths
    for key in ('loss_scale', 'token_type_ids'):
        assert len(dummy[key]) == 40, key
    assert dummy['channel'] == 'ch0'
    assert dummy['mm_token_type_ids'].shape == (40,) and dummy['mm_token_type_ids'].sum() == 0
    assert dummy['position_ids'].shape == (3, 40)
    assert (dummy['position_ids'] == torch.arange(40).expand(3, 40)).all()
    # text-only sampling: no media placeholder, no special token
    assert not ({32000, 1, 2} & set(dummy['input_ids'])), dummy['input_ids'][:20]
    # packing_row's field math over the batch no longer sees None / ragged lengths
    cat = torch.cat([r['mm_token_type_ids'] for r in batch], dim=-1)
    pos = torch.cat([r['position_ids'] for r in batch], dim=-1)
    assert cat.shape == (60,) and pos.shape == (3, 60)
    # padded layout (v10): NO dummy row -- the rows are padded to `target` through the
    # returned `padding_to` instead, so the carved shape is (mbs, target) and not
    # (mbs+1, target). See demo_v10_fixes.py §2 for the full case.
    t.padding_free = False
    batch2 = [dict(r) for r in rows]
    out_padding_to = t._append_warmup_dummy(batch2, 64, None)
    assert len(batch2) == len(rows) and out_padding_to == 64, (len(batch2), out_padding_to)
    # all-media batch: no text ids to sample -> pad_token_id fallback, still no crash
    t.padding_free = True
    batch3 = [{'input_ids': [32000, 32000], 'length': 2,
               'mm_token_type_ids': torch.tensor([1, 1])}]
    t._append_warmup_dummy(batch3, 8)
    assert batch3[-1]['input_ids'] == [0] * (8 - t._get_megatron_cp_length(2))
    print('   dummy carries mm_token_type_ids/3D position_ids/channel; text-only ids; '
          'CP-sized exactly (40 == 64 - 24)  OK')

    print('== 7. end-to-end CP precision: resolve == padded truth, pad lands exactly, '
          'split balanced ==')
    import random
    rng = random.Random(11)
    n7 = 400
    raw7 = [rng.randrange(37, 900) for _ in range(n7)]   # deliberately NOT 2cp-aligned
    # get_padding_to mirror of real swift (tp=1, no fp8, unfused attention): cp or None
    meg_utils.get_padding_to = lambda a: (a.context_parallel_size
                                          if getattr(a, 'context_parallel_size', 1) > 1
                                          else None)
    for cp in (2, 4):
        m = 2 * cp
        for mbs in (2, 4):
            args7 = types.SimpleNamespace(
                max_length=1 << 20, packing=False, padding_free=True,
                context_parallel_size=cp, micro_batch_size=mbs,
                data_sharding=False, train_dataloader_shuffle=True, group_by_length=False,
                num_train_epochs=2, train_iters=None, global_batch_size=mbs,
                tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
            ds7 = list(range(n7))
            pw.scan_sample_lengths = lambda *a: (list(raw7), 0)
            resolved = pw.get_pad_warmup_length(args7, ds7, tpl)
            padded7 = [pw._cp_pad_length(v, cp) for v in raw7]
            truth = pw.replay_peak_batch_length(args7, ds7, padded7, mbs, agg='sum',
                                                dp_rank=None)
            assert resolved == truth and resolved % m == 0, (cp, mbs, resolved, truth)
            # the raw-space peak would silently UNDER-reserve (the bug CP sizing fixed)
            raw_truth = pw.replay_peak_batch_length(args7, ds7, raw7, mbs, agg='sum',
                                                    dp_rank=None)
            assert raw_truth < resolved, (raw_truth, resolved)
            # Build the actual peak batch, pad it like `_handle_megatron_cp` does, and let
            # the patch's real `_append_warmup_dummy` close the gap to the target.
            epochs, _ = pw._replay_epochs(args7, n7, mbs, 1)
            B = pw._sampler_batches(n7, mbs, 1, 0, epochs, False, True)
            peak_rows = B[int(torch.tensor(padded7)[B].sum(1).argmax())].tolist()
            t = ns['_FakeTemplate']()
            t.use_megatron = True
            t.padding_free = True
            t.sequence_parallel_size = cp
            t.tokenizer = types.SimpleNamespace(pad_token_id=0, all_special_ids=[1, 2])
            t.image_token_id = 32000
            t._get_megatron_cp_length = types.MethodType(_cp_len, t)
            batch = []
            for i in peak_rows:
                v = raw7[i]
                batch.append({'input_ids': [5 + (j % 500) for j in range(v)], 'length': v,
                              'labels': [1] * v, 'loss_scale': [1.] * v,
                              'token_type_ids': [0] * v, 'channel': 'ch0',
                              'mm_token_type_ids': torch.zeros(v, dtype=torch.int64),
                              'position_ids': torch.arange(v).expand(3, v).clone()})
            t._append_warmup_dummy(batch, resolved)
            dummy_len = batch[-1]['length']
            # per-sample 2cp padding (`_handle_megatron_cp`), dummy already aligned
            assert dummy_len % m == 0, (cp, mbs, dummy_len)
            segs7, off7, labels_flat = [], 0, []
            for r in batch:
                lp = pw._cp_pad_length(r['length'], cp)
                segs7.append((off7, off7 + lp))
                labels_flat += ([-100] * lp if r is batch[-1] else [1] * lp)
                off7 += lp
            assert off7 == resolved, (cp, mbs, off7, resolved)   # lands EXACTLY on target
            # zigzag split: every rank holds resolved/cp tokens, dummy spread evenly
            lab_t = torch.tensor(labels_flat)
            for rank in range(cp):
                lab_r = zigzag_split(lab_t, segs7, cp, rank)
                assert lab_r.numel() == resolved // cp
                assert (lab_r == -100).sum().item() == dummy_len // cp
            print(f'   cp={cp} mbs={mbs}: resolved={resolved} == padded truth; '
                  f'raw truth={raw_truth} (would under-reserve); peak batch + dummy lands '
                  f'exactly; each of {cp} ranks holds {resolved // cp} tokens  OK')

    print('\nDEMO8 OK')


if __name__ == '__main__':
    main()
