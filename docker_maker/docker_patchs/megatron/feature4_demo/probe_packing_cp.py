#!/usr/bin/env python3
"""Probe: does CP padding push a packing micro-batch PAST the v10 target?

v10's packing target is `mbs * packing_length`, justified by "each bin <= packing_length".
But `_handle_megatron_cp` pads EVERY SAMPLE to a 2*cp multiple BEFORE `packing_row`
(template/base.py:1975 -> 1520), so a bin that was full in RAW length becomes longer once
CP-padded. If so, the real micro-batch exceeds the target = UNDER-reservation.

This probe only measures; it changes nothing.
"""
import os
import sys
import types

root = '/tmp/t2t'
sys.path.insert(0, root)
import swift  # noqa: E402
assert swift.__file__.startswith(root), swift.__file__
from swift.dataset.packing import calculate_matched_group  # noqa: E402
from swift.template.base import Template  # noqa: E402


def make_template(cp):
    t = Template.__new__(Template)
    t.processor = types.SimpleNamespace(pad_token_id=0, all_special_ids=[1, 2])
    t.padding_side = 'right'
    t.mode = 'train'
    t.use_megatron = True
    t.padding_free = True
    t.sequence_parallel_size = cp
    t.packing = True
    t.task_type = 'causal_lm'
    t.image_token_id = 32000
    t.max_length = 4096
    t.remove_unused_columns = True
    return t


def row(n, i):
    return {'input_ids': [5 + ((i + k) % 3000) for k in range(n)],
            'labels': [1] * n, 'length': n, 'loss_scale': [1.] * n}


def pack(lengths, pl):
    seqs = [(row(n, i), n) for i, n in enumerate(lengths)]
    bins, _ = calculate_matched_group(seqs, pl, is_finished=True)
    return [[r for r, _ in b] for b in bins]


def real_len(t, bins, cp):
    """Length the REAL collator produces, with no warm-up target at all.

    `padding_to=cp` reproduces production: `_get_data_collator` binds
    `partial(template.data_collator, padding_to=get_padding_to(args))`, and for plain t2t with
    CP (no TP-SP/fp8/fused) `get_padding_to` returns exactly `cp`.
    """
    res = t.data_collator([list(b) for b in bins], padding_to=cp if cp > 1 else None)
    return int(res['input_ids'].shape[-1])


PL, MBS = 1024, 4
for cp in (1, 2, 4, 8):
    print(f'\n--- cp={cp} (2cp={2*cp}) ---')
    for name, lengths in (
            # 201 = 25*8+1 -> pads by 7 at 2cp=8; 5 per bin = 1005 raw, 1040 after CP
            ('crafted len=201 (max CP amplification)', [201] * 40),
            ('crafted len=249', [249] * 40),
            ('t2t short 80-400', [80 + (i * 37) % 320 for i in range(400)]),
            ('long 560-1000', [560 + (i * 53) % 440 for i in range(200)]),
            # the under-reservation scales with (samples per bin) * 2*cp, so short samples at
            # high CP are where the flat cap should hurt MOST -- measure it, don't extrapolate.
            ('very short 20-50 (many samples/bin)', [20 + (i * 7) % 30 for i in range(400)]),
            ('tiny 8-16', [8 + (i % 9) for i in range(400)]),
    ):
        bins = pack(lengths, PL)
        n_batch = len(bins) // MBS
        if not n_batch:
            print(f'  {name}: too few bins ({len(bins)})')
            continue
        target = MBS * PL
        peak, over_batches = 0, 0
        for s in range(0, n_batch * MBS, MBS):
            group = bins[s:s + MBS]
            L = real_len(make_template(cp), group, cp)
            peak = max(peak, L)
            if L > target:
                over_batches += 1
        flag = 'UNDER-RESERVED' if peak > target else 'ok'
        print(f'  {name}: target={target} real_peak={peak} '
              f'{"+" + str(peak - target) if peak > target else ""} '
              f'({(peak/target):.4f}x) batches_over={over_batches}/{n_batch}  {flag}')
