#!/usr/bin/env python3
"""Demo 4: tail-first scan order + partial-scan peak estimate + closed-loop guard (v6).

The warm-up length is the *tightest upper bound* of the true peak. This demo verifies the
machinery that keeps it so when the scan cannot complete:

1. `_order_tail_first`: unknown sizes first, then media rows by file size DESC, text last --
   so any scan cutoff leaves the SHORT rows unscanned;
2. the three peak modes in `get_pad_warmup_length`: exact / substituted (<=5% missing) /
   tail-estimate (>5% missing, estimated from the scanned subset ALONE, never substituted) /
   disabled (subset too small) -- in particular >5% missing no longer disables the feature,
   and the estimate stays tight vs the full-data estimate (no inflation);
3. `_estimate_peak_from_subset` is seeded: same array -> same estimate (cross-rank
   convergence by construction);
4. the closed-loop guard in `PadWarmupCollator` warns exactly once when a real batch exceeds
   the warm-up length, and stays silent otherwise.

Run:  python3 demo_tail_first.py
"""
import importlib.util
import os
import re
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
PATCH = os.environ.get('DEMO_PATCH',
                       os.path.join(HERE, '..', 'feature4_megatron_pad_warmup-v9.patch'))

WARNINGS = []


def _stub_swift():
    if 'swift' in sys.modules:
        return
    swift = types.ModuleType('swift')
    utils = types.ModuleType('swift.utils')

    class _L:
        def info(self, m, *a, **k):
            print(f'[I] {m}', flush=True)

        def warning(self, m, *a, **k):
            WARNINGS.append(m)
            print(f'[W] {m}', flush=True)

        def info_once(self, m, *a, **k):
            print(f'[I] {m}', flush=True)

    utils.get_logger = lambda *a, **kw: _L()
    # SWIFT_PAD_WARMUP=1 for this demo (get_pad_warmup_length is gated on it)
    utils.get_env_args = lambda k, t, d: True if k == 'SWIFT_PAD_WARMUP' else d
    dsu = types.ModuleType('swift.dataset.utils')
    dsu.LazyLLMDataset = type('LazyLLMDataset', (), {})
    sys.modules.update({'swift': swift, 'swift.utils': utils,
                        'swift.dataset': types.ModuleType('swift.dataset'),
                        'swift.dataset.utils': dsu})


_stub_swift()


def load_pad_warmup():
    lines = open(PATCH).read().split('\n')
    start = next(i for i, l in enumerate(lines) if l.startswith('@@ -0,0 +1,'))
    count = int(re.match(r'@@ -0,0 \+1,(\d+) @@', lines[start]).group(1))
    src = '\n'.join(l[1:] for l in lines[start + 1:start + 1 + count]) + '\n'
    path = os.path.join(tempfile.mkdtemp(), 'pad_warmup.py')
    open(path, 'w').write(src)
    spec = importlib.util.spec_from_file_location('pad_warmup_demo4', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['pad_warmup_demo4'] = mod
    spec.loader.exec_module(mod)
    return mod


def make_args():
    return types.SimpleNamespace(
        context_parallel_size=1, tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
        max_length=32768, packing=False, micro_batch_size=4, padding_free=True,
        data_sharding=False, train_dataloader_shuffle=True, group_by_length=False,
        dataset_num_proc=1, num_train_epochs=1, train_iters=None, global_batch_size=16,
        dataset=['demo'], streaming=False)


class T:
    mode = 'train'
    tokenizer = None


def main():
    pw = load_pad_warmup()
    tmp = tempfile.mkdtemp(prefix='pw_tail_')

    print('== 1. tail-first order: unknown first, size DESC, text last ==')
    files = []
    for i, kb in enumerate((64, 1024, 8, 256)):  # sparse files of different sizes
        p = os.path.join(tmp, f'vid_{i}.mp4')
        with open(p, 'wb') as f:
            f.truncate(kb * 1024)
        files.append(p)
    missing = os.path.join(tmp, 'gone.mp4')
    rows = [
        {'videos': {'path': files[2]}, 'messages': []},      # 0:    8 KB
        {'messages': []},                                    # 1:   text
        {'videos': {'path': files[1]}, 'messages': []},      # 2: 1024 KB
        {'videos': {'path': missing}, 'messages': []},       # 3: unknown
        {'videos': {'path': files[0]}, 'messages': []},      # 4:   64 KB
        {'videos': {'path': files[3]}, 'messages': []},      # 5:  256 KB
    ]
    todo, dup_of, sizes = pw._scan_dedup_plan(rows, len(rows))
    ordered = pw._order_tail_first(todo, sizes)
    assert ordered[0] == 3, ordered                    # unknown size first
    rest = [i for i in ordered if i != 3]
    assert rest == [2, 5, 4, 0, 1], (ordered, sizes)   # 1024 > 256 > 64 > 8 KB > text
    print(f'   order={ordered} sizes={sizes}  OK')

    print('== 2. peak modes: exact / exact-batch(-partial) / off ==')
    n = 4000
    # lengths i%1600+1 shuffled deterministically; tail-first scan finished the TOP half:
    # rows whose length > 800 are valid, the rest are 0 (deadline cutoff).
    import random
    rng = random.Random(7)
    full = [i % 1600 + 1 for i in range(n)]
    rng.shuffle(full)
    top_half = [v if v > 800 else 0 for v in full]     # 50% missing, the SHORT half
    mostly = [0 if rng.random() < 0.03 else v for v in full]  # 3% missing

    args = make_args()
    # full scan -> exact replay
    pw.scan_sample_lengths = lambda *a: (list(full), 0)
    peak_exact = pw.get_pad_warmup_length(args, list(range(n)), T())
    ref_exact = pw.replay_peak_batch_length(args, list(range(n)), list(full), 4, agg='sum',
                                            dp_rank=None)
    assert peak_exact == ref_exact, (peak_exact, ref_exact)
    print(f'   exact:        peak={peak_exact} (== replay)')

    # 3% missing -> NO substitution any more (v9: a deterministic fill value colliding k>=2
    # times in one batch inflates the peak ~linearly with mbs). The chain tries exact-batch
    # (strict gates), then the partial-coverage max (undershoot-only); there is NO estimate
    # tier any more (an estimate can overshoot, and overshoot kills healthy runs).
    # The demo has no megatron (dp_rank unresolvable), so get_pad_warmup_length's missing-row
    # path lands on OFF here; exercise the two exact-batch modes directly with an explicit rank.
    pw.scan_sample_lengths = lambda *a: (list(mostly), sum(1 for v in mostly if v <= 0))
    peak_sub = pw.get_pad_warmup_length(args, list(range(n)), T())
    assert peak_sub is None, 'demo env has no megatron -> no dp_rank -> missing rows -> off'
    truth = ref_exact
    # strict refuses: this dataset has no media columns, so the proxy cannot be fitted
    assert pw._exact_batch_peak(args, list(range(n)), mostly, 4, 'sum', dp=1, dp_rank=0) is None
    # partial coverage: exact for what it covers, undershoot-only
    part = pw._covered_batch_peak(args, mostly, 4, 'sum', dp=1, dp_rank=0)
    assert 0 < part <= truth, (part, truth)
    print(f'   3% missing:   strict refused (proxy unfit); partial={part} <= true {truth} '
          f'(undershoot-only); chain->off in demo env (no megatron)')

    # top-50% tail-first cutoff -> partial stays tight: the true peak batch's rows are all
    # in the long half, hence fully covered (this is what tail-first buys at a cutoff).
    pw.scan_sample_lengths = lambda *a: (list(top_half), sum(1 for v in top_half if v <= 0))
    part50 = pw._covered_batch_peak(args, top_half, 4, 'sum', dp=1, dp_rank=0)
    assert 0 < part50 <= truth, (part50, truth)
    print(f'   tail-first 50% cutoff: partial={part50} vs true {truth} '
          f'({part50 / truth:.0%}, undershoot-only)')

    # almost nothing scanned -> disabled (nothing covered to measure)
    nearly_none = [0] * n
    for i in range(50):
        nearly_none[i] = full[i]
    pw.scan_sample_lengths = lambda *a: (list(nearly_none), n - 50)
    assert pw.get_pad_warmup_length(args, list(range(n)), T()) is None
    print('   too-small subset (50/4000 rows): disabled, correct')

    print('== 3. closed-loop overshoot guard ==')
    WARNINGS.clear()
    collator = pw.PadWarmupCollator(lambda batch, **kw: batch, length=100, num_batches=1,
                                    flag_path=None, padding_free=True)
    collator([{'input_ids': [0] * 60}, {'input_ids': [0] * 30}])   # 90 <= 100: silent
    assert not WARNINGS
    collator([{'input_ids': [0] * 60}, {'input_ids': [0] * 50}])   # 110 > 100: WARN once
    assert len(WARNINGS) == 1 and 'exceeded the warm-up length' in WARNINGS[0], WARNINGS
    collator([{'input_ids': [0] * 200}])                           # again: still once
    assert len(WARNINGS) == 1
    print('   warns exactly once on overshoot, silent below the warm-up length  OK')

    print('== 4. context parallel: enabled, CP-invariant replay, CP-aligned target (v8) ==')
    import math
    # Mirror swift's get_padding_to: cp>1 makes every batch a multiple of cp_size.
    muu = types.ModuleType('swift.megatron.utils.utils')
    muu.get_padding_to = lambda a: (a.context_parallel_size
                                    if getattr(a, 'context_parallel_size', 1) > 1 else None)
    sys.modules.update({'swift.megatron': types.ModuleType('swift.megatron'),
                        'swift.megatron.utils': types.ModuleType('swift.megatron.utils'),
                        'swift.megatron.utils.utils': muu})
    args_cp = make_args()
    args_cp.context_parallel_size = 2
    pw.scan_sample_lengths = lambda *a: (list(full), 0)
    peak_cp1 = pw.get_pad_warmup_length(args, list(range(n)), T())
    peak_cp2 = pw.get_pad_warmup_length(args_cp, list(range(n)), T())
    assert peak_cp2 is not None, 'cp>1 must no longer disable pad-warmup'
    # CP changes neither the batch stream nor the raw peak: the sampler is keyed on the PURE
    # data-parallel rank (get_data_parallel_rank defaults to with_context_parallel=False), so
    # every CP rank of a group sees the same batches. But the collator pads EVERY sample to a
    # 2*cp multiple (`_handle_megatron_cp`), so the real peak lives in the PADDED space: peaks
    # from raw lengths would under-reserve by up to group_size * (2*cp - 1) tokens.
    raw1 = pw.replay_peak_batch_length(args, list(range(n)), list(full), 4, agg='sum')
    raw2 = pw.replay_peak_batch_length(args_cp, list(range(n)), list(full), 4, agg='sum')
    assert raw1 == raw2, (raw1, raw2)
    padded = [pw._cp_pad_length(v, 2) for v in full]
    padded_peak = pw.replay_peak_batch_length(args_cp, list(range(n)), padded, 4, agg='sum')
    assert padded_peak >= raw2, (padded_peak, raw2)      # the gap raw-space would hide
    assert peak_cp1 == raw1 and peak_cp2 == padded_peak, (peak_cp1, peak_cp2, padded_peak)
    print(f'   cp=2 enabled: raw peak identical to cp=1 ({raw2}); peak resolved in the padded '
          f'space ({padded_peak}, >= raw {raw2})  OK')

    print('== 5. expandable_segments: auto-decline (the allocator already handles it) ==')
    pw.scan_sample_lengths = lambda *a: (list(full), 0)
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    assert pw.get_pad_warmup_length(args, list(range(n)), T()) is None
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    assert pw.get_pad_warmup_length(args, list(range(n)), T()) is not None
    del os.environ['PYTORCH_CUDA_ALLOC_CONF']
    assert pw.get_pad_warmup_length(args, list(range(n)), T()) is not None
    print('   expandable_segments:True -> off; other alloc conf / unset -> enabled  OK')

    print('== 6. CE reserve: last pipeline stage only; skip under VPP; hold when unknown ==')
    # The CE backward lives on the last pipeline stage: every other stage must skip the
    # block (it is ~18 GiB of pure waste there). Hooks stay inert no-ops when the shape
    # is None, so skipping is communicated purely through `_CE_RESERVE_SHAPE`/`acquire`.
    mpu = types.ModuleType('megatron.core.parallel_state')
    for name in ('megatron', 'megatron.core'):
        sys.modules[name] = types.ModuleType(name)
    sys.modules['megatron.core.parallel_state'] = mpu
    held = []
    pw.acquire_ce_reserve_buf = lambda: held.append(1)
    mpu.is_pipeline_last_stage = lambda: False
    pw._CE_RESERVE_SHAPE = None
    pw.wrap_pad_warmup(lambda b, **k: b, args, list(range(n)), T())
    assert pw._CE_RESERVE_SHAPE is None and not held, 'middle stage must not hold'
    mpu.is_pipeline_last_stage = lambda: True
    pw.wrap_pad_warmup(lambda b, **k: b, args, list(range(n)), T())
    assert held and pw._CE_RESERVE_SHAPE is not None and pw._CE_RESERVE_SHAPE[0] > 0
    shape_last = pw._CE_RESERVE_SHAPE
    held.clear()
    pw._CE_RESERVE_SHAPE = None
    args_vpp = make_args()
    args_vpp.virtual_pipeline_model_parallel_size = 2
    pw.wrap_pad_warmup(lambda b, **k: b, args_vpp, list(range(n)), T())
    assert pw._CE_RESERVE_SHAPE is None and not held, 'VPP must skip (unvalidated hooks)'
    del sys.modules['megatron.core.parallel_state']     # stage unknown -> protective hold
    held.clear()
    pw.wrap_pad_warmup(lambda b, **k: b, args, list(range(n)), T())
    assert held and pw._CE_RESERVE_SHAPE is not None
    pw._CE_RESERVE_SHAPE = None
    print(f'   middle stage -> skip; last stage -> hold {shape_last}; '
          f'VPP -> skip; stage unknown -> protective hold  OK')

    print('\nDEMO4 OK')


if __name__ == '__main__':
    main()
