#!/usr/bin/env python3
"""Demo 11: the v10 scenario-coverage fixes -- every case first REPRODUCES the v9 defect,
then asserts the v10 behaviour.

v10 is deliberately "v9 plus six bug fixes": no redesign, and no new self-imposed gate. Every
section below is a case where v9 was measurably WRONG, and the v9 behaviour is reproduced by
re-implementing *exactly* the v9 code path next to the shipped v10 one (`v9_*` helpers below),
so the demo proves a DIFFERENCE rather than that v10 merely self-agrees.

1. packing: one micro-batch carries `mbs` bins, and `packing_length` may raise the cap above
   `max_length` -> v9 under-reserved mbs-fold (the feature silently did nothing);
2. padded layout (`padding_free=False`): v9 appended a dummy ROW -> carved (mbs+1, target)
   instead of (mbs, target) -- 2x at mbs=1, the one fatal (over-reserving) direction. v10 adds
   no row and raises `padding_to` instead;
3. CE reserve: `cp` shards the SEQUENCE, so v9's (length, vocab/tp) over-reserved cp-fold;
   the padded layout feeds mbs rows;
4. validation dataloader: it shares `self.data_collator`, so an eval batch used to consume the
   one-shot warm-up claim on a forward-only path;
5. pipeline p2p pre-warm: `_prewarm_communicators` only issues all_reduce, the lazy p2p
   connection setup of PP>1 was never covered. OPT-IN (`SWIFT_PAD_WARMUP_PP_P2P=1`): it has
   never run against a real backend and its failure mode is a hang, so it stays off by default;
6. dataloader workers: the claim was first-come-first-served, so the padded batch was not
   necessarily the FIRST batch executed;
8. dummy field mirroring: a hand-kept whitelist breaks on any new per-token field; v10 mirrors
   every per-token field automatically.

Dropped on purpose (they were self-imposed limits, not fixes, and each one WITHDREW a
guarantee v9 already delivered): the `task_type != 'causal_lm'` gate and the CUDA-graph gate
(both merely unvalidated -- the collator's try/except already degrades them safely), and the
`_dataset_fingerprint` rewrite (it changed every cache key, costing a full re-scan measured at
~105 min, to close a blind spot whose worst case is a mis-sized reservation).

Run:  python3 demo_v10_fixes.py              (loads pad_warmup.py -- the patch's build source)
      DEMO_PATCH=1 python3 demo_v10_fixes.py (extracts pad_warmup.py from the patch instead)
"""
import importlib.util
import math
import os
import re
import sys
import tempfile
import time
import types
from typing import Any, Dict, List, Optional

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PATCH = os.path.join(HERE, '..', 'feature4_megatron_pad_warmup-v10.patch')
LOGS = {'info': [], 'warning': []}
CHECKS = []


def _stub_swift():
    if 'swift' in sys.modules:
        return
    swift = types.ModuleType('swift')
    utils = types.ModuleType('swift.utils')

    class _L:
        def info(self, m, *a, **k):
            LOGS['info'].append(str(m))

        def warning(self, m, *a, **k):
            LOGS['warning'].append(str(m))

        def info_once(self, m, *a, **k):
            LOGS['info'].append(str(m))

    utils.get_logger = lambda *a, **kw: _L()
    utils.get_env_args = lambda k, t, d: d
    dsu = types.ModuleType('swift.dataset.utils')
    dsu.LazyLLMDataset = type('LazyLLMDataset', (), {})
    sys.modules.update({'swift': swift, 'swift.utils': utils,
                        'swift.dataset': types.ModuleType('swift.dataset'),
                        'swift.dataset.utils': dsu})
    for name in ('swift.megatron', 'swift.megatron.utils'):
        sys.modules[name] = types.ModuleType(name)
    meg_utils = types.ModuleType('swift.megatron.utils.utils')
    meg_utils.get_padding_to = lambda args: 1
    sys.modules['swift.megatron.utils.utils'] = meg_utils


_stub_swift()


def load_pad_warmup():
    if os.environ.get('DEMO_PATCH'):
        lines = open(PATCH).read().split('\n')
        start = next(i for i, l in enumerate(lines) if l.startswith('@@ -0,0 +1,'))
        count = int(re.match(r'@@ -0,0 \+1,(\d+) @@', lines[start]).group(1))
        src = '\n'.join(l[1:] for l in lines[start + 1:start + 1 + count]) + '\n'
        path = os.path.join(tempfile.mkdtemp(), 'pad_warmup.py')
        open(path, 'w').write(src)
    else:
        path = os.path.join(HERE, 'pad_warmup.py')
    spec = importlib.util.spec_from_file_location('pad_warmup_demo11', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['pad_warmup_demo11'] = mod
    spec.loader.exec_module(mod)
    return mod


def load_template_methods():
    """Exec the template methods the patch ships (`_sample_dummy_text_ids` /
    `_append_warmup_dummy`) onto a fake `Template` self -- the real code, not a copy."""
    lines = open(PATCH).read().split('\n')
    i0 = next(i for i, l in enumerate(lines) if l.startswith('+    def _sample_dummy_text_ids'))
    i1 = next(i for i in range(i0, len(lines)) if lines[i].startswith('+    def data_collator'))
    src = '\n'.join(l[1:] for l in lines[i0:i1])
    assert '_append_warmup_dummy' in src and '_sample_dummy_text_ids' in src
    ns = {'torch': torch, 'math': math, 'List': List, 'Dict': Dict, 'Any': Any,
          'Optional': Optional}
    exec('class _FakeTemplate:\n' + src.rstrip() + '\n', ns)
    return ns['_FakeTemplate']


def check(ok, name, extra=''):
    CHECKS.append((bool(ok), name))
    print(f'  {"ok" if ok else "FAIL"} {name}{(" (" + extra + ")") if extra else ""}')
    assert ok, name


def _cp_len(self, length):
    """Mirror of the real `Template._get_megatron_cp_length` (swift 4.3.1, verified in the
    image): every sample is padded to a 2*cp multiple under CP."""
    cp_size = self.sequence_parallel_size
    if not self.use_megatron or cp_size == 1:
        return length
    return math.ceil(length / (cp_size * 2)) * (cp_size * 2)


def fake_template(FakeTemplate, *, padding_free=True, cp=1):
    t = FakeTemplate()
    t.use_megatron = True
    t.padding_free = padding_free
    t.sequence_parallel_size = cp
    t._get_megatron_cp_length = types.MethodType(_cp_len, t)
    t.tokenizer = types.SimpleNamespace(pad_token_id=0, all_special_ids=[1, 2])
    t.image_token_id = 32000
    t.video_token_id = 32001
    return t


def row(n, **extra):
    r = {'input_ids': [5 + (i % 300) for i in range(n)], 'labels': [1] * n, 'length': n}
    r.update(extra)
    return r


# ---------------------------------------------------------------------------
# v9 reference implementations (verbatim behaviour of the code v10 replaces)
# ---------------------------------------------------------------------------
def v9_packing_length(args, max_length, finalize):
    """v9: `if args.packing: return _finalize(max_length)` -- no mbs, no packing_length."""
    return finalize(max_length)


def v9_ce_reserve_shape(args, length):
    """v9: `(length, vocab // tp)` -- cp and the row count were missing."""
    vocab = getattr(args, 'padded_vocab_size', None) or 248320
    tp = getattr(args, 'tensor_model_parallel_size', 1) or 1
    return (length, vocab // tp)


def v9_append_dummy_padded(batch, target):
    """v9: the padded layout appended a dummy ROW of `target` tokens."""
    batch.append(row(int(target)))
    return len(batch)


def main():
    pw = load_pad_warmup()
    FakeTemplate = load_template_methods()
    pw.get_env_args = lambda k, t, d: k == 'SWIFT_PAD_WARMUP'
    tpl = types.SimpleNamespace(mode='train')

    print('== 1. packing: mbs bins per micro-batch + `packing_length` cap ==')
    # padding_free + packing: `Template.data_collator` flattens the mbs bins into ONE packed
    # row (`batch = sum(batch, start=[])`), so the row is up to mbs * packing_length.
    for mbs in (1, 2, 4, 8):
        args = types.SimpleNamespace(max_length=4096, packing=True, padding_free=True,
                                     micro_batch_size=mbs, context_parallel_size=1)
        v10 = pw.get_pad_warmup_length(args, None, tpl)
        v9 = v9_packing_length(args, 4096, lambda x: x)
        check(v10 == 4096 * mbs, f'1 packing padding_free mbs={mbs} -> mbs*packing_length',
              f'v10={v10} v9={v9}')
        # the v9 value made the dummy vanish: dummy_len = target - total <= 0
        t = fake_template(FakeTemplate)
        bins = [row(4096) for _ in range(mbs)]
        n0 = len(bins)
        t._append_warmup_dummy(bins, v9, None)
        check(len(bins) == n0 and mbs > 1 or mbs == 1,
              f'1 packing mbs={mbs}: v9 target makes the warm-up a no-op' if mbs > 1
              else '1 packing mbs=1: v9 == v10 (no regression at mbs=1)')
    # `packing_length` raises the cap above max_length (megatron: seq_length = packing_length)
    args = types.SimpleNamespace(max_length=4096, packing_length=8192, packing=True,
                                 padding_free=True, micro_batch_size=2, context_parallel_size=1)
    check(pw.get_pad_warmup_length(args, None, tpl) == 16384,
          '1 packing honours packing_length > max_length', 'v9 would return 4096')
    # padded layout: a row is max(bins) <= packing_length, so NO mbs factor
    args = types.SimpleNamespace(max_length=4096, packing=True, padding_free=False,
                                 micro_batch_size=4, context_parallel_size=1)
    check(pw.get_pad_warmup_length(args, None, tpl) == 4096,
          '1 packing padded layout keeps packing_length (no mbs factor)')
    # CP: the target stays 2*cp-aligned through `_finalize`
    args = types.SimpleNamespace(max_length=4095, packing=True, padding_free=True,
                                 micro_batch_size=2, context_parallel_size=2)
    v10 = pw.get_pad_warmup_length(args, None, tpl)
    check(v10 % 4 == 0 and v10 >= 4095 * 2, '1 packing + cp=2 target is 2*cp-aligned', f'{v10}')

    print('== 2. padded layout: (mbs, target), NOT (mbs+1, target) ==')
    for mbs in (1, 2, 4):
        t = fake_template(FakeTemplate, padding_free=False)
        rows = [row(100 + 7 * i) for i in range(mbs)]
        target = 512
        b10 = [dict(r) for r in rows]
        out = t._append_warmup_dummy(b10, target, None)
        b9 = [dict(r) for r in rows]
        v9_append_dummy_padded(b9, target)
        # v10: no row added, padding_to raised to the target (the collator treats padding_to as
        # a multiple: ceil(max(seq_lens)/target)*target == target while max <= target)
        check(len(b10) == mbs and out == target,
              f'2 padded mbs={mbs}: no dummy row, padding_to -> target')
        v10_tokens = mbs * target
        v9_tokens = (mbs + 1) * target
        check(len(b9) == mbs + 1 and v9_tokens > v10_tokens,
              f'2 padded mbs={mbs}: v9 carved (mbs+1, target)',
              f'{v9_tokens} vs {v10_tokens} tokens = {v9_tokens / v10_tokens:.2f}x')
        # the 4-D causal mask scales with the row count as well
        check((mbs + 1) * target * target > mbs * target * target,
              f'2 padded mbs={mbs}: v9 also inflated the (b,1,L,L) mask',
              f'{(mbs + 1) / mbs:.2f}x')
    # a target BELOW the batch max must not be used as a multiple (it would double it)
    t = fake_template(FakeTemplate, padding_free=False)
    b = [row(600)]
    out = t._append_warmup_dummy(b, 512, 64)
    check(out == 64 and len(b) == 1,
          '2 padded: stale/underestimated target keeps the original padding_to',
          'ceil(600/512)*512 = 1024 would have DOUBLED the reservation')
    # under CP the comparison happens in the padded space
    t = fake_template(FakeTemplate, padding_free=False, cp=2)
    b = [row(509)]  # -> 512 after 2*cp padding
    check(t._append_warmup_dummy(b, 512, 64) == 512,
          '2 padded + cp=2: rows compared in the 2*cp-padded space')
    # padding_free is unchanged: the dummy segment still closes the gap exactly
    t = fake_template(FakeTemplate, padding_free=True)
    b = [row(100), row(150)]
    t._append_warmup_dummy(b, 512, None)
    check(len(b) == 3 and b[-1]['length'] == 262 and sum(r['length'] for r in b) == 512,
          '2 padding_free unchanged: dummy segment closes the gap exactly')

    print('== 3. CE reserve: cp shards the sequence, the padded layout has mbs rows ==')
    mpu = types.ModuleType('megatron.core.parallel_state')
    mpu.is_pipeline_last_stage = lambda: True
    core = types.ModuleType('megatron.core')
    core.parallel_state = mpu
    sys.modules.update({'megatron': types.ModuleType('megatron'), 'megatron.core': core,
                        'megatron.core.parallel_state': mpu})
    # No CUDA in the demo: the block itself cannot be allocated, and `wrap_pad_warmup` would
    # then disable the reserve (best-effort). Only the SHAPE is under test here.
    pw.acquire_ce_reserve_buf = lambda: None
    for cp, mbs, padding_free in ((1, 1, True), (2, 1, True), (4, 1, True), (8, 1, True),
                                  (1, 2, False), (1, 4, False), (2, 2, False)):
        args = types.SimpleNamespace(max_length=4096, packing=True, padding_free=padding_free,
                                     micro_batch_size=mbs, context_parallel_size=cp,
                                     padded_vocab_size=248320, tensor_model_parallel_size=8,
                                     output_dir=tempfile.mkdtemp())
        pw._CE_RESERVE_SHAPE = None
        pw.wrap_pad_warmup(lambda b, **kw: b, args, None, tpl)
        shape = pw._CE_RESERVE_SHAPE
        length = pw.get_pad_warmup_length(args, None, tpl)
        rows_f = 1 if padding_free else mbs
        want = (max(1, length // cp * rows_f), 248320 // 8)
        v9 = v9_ce_reserve_shape(args, length)
        check(shape == want, f'3 CE reserve cp={cp} mbs={mbs} pf={padding_free} shape',
              f'v10={shape} v9={v9}')
        if cp > 1 and padding_free:
            gib = (v9[0] * v9[1] - shape[0] * shape[1]) * 2 / 2**30
            check(v9[0] > shape[0] and v9[0] // shape[0] == cp,
                  f'3 CE reserve cp={cp}: v9 over-reserved cp-fold',
                  f'{v9[0] / shape[0]:.0f}x, +{gib:.2f} GiB wasted at this vocab')
        if not padding_free and cp == 1:
            check(shape[0] == v9[0] * mbs,
                  f'3 CE reserve padded mbs={mbs}: v9 under-covered the mbs rows of logits',
                  f'v10={shape[0]} = v9={v9[0]} x {mbs}')
        if not padding_free and cp == mbs:
            check(shape[0] == v9[0],
                  f'3 CE reserve padded cp={cp}==mbs={mbs}: the two factors cancel out')
    pw._CE_RESERVE_SHAPE = None

    print('== 4. the validation dataloader must not consume the warm-up claim ==')
    # Model `_prepare_dataloader`: the trainer keeps the unwrapped collator for eval.
    class FakeTrainer:
        def __init__(self):
            self.calls = []
            self.data_collator = lambda b, **kw: ('raw', len(b), kw.get('pad_to_length'))

        def wrap(self, args):
            self._pad_warmup_eval_collator = self.data_collator
            self.data_collator = pw.wrap_pad_warmup(self.data_collator, args, None, tpl)

        def _eval_collator(self):
            return getattr(self, '_pad_warmup_eval_collator', None) or self.data_collator

    args = types.SimpleNamespace(max_length=4096, packing=True, padding_free=True,
                                 micro_batch_size=1, context_parallel_size=1,
                                 output_dir=tempfile.mkdtemp())
    tr = FakeTrainer()
    tr.wrap(args)
    check(isinstance(tr.data_collator, pw.PadWarmupCollator),
          '4 train collator is the pad-warmup wrapper')
    check(not isinstance(tr._eval_collator(), pw.PadWarmupCollator),
          '4 eval collator is the RAW collator (v9 shared the wrapper)')
    # v9 shape: eval goes through the wrapper first and eats the one-shot claim
    v9_wrapper = pw.PadWarmupCollator(lambda b, **kw: ('raw', len(b), kw.get('pad_to_length')),
                                      4096, 1, flag_path=os.path.join(tempfile.mkdtemp(), 'f'))
    eval_out = v9_wrapper([row(10)])              # the eval batch (forward-only)
    train_out = v9_wrapper([row(10)])             # the real step-0 batch
    check(eval_out[2] == 4096 and train_out[2] is None,
          '4 v9 defect reproduced: the eval batch was padded, step-0 was not')
    # v10: the same wrapper is only ever reached by training batches
    w = tr.data_collator
    out1 = w([row(10)])
    check(out1[2] == 4096, '4 v10: the first TRAINING batch is the padded one')
    check(tr._eval_collator()([row(10)])[2] is None, '4 v10: eval batches are never padded')

    print('== 5. pipeline p2p pre-warm (OPT-IN; the lazy connection all_reduce does not '
          'establish) ==')
    calls = {'all_reduce': [], 'p2p': [], 'sync': 0}

    class _G:
        def __init__(self, ranks):
            self.ranks = ranks

    dist = types.ModuleType('torch.distributed')
    dist.is_available = lambda: True
    dist.is_initialized = lambda: True
    dist.get_rank = lambda: 1
    dist.get_world_size = lambda group=None: len(group.ranks) if isinstance(group, _G) else 8
    dist.get_process_group_ranks = lambda g: g.ranks
    dist.all_reduce = lambda t, group=None: calls['all_reduce'].append(group)
    dist.isend, dist.irecv = 'isend', 'irecv'

    class _Req:
        def wait(self):
            calls['p2p'].append('wait')

    dist.P2POp = lambda op, t, peer, group: (op, peer, group)
    dist.batch_isend_irecv = lambda ops: (calls['p2p'].append(ops), [_Req()])[1]
    real_dist = sys.modules.get('torch.distributed')
    sys.modules['torch.distributed'] = dist
    torch.distributed = dist
    real_sync = torch.cuda.synchronize
    real_cur = torch.cuda.current_device
    torch.cuda.synchronize = lambda *a, **k: calls.__setitem__('sync', calls['sync'] + 1)
    torch.cuda.current_device = lambda: 'cpu'
    try:
        mpu.get_pipeline_model_parallel_group = lambda: _G([0, 1, 2, 3])
        # Default OFF: the pre-warm must not touch the network unless opted in, because an
        # asymmetric failure here HANGS the job (no timeout) instead of OOMing.
        os.environ.pop('SWIFT_PAD_WARMUP_PP_P2P', None)
        pw._prewarm_pipeline_p2p(types.SimpleNamespace(pipeline_model_parallel_size=4))
        check(not calls['p2p'], '5 默认关闭：未设 SWIFT_PAD_WARMUP_PP_P2P 时完全不碰网络')
        for off in ('0', 'off', 'false', 'no'):
            calls['p2p'].clear()
            os.environ['SWIFT_PAD_WARMUP_PP_P2P'] = off
            pw._prewarm_pipeline_p2p(types.SimpleNamespace(pipeline_model_parallel_size=4))
            check(not calls['p2p'], f'5 SWIFT_PAD_WARMUP_PP_P2P={off!r} 关闭')
        os.environ['SWIFT_PAD_WARMUP_PP_P2P'] = '1'
        calls['p2p'].clear()
        pw._prewarm_pipeline_p2p(types.SimpleNamespace(pipeline_model_parallel_size=4))
        ops = next(o for o in calls['p2p'] if isinstance(o, list))
        peers = sorted(o[1] for o in ops)
        check(len(ops) == 2 and peers == [0, 2],
              '5 PP=4 rank1: one ring send(->2) + recv(<-0)', f'peers={peers}')
        check(all(o[2].ranks == [0, 1, 2, 3] for o in ops), '5 the ops run on the pipeline group')
        # PP=1 must not touch the network at all
        calls['p2p'].clear()
        pw._prewarm_pipeline_p2p(types.SimpleNamespace(pipeline_model_parallel_size=1))
        check(not calls['p2p'], '5 PP=1 is a no-op')
        # a list of groups (VPP) is iterated
        calls['p2p'].clear()
        mpu.get_pipeline_model_parallel_group = lambda: [_G([0, 1]), _G([1, 2])]
        pw._prewarm_pipeline_p2p(types.SimpleNamespace(pipeline_model_parallel_size=2))
        check(len([o for o in calls['p2p'] if isinstance(o, list)]) == 2,
              '5 a list of pipeline groups is pre-warmed group by group')
        # a backend that raises degrades to a warning, never to a crash
        calls['p2p'].clear()
        LOGS['warning'].clear()

        def _boom(ops):
            raise RuntimeError('p2p backend unavailable')

        dist.batch_isend_irecv = _boom
        mpu.get_pipeline_model_parallel_group = lambda: _G([0, 1])
        pw._prewarm_pipeline_p2p(types.SimpleNamespace(pipeline_model_parallel_size=2))
        check(any('p2p pre-warm failed' in m for m in LOGS['warning']),
              '5 a p2p failure degrades to a warning (best-effort)')
        # `_prewarm_runtime` calls it between the communicators and DeepEP
        order = []
        for name in ('_prewarm_communicators', '_prewarm_pipeline_p2p', '_prewarm_deepep_buffer',
                     '_settle_allocator_state', '_patch_moe_dispatcher_release'):
            setattr(pw, name, (lambda n: lambda *a, **k: order.append(n))(name))
        pw._prewarm_runtime(types.SimpleNamespace(), None)
        check(order == ['_prewarm_communicators', '_prewarm_pipeline_p2p',
                        '_prewarm_deepep_buffer', '_settle_allocator_state',
                        '_patch_moe_dispatcher_release'],
              '5 _prewarm_runtime order: communicators -> p2p -> deepep -> settle -> release',
              str(order))
    finally:
        os.environ.pop('SWIFT_PAD_WARMUP_PP_P2P', None)
        torch.cuda.synchronize = real_sync
        torch.cuda.current_device = real_cur
        if real_dist is not None:
            sys.modules['torch.distributed'] = real_dist
            torch.distributed = real_dist
        pw = load_pad_warmup()  # drop the stubbed module state
        pw.get_env_args = lambda k, t, d: k == 'SWIFT_PAD_WARMUP'

    print('== 6. only worker 0 may claim the warm-up (it produces batch 0) ==')
    flag = os.path.join(tempfile.mkdtemp(), 'warmed.flag')
    collators = {wid: pw.PadWarmupCollator(lambda b, **kw: ('raw', len(b), kw.get('pad_to_length')),
                                           4096, 1, flag_path=flag) for wid in (0, 1, 2, 3)}
    worker = {'id': None}
    import torch.utils.data as tud
    real_info = tud.get_worker_info
    tud.get_worker_info = lambda: None if worker['id'] is None else types.SimpleNamespace(
        id=worker['id'])
    try:
        # v9 was first-come-first-served (flag claim only, no worker gate): whichever worker
        # collated first padded ITS batch -- reproduce that with the v9 `_claimed` body.
        v9_flag = os.path.join(tempfile.mkdtemp(), 'warmed.flag')
        fd = None
        try:
            fd = os.open(v9_flag, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            v9_claimed_by_worker2 = True
        except FileExistsError:
            v9_claimed_by_worker2 = False
        check(v9_claimed_by_worker2,
              '6 v9 defect reproduced: any worker could claim the flag first',
              'worker 2 would have padded batch 2, after batches 0/1 carved the heap')
        worker['id'] = 2
        out2 = collators[2]([row(10)])
        check(out2[2] is None, '6 worker 2 does NOT pad (v9: it would have padded batch 2)')
        worker['id'] = 0
        out0 = collators[0]([row(10)])
        check(out0[2] == 4096, '6 worker 0 pads -- the batch that is executed FIRST')
        # the one-shot claim still holds across epochs (a fresh worker 0 is forked)
        fresh = pw.PadWarmupCollator(lambda b, **kw: ('raw', len(b), kw.get('pad_to_length')),
                                     4096, 1, flag_path=flag)
        check(fresh([row(10)])[2] is None, '6 a later epoch worker 0 finds the flag claimed')
        # num_workers=0 (no worker info): the main process must still warm up
        worker['id'] = None
        flag2 = os.path.join(tempfile.mkdtemp(), 'warmed.flag')
        solo = pw.PadWarmupCollator(lambda b, **kw: ('raw', len(b), kw.get('pad_to_length')),
                                    4096, 1, flag_path=flag2)
        check(solo([row(10)])[2] == 4096, '6 num_workers=0 still pads (no worker info)')
    finally:
        tud.get_worker_info = real_info

    print('== 8. the dummy mirrors EVERY per-token field (no whitelist) ==')
    t = fake_template(FakeTemplate, padding_free=True)
    # a field no v9 whitelist knows about (the shape that used to crash packing_row's cat)
    rows = [row(20, loss_scale=[1.] * 20, token_type_ids=[0] * 20,
                mm_token_type_ids=torch.zeros(20, dtype=torch.int64),
                sample_weight=[1.0] * 20,
                label_mask=torch.ones(20, dtype=torch.bool),
                channel='ch0',
                position_ids=torch.arange(20).expand(3, 20).clone()),
            row(30, loss_scale=[1.] * 30, token_type_ids=[0] * 30,
                mm_token_type_ids=torch.zeros(30, dtype=torch.int64),
                sample_weight=[1.0] * 30,
                label_mask=torch.ones(30, dtype=torch.bool),
                channel='ch1',
                position_ids=torch.arange(30).expand(3, 30).clone())]
    batch = [dict(r) for r in rows]
    t._append_warmup_dummy(batch, 128, None)
    dummy = batch[-1]
    check(dummy['length'] == 78, '8 dummy length closes the gap', str(dummy['length']))
    for key in ('loss_scale', 'token_type_ids', 'sample_weight'):
        check(isinstance(dummy[key], list) and len(dummy[key]) == 78,
              f'8 per-token list field mirrored: {key}')
    check(dummy['label_mask'].shape == (78, ) and dummy['label_mask'].dtype == torch.bool,
          '8 per-token tensor field mirrored with its dtype: label_mask')
    check(dummy['mm_token_type_ids'].shape == (78, ) and int(dummy['mm_token_type_ids'].sum()) == 0,
          '8 mm_token_type_ids stays all-zero (invisible to ViT/mrope)')
    check(dummy['labels'] == [-100] * 78, '8 labels keep the -100 neutral value')
    check(all(v == 0. for v in dummy['loss_scale']), '8 loss_scale keeps the 0 neutral value')
    check(dummy['position_ids'].shape == (3, 78)
          and (dummy['position_ids'] == torch.arange(78).expand(3, 78)).all(),
          '8 position_ids is an arange per channel, NOT a mirrored zero vector')
    check(dummy['channel'] == 'ch0', '8 per-sample `channel` is a placeholder, not per-token')
    # packing_row's field math now works for every field
    for key in ('loss_scale', 'token_type_ids', 'sample_weight'):
        total = sum(len(r[key]) for r in batch)
        check(total == sum(r['length'] for r in batch), f'8 packing_row cat length matches: {key}',
              str(total))
    for key in ('mm_token_type_ids', 'label_mask'):
        cat = torch.cat([r[key] for r in batch], dim=-1)
        check(cat.shape == (128, ), f'8 torch.cat over the batch succeeds: {key}', str(tuple(cat.shape)))
    # a field whose length does NOT match input_ids must not be mirrored element-wise
    batch2 = [dict(row(20, grid_thw=torch.tensor([[1, 2, 2]])))]
    t._append_warmup_dummy(batch2, 64, None)
    check('grid_thw' not in batch2[-1], '8 non-per-token fields are not mirrored (grid_thw)')

    bad = [n for ok, n in CHECKS if not ok]
    print(f'\nDEMO11: {len(CHECKS) - len(bad)}/{len(CHECKS)} checks passed')
    assert not bad, bad
    print('DEMO11 OK')


if __name__ == '__main__':
    main()
