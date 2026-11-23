#!/usr/bin/env python3
"""Demo 12: does pad-warmup actually DO anything for a text-to-text run with `packing`?

This is the question v10 fix (1) claims to answer, and it is answered here against the REAL
upstream code, not stand-ins:

* bins are produced by the real `swift.dataset.packing.calculate_matched_group`
  (i.e. `binpacking.to_constant_volume(sequences, packing_length)`);
* the micro-batch is assembled the way the real dataloader does it (`mbs` bins per batch);
* the batch is collated by the REAL `Template.data_collator`, which is the function that
  flattens the bins (`batch = sum(batch, start=[])`), calls `_append_warmup_dummy`, and then
  packs the row -- none of that is reimplemented here.

What it checks
--------------
A. the layout fact that fix (1) rests on: one micro-batch really is `mbs` bins flattened into
   ONE padding_free row, so the row length is a SUM over bins, not a single bin;
B. v9's target (`max_length`) vs v10's (`mbs * packing_length`) under that layout: whether a
   dummy is appended at all, and what the carved row length ends up being;
C. "undershoot, never overshoot": over every micro-batch of a realistic t2t dataset, the v10
   target must be >= the real peak (never under), and the overshoot is quantified (bin fill
   ratio decides how tight it is);
D. the degenerate case fix (1) must not break: `mbs=1` (target == packing_length) and
   `packing_length > max_length`;
E. CP interaction: the 2*cp alignment of the target must not drop below the real peak.

Run inside the container (needs `binpacking` + the real swift):
    python demo_packing_t2t.py
"""
import os
import sys
import types

CHECKS = []


def check(ok, name, detail=''):
    CHECKS.append((bool(ok), name))
    print(f'  {"ok" if ok else "FAIL"} {name}' + (f' ({detail})' if detail else ''))


def load_v10_swift():
    """Import a swift tree that is a KNOWN v10 state.

    The container's own site-packages is a MIXED state (template/base.py carries an older
    patch, megatron/trainers/base.py is pristine), so it is not trustworthy as the
    code-under-test. A copy is made and the three patched files are overwritten with the v10
    ones, then that copy is put first on sys.path.
    """
    root = os.environ.get('V10_SWIFT_ROOT', '/tmp/t2t')
    if not os.path.isdir(os.path.join(root, 'swift')):
        sys.exit(f'{root}/swift missing -- build it first (see the runner command)')
    sys.path.insert(0, root)
    import swift
    assert os.path.realpath(swift.__file__).startswith(os.path.realpath(root)), \
        f'wrong swift picked up: {swift.__file__}'
    return swift


def load_pad_warmup(path=None, name='pw_t2t'):
    """Load a `pad_warmup` module from file (not via `import swift...`, which would drag in
    the megatron import chain). Defaults to the v10 file under test; pass `path` for the v9
    reference implementation extracted from `-v9.patch`."""
    import importlib.util
    if path is None:
        path = os.path.join(os.environ.get('V10_SWIFT_ROOT', '/tmp/t2t'),
                            'swift/megatron/trainers/pad_warmup.py')
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.get_env_args = lambda k, t_, d: k == 'SWIFT_PAD_WARMUP'
    return mod


def load_real_get_padding_to():
    """Extract the REAL `get_padding_to` from `swift/megatron/utils/utils.py` by source.

    Importing it normally (`from swift.megatron.utils.utils import ...`) executes
    `swift/megatron/__init__.py` -> `init_megatron_env()` -> `mcore_bridge` -> `megatron` ->
    `fla`, and `fla/utils.py` calls `torch.cuda.get_device_name(0)` at import time, which
    raises "No CUDA GPUs are available" on this box. The function is pure arithmetic over
    `args`, so its source is executed standalone -- still the shipped code, not a rewrite.
    """
    import ast
    path = os.path.join(os.environ.get('V10_SWIFT_ROOT', '/tmp/t2t'),
                        'swift/megatron/utils/utils.py')
    src = open(path).read()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == 'get_padding_to':
            ns = {}
            exec(compile(ast.Module(body=[node], type_ignores=[]), path, 'exec'), ns)
            return ns['get_padding_to']
    raise SystemExit('get_padding_to not found in ' + path)


def main():
    load_v10_swift()
    from swift.dataset.packing import calculate_matched_group
    from swift.template.base import Template

    print('== 0. the code under test is the real upstream one ==')
    import swift.template.base as tb
    import swift.dataset.packing as dp
    check('sum(batch, start=[])' in open(tb.__file__).read(),
          '0 真实 data_collator 含 bin 摊平语句 `batch = sum(batch, start=[])`')
    check('to_constant_volume' in open(dp.__file__).read(),
          '0 真实装箱走 binpacking.to_constant_volume')
    check(hasattr(Template, '_append_warmup_dummy'),
          '0 被测 Template 带 v10 的 _append_warmup_dummy')

    # ---------------------------------------------------------------- helpers
    def make_template(padding_free=True, packing=True, cp=1):
        """The real Template, no checkpoint: data_collator only touches these attributes."""
        t = Template.__new__(Template)
        t.processor = types.SimpleNamespace(pad_token_id=0, all_special_ids=[1, 2])
        t.padding_side = 'right'
        t.mode = 'train'
        t.use_megatron = True
        t.padding_free = padding_free
        t.sequence_parallel_size = cp
        t.packing = packing
        t.task_type = 'causal_lm'
        t.image_token_id = 32000
        t.max_length = 4096
        t.remove_unused_columns = True      # skips the `_extra_kwargs` merge tail
        return t

    def row(n, i=0):
        """One t2t sample of n tokens (no media -- this is the text-to-text case)."""
        return {'input_ids': [5 + ((i + k) % 3000) for k in range(n)],
                'labels': [5 + ((i + k) % 3000) for k in range(n)],
                'length': n, 'loss_scale': [1.] * n}

    def pack(lengths, packing_length):
        """Real binpacking -> list of bins, each bin a list of rows."""
        seqs = [(row(n, i), n) for i, n in enumerate(lengths)]
        bins, _ = calculate_matched_group(seqs, packing_length, is_finished=True)
        return [[r for r, _ in b] for b in bins]

    def collate(t, bins, pad_to_length=None):
        """Exactly what the dataloader hands to the collator: a LIST OF BINS."""
        batch = [list(b) for b in bins]          # fresh copy: the collator mutates it
        res = t.data_collator(batch, pad_to_length=pad_to_length)
        return res

    PACKING_LENGTH = 1024

    # ---------------------------------------------------------------- A: layout
    print('\n== A. 布局事实：一个 micro-batch = mbs 个 bin 摊平成一行 ==')
    t = make_template()
    lens_a = [300, 280, 260, 290, 310, 270, 320, 240, 280, 300, 260, 290]
    bins_a = pack(lens_a, PACKING_LENGTH)
    check(all(sum(r['length'] for r in b) <= PACKING_LENGTH for b in bins_a),
          f'A 每个 bin <= packing_length={PACKING_LENGTH}',
          f'bins={[sum(r["length"] for r in b) for b in bins_a]}')
    mbs = 3
    assert len(bins_a) >= mbs, f'装箱只得到 {len(bins_a)} 个 bin，不足 mbs={mbs}'
    mb = bins_a[:mbs]
    real_total = sum(r['length'] for b in mb for r in b)
    res = collate(t, mb)
    got = int(res['input_ids'].shape[-1])
    check(got == real_total,
          'A padding_free 下 mbs 个 bin 被拼成 ONE 行，长度 = 各 bin 之和',
          f'mbs={mbs}, 行长={got}, 各 bin={[sum(r["length"] for r in b) for b in mb]}')
    check(got > PACKING_LENGTH,
          'A 该行长度已超过单个 packing_length（所以 target 不能按单 bin 算）',
          f'{got} > {PACKING_LENGTH}')

    # ---------------------------------------------------------------- B: v9 vs v10
    print('\n== B. v9 target(max_length) vs v10 target(mbs*packing_length)：有没有效果 ==')
    max_length = PACKING_LENGTH          # megatron_base_args: seq_length = packing_length or max_length
    v9_target = max_length
    v10_target = mbs * PACKING_LENGTH

    base_len = int(collate(make_template(), mb)['input_ids'].shape[-1])
    v9_len = int(collate(make_template(), mb, pad_to_length=v9_target)['input_ids'].shape[-1])
    v10_len = int(collate(make_template(), mb, pad_to_length=v10_target)['input_ids'].shape[-1])

    check(v9_len == base_len,
          'B v9：target 比真实行长还小 -> dummy_len<=0 -> 一个 dummy 都没加（特性空转）',
          f'target={v9_target}, 真实={base_len}, 结果={v9_len}')
    check(v9_target < base_len,
          'B v9 的 target 确实低于真实行长（这是空转的原因，不是"预留不够"）',
          f'{v9_target} < {base_len}')
    check(v10_len == v10_target,
          'B v10：行被真实 pad 到 target（特性真实生效）',
          f'target={v10_target}, 结果={v10_len}')
    check(v10_len > v9_len,
          'B v10 雕出的形状严格大于 v9（v9 等于没装这个特性）',
          f'{v10_len} vs {v9_len}')
    # the dummy must be loss-neutral: it may not contribute gradient
    lab = res['labels'] if 'labels' in res else None
    r10 = collate(make_template(), mb, pad_to_length=v10_target)
    pad_region = r10['labels'][0, base_len:]
    check(int((pad_region != -100).sum()) == 0,
          'B v10 的 dummy 段 labels 全 -100（不产生 CE 梯度）',
          f'dummy 段长度={pad_region.numel()}')
    check(r10['position_ids'][0, base_len] == 0 if r10['position_ids'].dim() == 2
          else r10['position_ids'][..., base_len].min() == 0,
          'B v10 的 dummy 是独立因果段（position_ids 从 0 重新开始）')

    # ---------------------------------------------------------------- C: 只欠不超
    print('\n== C. 只欠不超：遍历真实数据集的每个 micro-batch ==')

    def sweep(name, lengths, packing_length, mbs_):
        bins = pack(lengths, packing_length)
        target = mbs_ * packing_length
        peaks, fills = [], []
        for s in range(0, len(bins) - mbs_ + 1, mbs_):
            group = bins[s:s + mbs_]
            tot = sum(r['length'] for b in group for r in b)
            peaks.append(tot)
            fills.append(tot / target)
        if not peaks:
            return None
        peak = max(peaks)
        check(peak <= target,
              f'C [{name}] 真实峰值 <= v10 target（不欠预留）',
              f'peak={peak}, target={target}')
        # and the real peak must be reachable by the collator at that target
        over = target / peak
        print(f'     [{name}] bins={len(bins)} mbs={mbs_} target={target} '
              f'真实峰值={peak} 超出={over:.3f}x 平均填充率={sum(fills)/len(fills):.3f}')
        return over

    # (c1) 典型 t2t：样本远短于 packing_length -> 装箱填得很满
    import random
    random.seed(0)
    short = [random.randint(80, 400) for _ in range(600)]
    over1 = sweep('t2t 短样本 80-400', short, PACKING_LENGTH, 4)
    check(over1 is not None and over1 < 1.05,
          'C 典型 t2t（样本远短于 packing_length）：target 几乎就是真实峰值',
          f'超出={over1:.3f}x')

    # (c2) 长样本：单条 > packing_length/2 -> 一个 bin 只能放一条 -> 填充率低
    long_ = [random.randint(560, 1000) for _ in range(300)]
    over2 = sweep('长样本 560-1000（bin 装不满）', long_, PACKING_LENGTH, 4)
    check(over2 is not None and over2 >= 1.0,
          'C 长样本场景：target 仍是上界（只欠不超的方向正确）')
    check(over2 is not None and over2 > over1,
          'C 长样本的超出幅度确实更大（填充率低 -> 上界更松）',
          f'{over2:.3f}x > {over1:.3f}x')

    # (c3) 极端：每条都接近 packing_length -> 每 bin 一条 -> 填充率 ~最差
    worst = [random.randint(900, 1020) for _ in range(200)]
    over3 = sweep('极端 900-1020（每 bin 一条）', worst, PACKING_LENGTH, 4)
    check(over3 is not None and over3 >= 1.0, 'C 极端场景仍不欠预留')

    # (c4) 混合（真实数据常见形状）
    mixed = ([random.randint(60, 200) for _ in range(400)]
             + [random.randint(700, 1000) for _ in range(120)])
    random.shuffle(mixed)
    over4 = sweep('混合长短', mixed, PACKING_LENGTH, 4)
    check(over4 is not None and over4 >= 1.0, 'C 混合场景仍不欠预留')

    # (c5) the case where "never overshoots" is least likely to be tight: few micro-batches
    # AND long samples, so the max over batches may not approach the hard cap. Report the
    # number honestly instead of claiming tightness everywhere.
    small_long = [random.randint(560, 1000) for _ in range(40)]
    over5 = sweep('小数据集+长样本（上界最松的场景）', small_long, PACKING_LENGTH, 4)
    check(over5 is not None and over5 >= 1.0, 'C 小数据集仍不欠预留（方向始终正确）')

    # ---------------------------------------------------------------- D: 边界
    print('\n== D. 边界：mbs=1 与 packing_length > max_length ==')
    bins_d = pack(short[:200], PACKING_LENGTH)
    t1 = make_template()
    one = bins_d[:1]
    tgt1 = 1 * PACKING_LENGTH
    len1 = int(collate(t1, one, pad_to_length=tgt1)['input_ids'].shape[-1])
    check(len1 == tgt1, 'D mbs=1：target == packing_length，行被 pad 到它',
          f'{len1} == {tgt1}')
    real1 = sum(r['length'] for r in one[0])
    check(tgt1 >= real1, 'D mbs=1 不欠预留', f'target={tgt1} >= 真实={real1}')

    # packing_length 抬高到 max_length 之上：v9 用 max_length 会欠得更多
    PL2 = 2048
    bins_e = pack(short[:300], PL2)
    mbs2 = 2
    grp = bins_e[:mbs2]
    real2 = sum(r['length'] for b in grp for r in b)
    v9_t2, v10_t2 = 1024, mbs2 * PL2          # v9: max_length=1024
    l9 = int(collate(make_template(), grp, pad_to_length=v9_t2)['input_ids'].shape[-1])
    l10 = int(collate(make_template(), grp, pad_to_length=v10_t2)['input_ids'].shape[-1])
    check(l9 == real2 and v9_t2 < real2,
          'D packing_length(2048) > max_length(1024)：v9 target 更低，依然空转',
          f'v9 target={v9_t2} < 真实={real2}')
    check(l10 == v10_t2 and v10_t2 >= real2,
          'D v10 按 mbs*packing_length 取上界，生效且不欠',
          f'target={v10_t2} >= 真实={real2}')

    # ---------------------------------------------------------------- E: CP
    # Real link: `_get_data_collator` binds `partial(template.data_collator,
    # padding_to=get_padding_to(args))`, and `get_padding_to` returns a MULTIPLE (cp * tp-sp *
    # fp8/fused factors), not an absolute length -- `_data_collator` then does
    # `padding_to = ceil(max(seq_lens) / padding_to) * padding_to`. For plain t2t without
    # TP-SP / CP / fp8 / fused it is None, which is why sections A-D pass None.
    # So here the target comes from the PRODUCTION function `get_pad_warmup_length`, and the
    # collator is driven with the real multiple.
    print('\n== E. CP：生产函数算出的 target 与真实 collator 的闭环 ==')
    pw = load_pad_warmup()
    pw.get_env_args = lambda k, t_, d: k == 'SWIFT_PAD_WARMUP'
    real_get_padding_to = load_real_get_padding_to()

    for cp in (1, 2, 4):
        args = types.SimpleNamespace(
            max_length=PACKING_LENGTH, packing=True, packing_length=PACKING_LENGTH,
            padding_free=True, micro_batch_size=mbs, context_parallel_size=cp,
            tensor_model_parallel_size=1, sequence_parallel=False,
            fp8_recipe=None, fp8_format=None, fp4_format=None, fp8=None, fp4=None,
            attention_backend='auto')
        tpl = types.SimpleNamespace(mode='train')
        target = pw.get_pad_warmup_length(args, None, tpl)
        check(target is not None, f'E cp={cp}：packing 分支算出 target（未关闭）', str(target))
        check(target >= mbs * PACKING_LENGTH,
              f'E cp={cp}：target >= mbs*packing_length（不欠上界）',
              f'{target} >= {mbs * PACKING_LENGTH}')
        if cp > 1:
            check(target % (2 * cp) == 0,
                  f'E cp={cp}：target 是 2*cp 的倍数（_handle_megatron_cp 逐样本 2cp padding '
                  f'不会再把 dummy 二次 ceil）', f'{target} % {2 * cp} == 0')
        # drive the real collator with the real multiple
        # The real `get_padding_to`, loaded from its source file: `import swift.megatron...`
        # pulls in mcore_bridge -> megatron -> fla, which needs a real GPU and dies here with
        # "No CUDA GPUs are available". The function itself is pure arg arithmetic.
        mult = real_get_padding_to(args)
        check(mult == (cp if cp > 1 else None),
              f'E cp={cp}：get_padding_to 返回的是倍数而非绝对长度', f'{mult}')
        tc = make_template(cp=cp)
        res_cp = tc.data_collator([list(b) for b in mb], padding_to=mult,
                                  pad_to_length=target)
        lc = int(res_cp['input_ids'].shape[-1])
        check(lc >= real_total, f'E cp={cp}：collate 后行长 >= 真实总长（不欠）',
              f'{lc} >= {real_total}')
        check(lc == target, f'E cp={cp}：行长恰好落在 target 上（没有被二次 ceil 冲高）',
              f'{lc} == {target}')

    # ---------------------------------------------------------------- F: v9 module vs v10 module
    # Sections B/D used a hand-written `v9_target = max_length`. Here the SAME comparison is
    # driven by the REAL v9 module (extracted from `-v9.patch`, 2610 lines), so the "v9 did
    # nothing" conclusion comes from v9's own code rather than from my reading of it.
    print('\n== F. 真实 V9 模块对照（不依赖我手写 target） ==')
    v9_path = os.environ.get('V9_PAD_WARMUP', '/tmp/v9_pad_warmup.py')
    if not os.path.isfile(v9_path):
        print(f'  SKIP {v9_path} 不存在，跳过真实 V9 对照')
    else:
        pw9 = load_pad_warmup(v9_path, 'pw_v9_ref')
        pw10 = load_pad_warmup(name='pw_v10_ref')

        def mk_args(mbs_, cp=1, pl=PACKING_LENGTH, ml=PACKING_LENGTH):
            return types.SimpleNamespace(
                max_length=ml, packing=True, packing_length=pl, padding_free=True,
                micro_batch_size=mbs_, context_parallel_size=cp,
                tensor_model_parallel_size=1, sequence_parallel=False,
                fp8_recipe=None, fp8_format=None, fp4_format=None, fp8=None, fp4=None,
                attention_backend='auto')

        tpl = types.SimpleNamespace(mode='train')
        t9 = pw9.get_pad_warmup_length(mk_args(mbs), None, tpl)
        t10 = pw10.get_pad_warmup_length(mk_args(mbs), None, tpl)
        check(t9 == PACKING_LENGTH,
              'F 真实 V9 模块：packing 分支返回 _finalize(max_length)，与 mbs 无关',
              f'V9 target={t9}, mbs={mbs}')
        check(t10 == mbs * PACKING_LENGTH,
              'F 真实 V10 模块：target = mbs*packing_length', f'V10 target={t10}')

        # the same real bins through the real collator, once per target
        # Fresh bins: `collate` only shallow-copies the bin list, and section E ran the real
        # collator with cp=4, whose `_handle_megatron_cp` pads every sample IN PLACE to a 2*cp
        # multiple -- so `mb`/`real_total` from section A are stale by now (observed: 2640 ->
        # 2664, i.e. 9 samples rounded up to 8). Each section must own its rows.
        mb_f = pack(lens_a, PACKING_LENGTH)[:mbs]
        real_f = sum(r['length'] for b in mb_f for r in b)
        check(real_f == real_total,
              'F 用独立新 bin（前段 CP 用例会就地 pad 样本，不能复用）',
              f'{real_f} == {real_total}')
        l9 = int(collate(make_template(), mb_f, pad_to_length=t9)['input_ids'].shape[-1])
        l10 = int(collate(make_template(), mb_f, pad_to_length=t10)['input_ids'].shape[-1])
        check(l9 == real_f,
              'F 用 V9 自己的 target 过真实 collator：行长不变（确认空转）',
              f'V9 target={t9} < 真实={real_f} -> 结果={l9}')
        check(l10 == t10,
              'F 用 V10 的 target 过真实 collator：行长被 pad 到 target',
              f'{l10} == {t10}')

        # how much of a realistic dataset v9 could never cover
        for name, lengths in (('t2t 短样本', short), ('长样本', long_), ('混合', mixed)):
            bins = pack(lengths, PACKING_LENGTH)
            n_batch = len(bins) // mbs
            over = 0
            for s in range(0, n_batch * mbs, mbs):
                tot = sum(r['length'] for b in bins[s:s + mbs] for r in b)
                if tot > t9:
                    over += 1
            check(over == n_batch,
                  f'F [{name}] V9 target 低于**每一个** micro-batch 的真实长度（100% 空转）',
                  f'{over}/{n_batch} 个 batch 超出 V9 target={t9}')

        # the boundary that must NOT change: mbs=1
        for mbs_ in (1, 2, 4, 8):
            a9 = pw9.get_pad_warmup_length(mk_args(mbs_), None, tpl)
            a10 = pw10.get_pad_warmup_length(mk_args(mbs_), None, tpl)
            if mbs_ == 1:
                check(a9 == a10 == PACKING_LENGTH,
                      'F mbs=1：V9 与 V10 完全一致（修复没有误伤本来就正确的场景）',
                      f'V9={a9}, V10={a10}')
            else:
                check(a10 == mbs_ * a9,
                      f'F mbs={mbs_}：V10 恰好是 V9 的 mbs 倍（不多不少）',
                      f'V9={a9}, V10={a10}')

    # ---------------------------------------------------------------- G: packing + CP, real bins
    # The flat cap `mbs * packing_length` assumes "bin <= packing_length", but CP rounds EVERY
    # SAMPLE up to a 2*cp multiple before `packing_row`, so a full bin overshoots. The fix reads
    # the REAL `PackingDataset` layout (packed_idx / packed_length, built by upstream itself) and
    # sums the CP-rounded top-mbs bins. Here a real PackingDataset is instantiated, so the bins
    # under test are upstream's own, and every micro-batch is collated by the real collator.
    print('\n== G. packing+CP：真实 PackingDataset 装箱 + 精确上界 ==')
    from swift.dataset.packing import PackingDataset

    class FakeDS:
        """Minimal stand-in for the LazyLLMDataset upstream passes in: PackingDataset only
        reads `dataset['lengths']` (to pack) and `dataset[i]` (to build a row)."""

        def __init__(self, lengths):
            self.lengths = list(lengths)

        def __len__(self):
            return len(self.lengths)

        def __getitem__(self, k):
            if k == 'lengths':
                return self.lengths
            return row(self.lengths[k], k)

    def real_peak_over_batches(pd, cp, mbs_):
        peak = 0
        n = len(pd)
        for s in range(0, n - mbs_ + 1, mbs_):
            batch = [pd[i] for i in range(s, s + mbs_)]
            res = make_template(cp=cp).data_collator(batch, padding_to=cp if cp > 1 else None)
            peak = max(peak, int(res['input_ids'].shape[-1]))
        return peak

    def mk_args_g(mbs_, cp):
        return types.SimpleNamespace(
            max_length=PACKING_LENGTH, packing=True, packing_length=PACKING_LENGTH,
            padding_free=True, micro_batch_size=mbs_, context_parallel_size=cp,
            tensor_model_parallel_size=1, sequence_parallel=False,
            fp8_recipe=None, fp8_format=None, fp4_format=None, fp8=None, fp4=None,
            attention_backend='auto')

    tpl_g = types.SimpleNamespace(mode='train')
    pw_g = load_pad_warmup(name='pw_g')

    # Gold standard for "the layout the run will really see": the REAL
    # MegatronPretrainingRandomSampler from the shipped batch_sampler.py, loaded from its source
    # file so `swift/megatron/__init__` (mcore_bridge -> fla -> CUDA) never executes. It is
    # driven over the same epochs the replay uses, and every batch it yields is collated by the
    # real collator.
    import importlib.util as _iu
    _sp = _iu.spec_from_file_location(
        'swift_real_batch_sampler',
        os.path.join(os.environ.get('V10_SWIFT_ROOT', '/tmp/t2t'),
                     'swift/megatron/trainers/batch_sampler.py'))
    _sm = _iu.module_from_spec(_sp)
    _sp.loader.exec_module(_sm)
    REAL_SAMPLER = _sm.MegatronPretrainingRandomSampler

    def real_sampler_peak(pd, cp, mbs_, epochs, dp=1, rank=0):
        total = len(pd)
        step_samples = mbs_ * dp
        active = (total // step_samples) * step_samples
        peak = 0
        for epoch in epochs:
            sampler = REAL_SAMPLER(pd, total_samples=total, consumed_samples=0,
                                   micro_batch_size=mbs_, data_parallel_rank=rank,
                                   data_parallel_size=dp, data_sharding=False,
                                   shuffle=True, group_by_length=False)
            sampler.consumed_samples = epoch * active
            for batch in sampler:
                res = make_template(cp=cp).data_collator(
                    [pd[i] for i in batch], padding_to=cp if cp > 1 else None)
                peak = max(peak, int(res['input_ids'].shape[-1]))
        return peak

    for name, lengths in (('crafted 201（CP 放大最大）', [201] * 40),
                          ('t2t 短样本', short[:200]),
                          ('长样本', long_[:120])):
        for cp in (1, 2, 4, 8):
            pd = PackingDataset(make_template(cp=cp), FakeDS(lengths),
                                packing_length=PACKING_LENGTH, packing_num_proc=1)
            mbs_g = 4
            flat = pw_g._cp_pad_length(pw_g._align(mbs_g * PACKING_LENGTH, mk_args_g(mbs_g, cp)),
                                       cp)
            new = pw_g.get_pad_warmup_length(mk_args_g(mbs_g, cp), pd, tpl_g)
            epochs_g = pw_g._replay_epochs(mk_args_g(mbs_g, cp), len(pd), mbs_g, 1)[0]
            gold = real_sampler_peak(pd, cp, mbs_g, epochs_g)
            check(new is not None and new >= gold,
                  f'G [{name}] cp={cp}：target >= 真实 sampler 布局峰值（不欠）',
                  f'target={new}, 真实={gold}')
            if cp > 1:
                check(gold > flat or new >= gold,
                      f'G [{name}] cp={cp}：硬上界欠预留已被精确路径覆盖',
                      f'硬上界={flat}, 真实峰值={gold}, 新 target={new}')
            print(f'     [{name}] cp={cp}: 真实 sampler 峰值={gold} 硬上界={flat} '
                  f'新 target={new} '
                  f'({"硬上界欠" if flat < gold else "硬上界够"}; '
                  f'target/真实={new/gold:.3f}x)')

    # mbs=1 must stay correct, and the layout-less fallback must still work
    pd1 = PackingDataset(make_template(cp=1), FakeDS(short[:50]),
                         packing_length=PACKING_LENGTH, packing_num_proc=1)
    t_mbs1 = pw_g.get_pad_warmup_length(mk_args_g(1, 1), pd1, tpl_g)
    gold1 = real_sampler_peak(pd1, 1, 1, [0])
    check(t_mbs1 >= gold1, 'G mbs=1：精确路径同样不欠', f'target={t_mbs1}, 峰值={gold1}')
    check(t_mbs1 <= PACKING_LENGTH, 'G mbs=1：target 不超过 packing_length（不比 V9 更松）',
          f'{t_mbs1} <= {PACKING_LENGTH}')

    # no packed layout -> flat cap fallback (streaming / IterablePackingDataset)
    bare = types.SimpleNamespace()  # no packed_idx at all
    fb = pw_g.get_pad_warmup_length(mk_args_g(4, 8), bare, tpl_g)
    check(fb == pw_g._cp_pad_length(pw_g._align(4 * PACKING_LENGTH, mk_args_g(4, 8)), 8),
          'G 无 bin 布局（streaming 等）时回退硬上界', f'{fb}')
    broken = types.SimpleNamespace(packed_idx=[[0, 1]], packed_length=[10],
                                   dataset=object())  # lengths not readable
    fb2 = pw_g.get_pad_warmup_length(mk_args_g(4, 4), broken, tpl_g)
    check(fb2 == pw_g._cp_pad_length(pw_g._align(4 * PACKING_LENGTH, mk_args_g(4, 4)), 4),
          'G 布局读取失败时安全回退硬上界（不抛异常）', f'{fb2}')

    # ---------------------------------------------------------------- H: the three layers
    # `_packing_bin_peak` now tries a SAMPLER REPLAY over the bin sequence first (the dataloader
    # samples bins through MegatronPretrainingRandomSampler, so the bin lengths are the exact
    # analogue of the sample-level replay), falling back to the top-mbs bound. The replay is what
    # removes the ~14% looseness the top-mbs bound had in the one-sample-per-bin (long sample)
    # regime. Here all three numbers are measured against the same real layout: the replay
    # target, the top-mbs bound, and the real collator's worst shuffled layout.
    print('\n== H. 三层口径：replay 消除长样本场景的 top-mbs 松度 ==')

    def mk_args_h(mbs_, cp, train_iters=50, gbs=32):
        return types.SimpleNamespace(
            max_length=PACKING_LENGTH, packing=True, packing_length=PACKING_LENGTH,
            padding_free=True, micro_batch_size=mbs_, context_parallel_size=cp,
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
            train_iters=train_iters, global_batch_size=gbs, num_train_epochs=1,
            data_sharding=False, train_dataloader_shuffle=True, group_by_length=False,
            fp8_recipe=None, fp8_format=None, fp4_format=None, fp8=None, fp4=None,
            attention_backend='auto')

    def bin_lens_of(pd, cp):
        lengths = pd.dataset['lengths']
        per = [sum(x) if isinstance(x, list) else int(x) for x in lengths]
        f = (lambda v: pw_g._cp_pad_length(v, cp)) if cp > 1 else (lambda v: int(v))
        return [sum(f(per[i]) for i in bin) for bin in pd.packed_idx]

    for name, lengths in (('长样本（每 bin 一条，top-mbs 最松）', long_[:120]),
                          ('t2t 短样本', short[:200]),
                          ('crafted 201', [201] * 40)):
        for cp in (1, 4, 8):
            pd = PackingDataset(make_template(cp=cp), FakeDS(lengths),
                                packing_length=PACKING_LENGTH, packing_num_proc=1)
            mbs_h = 4
            new = pw_g.get_pad_warmup_length(mk_args_h(mbs_h, cp), pd, tpl_g)
            top = pw_g._cp_pad_length(
                pw_g._align(sum(sorted(bin_lens_of(pd, cp), reverse=True)[:mbs_h]),
                            mk_args_h(mbs_h, cp)), cp)
            epochs_h = pw_g._replay_epochs(mk_args_h(mbs_h, cp), len(pd), mbs_h, 1)[0]
            gold = real_sampler_peak(pd, cp, mbs_h, epochs_h)
            check(new == gold,
                  f'H [{name}] cp={cp}：replay 与真实 sampler 布局峰值完全一致（精确复现）',
                  f'replay={new} == 真实={gold}')
            check(new <= top,
                  f'H [{name}] cp={cp}：replay 不比 top-mbs 松',
                  f'replay={new}, top-mbs={top}')
            print(f'     [{name}] cp={cp}: replay={new} == 真实={gold}, top-mbs={top} '
                  f'(replay/真实={new/gold:.3f}x, 比 top-mbs 省 {top - new})')

    # args without train_iters / num_train_epochs / global_batch_size: the replay still runs
    # (defaults to epoch 0) and must still not under-reserve.
    broken_args = types.SimpleNamespace(
        max_length=PACKING_LENGTH, packing=True, packing_length=PACKING_LENGTH,
        padding_free=True, micro_batch_size=4, context_parallel_size=4,
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
        data_sharding=False, train_dataloader_shuffle=True, group_by_length=False)
    pd_b = PackingDataset(make_template(cp=4), FakeDS(long_[:80]),
                          packing_length=PACKING_LENGTH, packing_num_proc=1)
    t_broken = pw_g.get_pad_warmup_length(broken_args, pd_b, tpl_g)
    gold_b = real_sampler_peak(pd_b, 4, 4, [0])
    check(t_broken == gold_b, 'H args 缺 epoch 相关字段（replay 缺省单 epoch）与真实布局一致',
          f'{t_broken} == {gold_b}')

    bad = [n for ok, n in CHECKS if not ok]
    print(f'\nDEMO12: {len(CHECKS) - len(bad)}/{len(CHECKS)} checks passed')
    assert not bad, bad
    print('DEMO12 OK')


if __name__ == '__main__':
    main()
