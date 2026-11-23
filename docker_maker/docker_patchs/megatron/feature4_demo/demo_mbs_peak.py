#!/usr/bin/env python3
"""Demo 9: mbs>1 下 tail-first 峰值估计的精度专项验证（§9.12 / §9.13）。

动机：mbs>1 时峰值是"batch 内 mbs 条样本长度之和"，求和窗口是否会放大估计偏差？
tail-first 是否退回了 §4.1 被否决的 top-k 老路？

方法：10 万行 clustered 重尾数据集（lognormal 主体 + 0.4% 尾 8-40x，索引=长度升序，
索引序的最坏布局；大小代理带噪 rho~0.85）。真值 = 生产函数 `_replay_peak_vectorized`
对真实 sampler 的逐 batch 精确重放（dp=8 x 3 epoch，按本 dp_rank）。估计走生产函数
（单一事实源：从 patch 提取）。

历史结论（2026-11-12 实测，10 万行，v8 CP 线；估计器 tail-estimate 已于 v9 删除）：
  * tail-estimate 在 coverage >= 30% 时对 mbs 1/2/4/8 全部贴在真值的 100-101%；
    极端截断 10% + mbs=8 漂到 ~123%（双向误差——在"可欠不可超"原则下不合格，v9 删除）。
  * substituted（<=5% 缺失按 max(有效) 顶替）膨胀随 mbs 近线性放大：198% (mbs=2) /
    296% (mbs=4) / 478% (mbs=8)——batch 内 k>=2 条顶替行碰撞。v9 已整体摘除该模式。

本 demo（v9）验证新模式链：partial-coverage 峰值对任意 mbs、任意截断点恒 <= 真值
（只欠不超）；索引序对照证明 tail-first 的必要性；严格 exact-batch（两遍扫描）在
proxy 可信时 == 真值。Part F 是 exact-batch × CP 叠加专项（§9.14 TODO 结案）：
两遍扫描的选择在 raw 空间拟合、覆盖校验与峰值求解在 padded 空间，验证 2cp 量化
不掀翻覆盖门禁、严格模式在 padded 空间仍 == padded 真值。

Run:  python3 demo_mbs_peak.py        (PW_SOURCE=local 用本地 pad_warmup.py)
"""
import importlib.util
import math
import os
import random
import re
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
PATCH = os.environ.get('DEMO_PATCH',
                       os.path.join(HERE, '..', 'feature4_megatron_pad_warmup-v10.patch'))

N = 100_000
DP = 8
EPOCHS = [0, 1, 2]
TAIL_ROWS = 400          # 0.4% 重尾
MBS_LIST = [1, 2, 4, 8]


def load_pad_warmup():
    if 'swift' not in sys.modules:
        swift = types.ModuleType('swift')
        utils = types.ModuleType('swift.utils')

        class _L:
            def _p(self, lvl, msg):
                print(f'[{lvl}] {msg}', flush=True)

            def info(self, m, *a, **k):
                pass

            def warning(self, m, *a, **k):
                pass

            def info_once(self, m, *a, **k):
                pass
        utils.get_logger = lambda *a, **kw: _L()
        utils.get_env_args = lambda k, t, d: d
        dsu = types.ModuleType('swift.dataset.utils')
        dsu.LazyLLMDataset = type('LazyLLMDataset', (), {})
        sys.modules.update({'swift': swift, 'swift.utils': utils,
                            'swift.dataset': types.ModuleType('swift.dataset'),
                            'swift.dataset.utils': dsu})
    if os.environ.get('PW_SOURCE', 'patch') == 'local':
        src_path = os.path.join(HERE, 'pad_warmup.py')
    else:
        lines = open(PATCH).read().split('\n')
        start = next(i for i, l in enumerate(lines) if l.startswith('@@ -0,0 +1,'))
        count = int(re.match(r'@@ -0,0 \+1,(\d+) @@', lines[start]).group(1))
        src = '\n'.join(l[1:] for l in lines[start + 1:start + 1 + count]) + '\n'
        src_path = os.path.join(tempfile.mkdtemp(), 'pad_warmup.py')
        open(src_path, 'w').write(src)
    spec = importlib.util.spec_from_file_location('pad_warmup_demo9', src_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['pad_warmup_demo9'] = mod
    spec.loader.exec_module(mod)
    return mod


OK = 0


def check(name, cond, detail=''):
    global OK
    assert cond, f'{name} {detail}'
    OK += 1
    print(f'  ok {name} {detail}')


def make_dataset(n=N, seed=7):
    """lognormal 主体 + 0.4% 重尾(8-40x); size 是 length 的带噪代理(log 域 rho~0.85).
    落盘索引序 = 长度升序(clustered, 索引序扫描的最坏情况, 同 demo6)."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        ln = rng.lognormvariate(0.0, 0.55)
        length = int(2000 * ln) + 200
        size = int(8_000_000 * ln)
        if i < TAIL_ROWS:
            f = 8 + 32 * rng.random()
            length = int(length * f)
            size = int(size * f)
        rows.append((length, int(size * math.exp(rng.gauss(0, 0.3)))))
    rows.sort(key=lambda t: t[0])          # clustered: 索引越靠后越长
    return [r[0] for r in rows], [r[1] for r in rows]


def exact_peaks(pw, lengths, mbs):
    """每个 dp_rank 的精确峰值 (真实 sampler 逐 batch 重放)."""
    return [pw._replay_peak_vectorized(lengths, N, mbs, DP, EPOCHS, [r], 'sum', False, True)
            for r in range(DP)]


def main():
    pw = load_pad_warmup()
    lengths, sizes = make_dataset()
    tail_order = pw._order_tail_first(list(range(N)), {i: s for i, s in enumerate(sizes)})
    print(f'dataset: n={N}, max={max(lengths)}, mean={sum(lengths) // N}, '
          f'tail={TAIL_ROWS} rows (8-40x), clustered layout, dp={DP}, epochs={len(EPOCHS)}')

    # A. per-rank 精确真值：per-rank 离散度是"按本卡峰值"的必要性证据
    print('-- A. 精确 replay 真值 (per-rank)')
    truths = {}
    for mbs in MBS_LIST:
        pk = exact_peaks(pw, lengths, mbs)
        truths[mbs] = pk
        print(f'   mbs={mbs}: per-rank min={min(pk)}, max={max(pk)}, '
              f'spread={max(pk) / min(pk):.2f}x')
    check('A per-rank 峰值离散显著（按本卡峰值的必要性）',
          all(max(p) > 1.5 * min(p) for p in truths.values()),
          f"(spread={max(truths[4]) / min(truths[4]):.2f}x @mbs=4)")

    # B. tail-first 截断 × 生产路径：partial-coverage（只欠不超）
    print('-- B. tail-first 截断下 partial-coverage/真值 (真值=各卡精确峰值的 max)')
    print(f'   {"mbs":>4} {"cov":>6} {"est":>9} {"truth":>9} {"ratio":>7}')
    args = types.SimpleNamespace(data_sharding=False, train_dataloader_shuffle=True,
                                 group_by_length=False, train_iters=None, num_train_epochs=3,
                                 global_batch_size=32, micro_batch_size=4, max_length=1 << 20,
                                 tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
                                 context_parallel_size=1)
    for mbs in MBS_LIST:
        truth = max(truths[mbs])
        best_rank = truths[mbs].index(truth)
        for cov in (0.95, 0.5, 0.3, 0.1):
            scanned_idx = set(tail_order[:int(N * cov)])
            partial = [v if i in scanned_idx else 0 for i, v in enumerate(lengths)]
            est = pw._covered_batch_peak(args, partial, mbs, 'sum', dp=DP, dp_rank=best_rank)
            ratio = est / truth if est else 0.0
            print(f'   {mbs:>4} {cov:>6.0%} {est or 0:>9} {truth:>9} {ratio:>6.1%}')
            # 只欠不超是硬保证；贴住真值不是 partial 的职责（95% 覆盖 + mbs=8 时
            # P(batch 全覆盖)≈0.95^8≈66%，极值统计缺支撑点会明显欠）——贴住由严格模式
            # 的两遍扫描保证（Part E），partial 只是降级兜底。低覆盖 + 大 mbs 下一个全
            # 覆盖 batch 都不存在（est=None），那是特性关闭的信号，同样合法。
            check(f'B mbs={mbs} cov={cov:.0%} partial 只欠不超（est<=真值或 None=off）',
                  est is None or 0 < est <= truth, f'(est={est})')
            if cov >= 0.5:
                check(f'B mbs={mbs} cov={cov:.0%} partial 有值', est is not None)

    # C. 对照 —— 索引序截断 30% (clustered 布局)：覆盖的 batch 全是短行，partial 灾难性欠
    print('-- C. 对照 —— 索引序截断 30% (clustered 布局), 同一 partial 路径')
    for mbs in MBS_LIST:
        truth = max(truths[mbs])
        best_rank = truths[mbs].index(truth)
        partial = [v if i < int(N * 0.3) else 0 for i, v in enumerate(lengths)]
        est = pw._covered_batch_peak(args, partial, mbs, 'sum', dp=DP, dp_rank=best_rank)
        print(f'   mbs={mbs}: {"None(off)" if est is None else f"ratio={est / truth:.1%}"}')
        check(f'C mbs={mbs} 索引序灾难性欠（tail-first 的必要性）',
              est is None or est < 0.5 * truth,
              f'(est={est})')

    # E. 严格 exact-batch：完美 proxy + 两遍扫描 → 峰值 == 真值（这正是设计目标：
    # partial 的欠额由"第二遍把 Top-N batch 的行扫齐"消除）
    print('-- E. 严格 exact-batch（合成完美 proxy, 模拟两遍扫描）== 真值')
    for mbs in MBS_LIST:
        truth = max(truths[mbs])
        best_rank = truths[mbs].index(truth)
        frames = list(lengths)          # 完美线性 proxy（MAPE=0）
        chars = [0] * N
        todo = list(range(N))
        sizes_dict = {i: frames[i] for i in todo}
        orig_proxy = pw._proxy_arrays
        pw._proxy_arrays = lambda _ds, _n: (frames, chars, todo, {}, sizes_dict)
        try:
            # 模拟第一遍只扫了标定头（其余 0），严格模式必因覆盖不足拒绝
            calib_only = [0] * N
            head = pw._order_tail_first(todo, sizes_dict, {})[:pw._EXACT_BATCH_CALIB_ROWS]
            for i in head:
                calib_only[i] = lengths[i]
            refused = pw._exact_batch_peak(args, None, calib_only, mbs, 'sum',
                                           dp=DP, dp_rank=best_rank)
            check(f'E mbs={mbs} 只扫标定头时严格模式拒绝（覆盖门禁）', refused is None)
            # 第二遍：标定头 + 所有 dp rank 的 Top-N 选择行已扫（_exact_batch_scan_plan 的产物）
            coef = pw._fit_length_proxy(frames, chars, calib_only, head)
            assert coef is not None, 'perfect proxy must pass the MAPE gate'
            sel_rows = pw._top_batch_rows(args, N, mbs, 'sum', frames, chars, coef, dp=DP)
            scanned = [v if i in set(sel_rows) or i in set(head) else 0
                       for i, v in enumerate(lengths)]
            est = pw._exact_batch_peak(args, None, scanned, mbs, 'sum', dp=DP,
                                       dp_rank=best_rank)
            check(f'E mbs={mbs} 严格模式 == 真值', est == truth,
                  f'(est={est}, truth={truth}, 扫描行数={sum(1 for v in scanned if v > 0)}/{N})')
        finally:
            pw._proxy_arrays = orig_proxy

    # F. exact-batch × CP 叠加语义（§9.14 TODO 结案专项）。生产顺序：_scan_lengths 两遍
    # 扫描的选择在 raw 长度空间拟合 proxy，get_pad_warmup_length 把已扫长度转入 padded
    # 空间（_cp_pad_length）后才调 _exact_batch_peak 求解。两个空间之间 Top-N 排名不会
    # 错位：覆盖门禁排序用的是 proxy 得分（raw 帧数的平滑函数，批级得分只差常数
    # mbs*c'），2cp 量化残差只进最终求值、不进排名；真峰值 batch 在 proxy 排名 ~6-30
    # 位，远在 Top-200 边界之内。
    print('-- F. exact-batch × CP 叠加（raw 选择 / padded 求解）')
    for cp in (2, 4):
        m = 2 * cp
        padded_lengths = [pw._cp_pad_length(v, cp) for v in lengths]
        args_cp = types.SimpleNamespace(**{**vars(args), 'context_parallel_size': cp})
        frames = list(lengths)          # proxy 特征是数据集属性，CP 不改变它
        chars = [0] * N
        todo = list(range(N))
        sizes_dict = {i: frames[i] for i in todo}
        orig_proxy = pw._proxy_arrays
        pw._proxy_arrays = lambda _ds, _n: (frames, chars, todo, {}, sizes_dict)
        try:
            head = pw._order_tail_first(todo, sizes_dict, {})[:pw._EXACT_BATCH_CALIB_ROWS]
            head_set = set(head)
            calib_only = [0] * N
            for i in head:
                calib_only[i] = lengths[i]
            for mbs in MBS_LIST:
                truths_cp = [pw._replay_peak_vectorized(padded_lengths, N, mbs, DP, EPOCHS,
                                                        [r], 'sum', False, True)
                             for r in range(DP)]
                truth_cp = max(truths_cp)
                best_rank = truths_cp.index(truth_cp)
                check(f'F cp={cp} mbs={mbs} padded 真值 >= raw 真值且 2cp 对齐',
                      truth_cp >= truths[mbs][best_rank] and truth_cp % m == 0,
                      f'(raw={truths[mbs][best_rank]}, padded={truth_cp})')
                # 生产第一遍：raw 标定头拟合 proxy + raw 空间 Top-N 选择（scan 阶段）
                coef_raw = pw._fit_length_proxy(frames, chars, calib_only, head)
                assert coef_raw is not None
                sel = set(pw._top_batch_rows(args_cp, N, mbs, 'sum', frames, chars,
                                             coef_raw, dp=DP))
                # caller 把已扫长度转入 padded 空间后求解（get_pad_warmup_length 的顺序）
                scanned_padded = [pw._cp_pad_length(lengths[i], cp)
                                  if i in sel or i in head_set else 0 for i in range(N)]
                est = pw._exact_batch_peak(args_cp, None, scanned_padded, mbs, 'sum',
                                           dp=DP, dp_rank=best_rank)
                check(f'F cp={cp} mbs={mbs} 严格模式 == padded 真值（raw选择→padded求解）',
                      est == truth_cp, f'(est={est}, truth={truth_cp})')
                # 覆盖门禁在 padded 空间同样生效：只扫标定头必拒
                refused = pw._exact_batch_peak(args_cp, None,
                                               [pw._cp_pad_length(lengths[i], cp)
                                                if i in head_set else 0 for i in range(N)],
                                               mbs, 'sum', dp=DP, dp_rank=best_rank)
                check(f'F cp={cp} mbs={mbs} 只扫标定头时严格模式拒绝（覆盖门禁）',
                      refused is None)
                # partial 在 padded 空间仍只欠不超
                scanned95 = [pw._cp_pad_length(lengths[i], cp)
                             if i in set(tail_order[:int(N * 0.95)]) else 0
                             for i in range(N)]
                est_p = pw._covered_batch_peak(args_cp, scanned95, mbs, 'sum', dp=DP,
                                               dp_rank=best_rank)
                check(f'F cp={cp} mbs={mbs} partial(padded) 有值且只欠不超',
                      est_p is not None and 0 < est_p <= truth_cp, f'(est={est_p})')
        finally:
            pw._proxy_arrays = orig_proxy

    print(f'DEMO9 OK: {OK} checks passed')


if __name__ == '__main__':
    main()
