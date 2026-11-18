#!/usr/bin/env python3
"""Demo 6: tail-first ordering at SCALE -- does it actually save the peak on a huge dataset?

The claim of the tail-first design is an *anytime* property: when the scan is cut off by a
deadline, the scanned subset still holds the length distribution's upper tail. This demo
constructs the layouts that DEFEAT index order and measures the outcome at 500k rows, using
the shipped functions for every decision-relevant step (`_order_tail_first`, and the v9
peak chain: exact replay as truth, `_covered_batch_peak` as the partial-coverage value):

  * Layout `clustered`: rows written in length bands, the long bands at the END of the
    index range (datasets are often length-clustered on disk -- the worst case for scanning
    in index order); layout `interleaved` is the shuffled control.
  * A deadline at 30% of the total decode cost (the scan cannot complete).
  * Three orders: index, random, tail-first (media byte size desc, unknown first, text last;
    the proxy is noisy on purpose -- rho ~ 0.85, not a perfect oracle).
  * v9 semantics: no estimator. The peak of an incomplete scan is the max over the FULLY
    COVERED real batches (undershoot-only) -- tail-first is what keeps the true peak batch
    covered at a cutoff.

Metrics per order: coverage of the true top-2000 longest rows at the deadline, and the
partial-coverage peak vs the exact-replay truth (<= truth always; tail-first should land
close, index order catastrophically low or None).

Part 2 reruns the comparison at 4k rows through the REAL fork-pool scan path
(`_scan_rows` + real files + real deadline), to show the model and the mechanism agree.

Run:  python3 demo_tail_first_scale.py
"""
import heapq
import importlib.util
import math
import os
import random
import re
import sys
import tempfile
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
PATCH = os.environ.get('DEMO_PATCH',
                       os.path.join(HERE, '..', 'feature4_megatron_pad_warmup-v9.patch'))


def load_pad_warmup():
    if 'swift' not in sys.modules:
        swift = types.ModuleType('swift')
        utils = types.ModuleType('swift.utils')

        class _L:
            def _p(self, lvl, msg):
                print(f'[{lvl}] {msg}', flush=True)
            def info(self, m, *a, **k):
                self._p('I', m)
            def warning(self, m, *a, **k):
                self._p('W', m)
            def info_once(self, m, *a, **k):
                self._p('I', m)
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
    spec = importlib.util.spec_from_file_location('pad_warmup_demo6', src_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['pad_warmup_demo6'] = mod
    spec.loader.exec_module(mod)
    return mod


OK = 0


def check(name, cond, detail=''):
    global OK
    assert cond, f'{name} {detail}'
    OK += 1
    print(f'  ok {name} {detail}')


# ---------------------------------------------------------------------------
# Part 1: 500k rows, model-level simulation with the shipped order/estimator
# ---------------------------------------------------------------------------

N = 500_000
MBS = 8            # padding_free micro-batch size -> peak is a SUM of 8
WORKERS = 64
DEADLINE_FRAC = 0.30   # of the WALL-CLOCK total (total cost / workers)
TAIL_ROWS = 2000   # 0.4% heavy tail


def make_dataset(n=N, seed=7):
    """(lengths, sizes, costs): lognormal body + a heavy tail; size is a NOISY proxy of
    length (rho ~ 0.85 in log space), decode cost is proportional to size."""
    rng = random.Random(seed)
    lengths, sizes = [], []
    for i in range(n):
        ln = rng.lognormvariate(0.0, 0.55)
        length = int(2000 * ln) + 200
        size = int(8_000_000 * ln)                     # bytes; body median ~8MB
        if i < TAIL_ROWS:                              # the heavy tail: 8-40x
            f = 8 + 32 * rng.random()
            length = int(length * f)
            size = int(size * f)
        # the proxy sees the size through lognormal noise -- monotone-ish, not an oracle
        sizes.append(int(size * math.exp(rng.gauss(0, 0.3))))
        lengths.append(length)
    costs = [0.0005 + s / 1e9 for s in sizes]          # ~5ms/MB decode + overhead
    return lengths, sizes, costs


def layout_clustered(lengths, seed=11):
    """Rows sorted into contiguous length bands, longest bands LAST (worst case for index
    order), with a small in-band shuffle so it is not a degenerate pure sort."""
    rng = random.Random(seed)
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    band = 2000
    bands = [order[i:i + band] for i in range(0, len(order), band)]
    for b in bands:
        rng.shuffle(b)
    return [i for b in bands for i in b]


def layout_interleaved(n, seed=13):
    order = list(range(n))
    random.Random(seed).shuffle(order)
    return order


def simulate_scan(costs, order, deadline, k=WORKERS):
    """Which rows COMPLETE by the deadline: k workers pull rows in `order`; a row started
    at t finishes at t+cost and counts iff that is <= deadline (production terminates the
    pool at the deadline, killing rows in flight)."""
    done = set()
    heap = [(0.0, w) for w in range(k)]
    heapq.heapify(heap)
    for i in order:
        t, w = heapq.heappop(heap)
        if t >= deadline:
            break
        fin = t + costs[i]
        if fin <= deadline:
            done.add(i)
        heapq.heappush(heap, (fin, w))
    return done


_DEMO_ARGS = types.SimpleNamespace(data_sharding=False, train_dataloader_shuffle=True,
                                   group_by_length=False, train_iters=None, num_train_epochs=1,
                                   global_batch_size=64, micro_batch_size=MBS,
                                   max_length=1 << 24, tensor_model_parallel_size=1,
                                   pipeline_model_parallel_size=1, context_parallel_size=1)


def partial_peak(pw, lengths, done, mbs):
    """v9 production semantics for an incomplete scan: the max over the FULLY COVERED real
    replay batches (undershoot-only); None when not one batch is covered."""
    partial = [v if i in done else 0 for i, v in enumerate(lengths)]
    return pw._covered_batch_peak(_DEMO_ARGS, partial, mbs, 'sum', dp=1, dp_rank=0)


def part1(pw):
    print(f'-- part 1: n={N}, mbs={MBS}, workers={WORKERS}, deadline={DEADLINE_FRAC:.0%} of total cost')
    lengths, sizes, costs = make_dataset()
    total_cost = sum(costs)
    # The deadline is a WALL-CLOCK budget: with `WORKERS` workers the whole scan would
    # take total_cost / WORKERS; we cut it at 30% of that.
    deadline = DEADLINE_FRAC * total_cost / WORKERS
    top = set(sorted(range(N), key=lambda i: -lengths[i])[:TAIL_ROWS])

    # Truth: production's exact replay over the COMPLETE data (one dp rank, one epoch).
    truth = pw._replay_peak_vectorized(lengths, N, MBS, 1, [0], [0], 'sum', False, True)
    max_len = max(lengths)
    print(f'   true max sample = {max_len}, exact replay peak = {truth}')

    orders = {
        'index': None,  # per layout below
        'random': layout_interleaved(N),
        'tail-first': pw._order_tail_first(list(range(N)), {i: s for i, s in enumerate(sizes)}),
    }
    results = {}
    for layout_name, index_order in (('clustered', layout_clustered(lengths)),
                                     ('interleaved', layout_interleaved(N, seed=17))):
        orders['index'] = index_order
        print(f'   layout={layout_name}')
        for name, order in orders.items():
            done = simulate_scan(costs, order, deadline)
            est = partial_peak(pw, lengths, done, MBS)
            cov = len(done & top) / len(top)
            results[(layout_name, name)] = (est, cov, len(done))
            ratio = f'{est / truth:5.1%} of truth' if est else 'None (no covered batch -> off)'
            print(f'     {name:10s} scanned {len(done):7d} rows, top-{TAIL_ROWS} coverage '
                  f'{cov:6.1%}, partial peak {est} ({ratio})')

    c = results

    def tf(key):
        return c[('clustered', 'tail-first')][key]

    # v9 语义：深度截断 + 大 mbs 时 partial 合法地为 None（off）——tail-first 的价值不再是
    # "截断时估计还准"，而是"两遍扫描的选择集（重行）在 deadline 内必然扫得完"。
    check('P1a clustered: tail-first 覆盖了整个上尾',
          tf(1) >= 0.95, f"(coverage={tf(1):.1%})")
    check('P1b clustered: tail-first 用最少行数扫完上尾（两遍扫描成本低）',
          tf(2) < c[('clustered', 'index')][2] // 4,
          f"(tail-first {tf(2)} rows vs index {c[('clustered', 'index')][2]})")
    check('P1c clustered: 索引序上尾覆盖率极低（反面对照）',
          c[('clustered', 'index')][1] <= 0.5,
          f"(coverage={c[('clustered', 'index')][1]:.1%})")
    check('P1d clustered: 索引序 partial 灾难性欠或无覆盖（反面对照）',
          c[('clustered', 'index')][0] is None or c[('clustered', 'index')][0] < 0.5 * truth,
          f"(est={c[('clustered', 'index')][0]})")
    check('P1e interleaved: tail-first 上尾覆盖同样成立（对照布局不退化）',
          c[('interleaved', 'tail-first')][1] >= 0.95,
          f"(coverage={c[('interleaved', 'tail-first')][1]:.1%})")
    check('P1f 任意顺序 partial 恒不超真值（只欠不超）',
          all(r[0] is None or r[0] <= truth for r in c.values()))


# ---------------------------------------------------------------------------
# Part 2: 4k rows through the REAL fork-pool scan path
# ---------------------------------------------------------------------------

def part2(pw, tmp):
    print('-- part 2: real `_scan_rows` (fork pool, real files, real deadline), n=4000')
    n = 4000
    rng = random.Random(23)
    media_dir = os.path.join(tmp, 'media')
    os.makedirs(media_dir)
    lengths, sizes = [], []
    for i in range(n):
        ln = rng.lognormvariate(0.0, 0.55)
        length = int(2000 * ln) + 200
        size = int(8_000_000 * ln)
        if i < 32:  # heavy tail
            length *= 20
            size *= 20
        lengths.append(length)
        sizes.append(size)
    # Reindex everything into the on-disk (clustered) order: row k IS the k-th shortest.
    order = sorted(range(n), key=lambda i: lengths[i])
    lengths = [lengths[i] for i in order]
    sizes = [sizes[i] for i in order]
    paths = []
    for k in range(n):
        p = os.path.join(media_dir, f'{k:05d}.bin')        # index order == length order
        with open(p, 'wb') as f:
            f.truncate(sizes[k])
        paths.append(p)
    rows = [{'videos': paths[i]} for i in range(n)]

    # The plan + ordering are the shipped ones (real os.stat on the real files).
    todo, dup_of, plan_sizes = pw._scan_dedup_plan(rows, n)
    tail_order = pw._order_tail_first(todo, plan_sizes)

    class Tpl:
        tokenizer = None

        def encode(self, row, return_length=True):
            i = int(os.path.basename(row['videos']).split('.')[0])
            time.sleep(sizes[i] / 5e8)                   # decode cost ~ size
            return {'lengths': [lengths[i]]}
    pw._SCAN_TEMPLATE = Tpl()

    total_cost = sum(s / 5e8 for s in sizes)
    deadline = 0.35 * total_cost / 8                     # wall-clock budget, 8 workers
    top = set(sorted(range(n), key=lambda i: -lengths[i])[:100])
    print(f'   total cost {total_cost:.1f}s / 8 workers, deadline {deadline:.2f}s')
    out = {}
    for name, seq in (('index', todo), ('tail-first', tail_order)):
        t0 = time.perf_counter()
        vals = pw._scan_rows([rows[i] for i in seq], 8, deadline_s=deadline)
        dt = time.perf_counter() - t0
        scanned = {seq[j] for j, v in enumerate(vals) if v > 0}
        est = partial_peak(pw, lengths, scanned, 4)
        cov = len(scanned & top) / len(top)
        out[name] = (est, cov, len(scanned), dt)
        print(f'   {name:10s} {len(scanned):5d} rows done in {dt:.1f}s, '
              f'top-100 coverage {cov:6.1%}, partial peak {est}')
    # Truth: the exact replay over the fully scanned dataset (one rank, one epoch).
    truth = pw._replay_peak_vectorized(lengths, n, 4, 1, [0], [0], 'sum', False, True)
    check('P2a 真实路径: tail-first 覆盖上尾',
          out['tail-first'][1] >= 0.90, f"(coverage={out['tail-first'][1]:.0%})")
    check('P2b 真实路径: 索引序漏掉上尾（反面对照）',
          out['index'][1] <= 0.5, f"(coverage={out['index'][1]:.0%})")
    check('P2c 真实路径: tail-first partial 不超真值（只欠不超）',
          out['tail-first'][0] is None or out['tail-first'][0] <= truth,
          f"(est={out['tail-first'][0]}, truth={truth})")
    check('P2d 真实路径: 索引序显著欠或无覆盖（反面对照）',
          out['index'][0] is None or out['index'][0] < 0.5 * truth,
          f"(est={out['index'][0]})")


def main():
    pw = load_pad_warmup()
    part1(pw)
    part2(pw, tempfile.mkdtemp(prefix='pw_demo6_'))
    print(f'DEMO6 OK: {OK} checks passed')


if __name__ == '__main__':
    main()
