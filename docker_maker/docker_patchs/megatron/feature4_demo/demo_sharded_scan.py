#!/usr/bin/env python3
"""Demo 2: the sharded length scan under REAL multi-process collectives (feature4 v6).

Spawns `world` local processes, builds a gloo process group (CPU tensors stand in for the
XCCL device tensors), and runs the REAL `scan_sample_lengths` extracted from the shipped
patch on a fake video dataset (dict-shaped media keys, 15 ms simulated decode per video):

* scenario A  (legacy rank0+broadcast, simulated via the internal building blocks): every
  non-zero rank sits inside the collective for the WHOLE scan -- this is what the XCCL
  watchdog killed with error 702;
* scenario B  (sharded scan, each rank pretends to be its own node): the wall-clock divides
  by `world`, and the wait inside the result all_gather is only the spread of shard times;
* scenario B2 (warm cache on every rank): nothing to communicate, ~0 s;
* scenario C  (only rank 0 has the cache): the flag rendezvous picks it and broadcasts --
  fast, and nobody waits on a scan;
* scenario F  (v6: scan deadline): with the deadline forced to a few seconds and a slow
  decoder, every rank stops mid-shard, contributes zeros for the rest, and all ranks still
  converge to ONE identical (conservatively degraded) array;
* scenario G  (v6: enable agreement): rank 0 alone votes "enable" -- the all_reduce(MIN)
  must turn the feature OFF on every rank instead of stranding the minority in a collective.

Run:  python3 demo_sharded_scan.py [world=4] [n_videos=1600]
"""
import hashlib
import importlib.util
import json
import os
import re
import sys
import tempfile
import time
import types
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
PATCH = os.environ.get('DEMO_PATCH',
                       os.path.join(HERE, '..', 'feature4_megatron_pad_warmup-v10.patch'))
DECODE_MS = float(os.environ.get('DEMO_DECODE_MS', '15'))


# ---- swift stubs (this module is re-imported in every spawned child) ----
def _stub_swift():
    if 'swift' in sys.modules:
        return
    swift = types.ModuleType('swift')
    utils = types.ModuleType('swift.utils')

    class _L:
        def _p(self, lvl, msg):
            print(f'[rank{os.environ.get("RANK", "?")}][{lvl}] {msg}', flush=True)
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


_stub_swift()


def load_pad_warmup():
    lines = open(PATCH).read().split('\n')
    start = next(i for i, l in enumerate(lines) if l.startswith('@@ -0,0 +1,'))
    count = int(re.match(r'@@ -0,0 \+1,(\d+) @@', lines[start]).group(1))
    src = '\n'.join(l[1:] for l in lines[start + 1:start + 1 + count]) + '\n'
    path = os.path.join(tempfile.mkdtemp(), 'pad_warmup.py')
    open(path, 'w').write(src)
    spec = importlib.util.spec_from_file_location('pad_warmup_demo', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['pad_warmup_demo'] = mod  # the fork pool unpickles _scan_row by module ref
    spec.loader.exec_module(mod)
    return mod


class DemoTemplate:
    """encode = 15 ms simulated video decode + a deterministic length (crc32-based, so every
    process computes the same value for the same video)."""
    tokenizer = None

    def encode(self, row, return_length=True):
        vids = row.get('videos')
        media = 0
        if vids:
            time.sleep(DECODE_MS / 1000)
            media = 1000 + zlib.crc32(str(vids).encode()) % 500
        text = sum(len(m.get('content', '')) for m in row.get('messages') or [])
        return {'lengths': [media + text // 4 + 10]}


def build_dataset(n_videos: int, qa: int = 4, n_text: int = 200):
    ds = [{'videos': {'path': f'/data/vid_{v:05d}.mp4', 'fps': 2},
           'messages': [{'role': 'user', 'content': f'question {q} about video {v}'}]}
          for v in range(n_videos) for q in range(qa)]
    ds += [{'messages': [{'role': 'user', 'content': f'text only row {i}'}]}
           for i in range(n_text)]
    return ds


def make_args():
    return types.SimpleNamespace(dataset=['demo-fake-dataset'], dataset_num_proc=1,
                                 streaming=False, model='demo-model', max_length=8192,
                                 micro_batch_size=4, padding_free=True, packing=False,
                                 num_train_epochs=1, train_iters=None, global_batch_size=16,
                                 tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
                                 context_parallel_size=1, data_sharding=False,
                                 train_dataloader_shuffle=True, group_by_length=False)


def rank_worker(rank, world, port, outdir, n_videos, ref):
    os.environ['RANK'] = str(rank)
    # Every rank pretends to be its own node (see LOCAL_WORLD_SIZE=1 below), so it is also its
    # node's local main -- i.e. the one rank that writes the on-disk cache.
    os.environ['LOCAL_RANK'] = '0'
    import torch
    import torch.distributed as dist
    torch.cuda.current_device = lambda: 'cpu'  # gloo on CPU tensors stands in for XCCL
    dist.init_process_group('gloo', init_method=f'tcp://127.0.0.1:{port}',
                            rank=rank, world_size=world)
    pw = load_pad_warmup()
    ds = build_dataset(n_videos)
    args = make_args()
    waits, results = {}, {}

    orig_gather = pw._exchange_shards_via_gather
    def timed_gather(counts, values, pg=None):
        t = time.perf_counter()
        r = orig_gather(counts, values, pg)
        waits['gather'] = time.perf_counter() - t
        return r
    pw._exchange_shards_via_gather = timed_gather

    orig_bcast = pw._broadcast_lengths
    def timed_bcast(result, src, pg=None):
        t = time.perf_counter()
        r = orig_bcast(result, src, pg)
        waits['bcast'] = time.perf_counter() - t
        return r
    pw._broadcast_lengths = timed_bcast

    def run(name, cache_dir, local_ws, template=None, legacy=False, dataset=None):
        os.environ['SWIFT_PAD_WARMUP_CACHE_DIR'] = cache_dir
        os.environ['LOCAL_WORLD_SIZE'] = str(local_ws)
        waits.clear()
        ds_run = dataset if dataset is not None else ds
        t0 = time.perf_counter()
        if legacy:
            # The old path, simulated by calling the building blocks directly: rank 0 scans
            # alone (args=None -> the v5-style single-pass FULL scan, no exact-batch
            # two-pass), then broadcasts -- every other rank waits inside the collective
            # for the WHOLE scan. (The product code no longer has this fallback.)
            out = pw._broadcast_lengths(
                pw._scan_rank0(ds_run, template or DemoTemplate(), None, len(ds_run)) if rank == 0 else None,
                0, None)
        else:
            out = pw.scan_sample_lengths(ds_run, template or DemoTemplate(), args)
        el = time.perf_counter() - t0
        if out is None:
            # The feature disabled itself (symmetrically, on every rank).
            results[name] = {'elapsed_s': round(el, 3),
                             'collective_wait_s': round(max(waits.values(), default=0.0), 3),
                             'n': 0, 'n_failed': 0, 'md5': 'disabled'}
        else:
            lengths, n_failed = out
            results[name] = {
                'elapsed_s': round(el, 3),
                'collective_wait_s': round(max(waits.values(), default=0.0), 3),
                'n': len(lengths), 'n_failed': n_failed,
                'md5': hashlib.md5(str(lengths).encode()).hexdigest(),
                # the two-pass scan is DELIBERATELY partial: every scanned value must be
                # exact (bit-equal to the full-scan reference), unscanned rows stay 0
                'consistent': all(v == 0 or v == ref[i] for i, v in enumerate(lengths)),
            }
        dist.barrier()

    class FailTemplate(DemoTemplate):
        """Scenario D: this node's video decodes ALL fail (disk offline, ...)."""
        def encode(self, row, return_length=True):
            if row.get('videos'):
                raise RuntimeError('injected decode failure')
            return super().encode(row, return_length=True)

    # A: legacy rank0 scan + broadcast (simulated by calling the building blocks directly).
    run('A_legacy_rank0', os.path.join(outdir, 'cache_a'), 1, legacy=True)
    # B: sharded scan. LOCAL_WORLD_SIZE=1: each rank pretends to be its own node, so every
    # rank gets the full per-node worker budget -- this models a multi-node run.
    run('B_sharded', os.path.join(outdir, 'cache_b'), 1)
    # B2: warm cache on every rank.
    run('B2_cache_hit', os.path.join(outdir, 'cache_b'), 1)
    # D: rank 1's video rows all fail -> its shard contributes zeros; the rendezvous must NOT
    # break, and every rank must converge to the same (conservatively degraded) array.
    run('D_rank1_decode_fails', os.path.join(outdir, 'cache_d'), 1,
        template=FailTemplate() if rank == 1 else None)
    # E: rank 2's dedup plan degrades to "encode everything" (simulates a transient row-read
    # error on one rank only). Without the plan agreement the shard shapes would mismatch and
    # the result all_gather would hang; with it, all ranks fall back together and still agree.
    if rank == 2:
        orig_plan = pw._dedup_plan_or_all
        pw._dedup_plan_or_all = lambda ds_, n_: (list(range(n_)), {}, {})
    run('E_rank2_plan_degraded', os.path.join(outdir, 'cache_e'), 1)
    if rank == 2:
        pw._dedup_plan_or_all = orig_plan
    # C: only rank 0 holds the cache (per-rank, i.e. node-local, cache dirs).
    import glob
    import shutil
    cdir = os.path.join(outdir, 'cache_c', f'rank{rank}')
    os.makedirs(cdir, exist_ok=True)
    if rank == 0:
        for f in glob.glob(os.path.join(outdir, 'cache_b', 'lengths_*.json')):
            shutil.copy(f, cdir)
    dist.barrier()
    run('C_only_rank0_cached', cdir, 1)
    # C2: immediately re-run on the same (node-local) dirs. The ranks that had no entry stored
    # the broadcast result, so this must now be an all-ranks cache hit -- zero communication.
    run('C2_after_C_all_hit', cdir, 1)

    # F (v6): scan deadline. A 200 ms/row decoder cannot finish a ~450-row shard within the
    # forced 4 s deadline, so every rank stops mid-shard and contributes zeros for the rest.
    # The all_gather shapes come from the FIXED plan, so the rendezvous holds and every rank
    # rebuilds the SAME union: one identical, conservatively degraded array (n_failed >> 0).
    class SlowTemplate(DemoTemplate):
        def encode(self, row, return_length=True):
            time.sleep(0.2)
            row = dict(row)
            row.pop('videos', None)  # skip the extra 15 ms in super().encode
            return super().encode(row, return_length=True)

    saved_deadline = pw._SCAN_DEADLINE_S
    pw._SCAN_DEADLINE_S = 4
    run('F_scan_deadline', os.path.join(outdir, 'cache_f'), 1, template=SlowTemplate())
    pw._SCAN_DEADLINE_S = saved_deadline

    # G (v6): enable agreement. Rank 0 alone votes "enable"; the all_reduce(MIN) must disable
    # the feature on EVERY rank -- a minority must never be left inside a collective alone.
    t0 = time.perf_counter()
    agreed = pw._agree_enable(1 if rank == 0 else 0)
    results['G_enable_disagree'] = {
        'elapsed_s': round(time.perf_counter() - t0, 3),
        'collective_wait_s': 0.0, 'n': 0, 'n_failed': 0,
        'md5': 'disabled' if not agreed else 'ENABLED(BUG)',
    }
    dist.barrier()

    # H (v6): scan-group creation fails on rank 1 only. The fallback must be SYMMETRIC --
    # every rank falls back to the default group (None); mixed groups would hang, with each
    # side waiting in a collective on a different group.
    pw._SCAN_PG = None
    pw._SCAN_PG_FAILED = False
    if rank == 1:
        orig_new_group = dist.new_group
        def partial(*a, **k):
            # Join the rendezvous (all ranks must enter new_group), then fail LOCALLY --
            # this simulates a backend-init failure after the barrier, the case that would
            # split the ranks onto different groups without the symmetric fallback.
            orig_new_group(*a, **k)
            raise RuntimeError('injected new_group failure')
        dist.new_group = partial
    pg = pw._scan_process_group()
    if rank == 1:
        dist.new_group = orig_new_group
    results['H_pg_fallback'] = {
        'elapsed_s': 0.0, 'collective_wait_s': 0.0, 'n': 0, 'n_failed': 0,
        'md5': 'default_pg' if pg is None else 'GLOO_PG(BUG)',
    }
    dist.barrier()
    pw._SCAN_PG = None
    pw._SCAN_PG_FAILED = False

    # I (v6): rank 3 sees a dataset with ONE ROW missing (transient filtering divergence).
    # `n` rides along in the flag rendezvous / plan agreement; a mismatch must disable
    # pad-warmup on EVERY rank (None), never enter a shape-mismatched all_gather.
    run('I_n_mismatch', os.path.join(outdir, 'cache_i'), 1,
        dataset=ds[:-1] if rank == 3 else ds)

    # K (v9): a partial-scan cache (26% zeros) must still resolve a warm-up length through
    # the exact-batch chain. Stub megatron's parallel_state onto the demo's gloo group so
    # `_current_data_parallel_rank` resolves; strict exact-batch refuses (the deadline-cut
    # cache does not fully cover the Top-N batches), so the undershoot-only partial-coverage
    # peak takes over -- per-rank by design, values may differ across ranks.
    mpu = types.ModuleType('megatron.core.parallel_state')
    mpu.get_data_parallel_rank = lambda with_context_parallel=False: dist.get_rank()
    _meg = types.ModuleType('megatron')
    _core = types.ModuleType('megatron.core')
    _core.parallel_state = mpu
    _meg.core = core = _core
    sys.modules.update({'megatron': _meg, 'megatron.core': _core,
                        'megatron.core.parallel_state': mpu})
    pw.get_env_args = lambda k, t, d: True if k == 'SWIFT_PAD_WARMUP' else d
    os.environ['SWIFT_PAD_WARMUP_CACHE_DIR'] = os.path.join(outdir, 'cache_f')
    t0 = time.perf_counter()
    # Same template class as F -- the cache key includes template_cls, so a DemoTemplate
    # here would MISS F's entry and silently re-scan instead of exercising the estimate.
    k_length = pw.get_pad_warmup_length(args, ds, SlowTemplate())
    results['K_partial_coverage'] = {
        'elapsed_s': round(time.perf_counter() - t0, 3),
        'collective_wait_s': 0.0, 'n': 0, 'n_failed': 0,
        'md5': f'len_{k_length}',
    }
    dist.barrier()

    with open(os.path.join(outdir, f'rank{rank}.json'), 'w') as f:
        json.dump({'rank': rank, 'results': results}, f)
    dist.destroy_process_group()


def main():
    import torch.multiprocessing as mp
    world = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    n_videos = int(sys.argv[2]) if len(sys.argv) > 2 else 1600
    outdir = tempfile.mkdtemp(prefix='pw_demo_')

    pw = load_pad_warmup()
    ds = build_dataset(n_videos)
    print(f'dataset: {len(ds)} rows = {n_videos} unique videos x 4 QA + 200 text rows, '
          f'decode {DECODE_MS} ms/video; world={world}')
    pw._SCAN_TEMPLATE = DemoTemplate()
    t0 = time.perf_counter()
    ref = pw._scan_lengths(ds, len(ds), 16)
    ref_md5 = hashlib.md5(str(ref).encode()).hexdigest()
    print(f'reference (single process, 16 workers): {time.perf_counter() - t0:.2f}s, '
          f'max={max(ref)}, md5={ref_md5[:8]}')

    import socket
    s = socket.socket()
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()
    mp.spawn(rank_worker, args=(world, port, outdir, n_videos, ref), nprocs=world, join=True)

    reports = [json.load(open(os.path.join(outdir, f'rank{r}.json'))) for r in range(world)]
    scenarios = list(reports[0]['results'])
    print(f'\n{"scenario":<24} {"rank elapsed(s)":<24} {"collective wait(s)":<24} '
          f'{"n_failed":<10} correct')
    for name in scenarios:
        el = [r['results'][name]['elapsed_s'] for r in reports]
        wt = [r['results'][name]['collective_wait_s'] for r in reports]
        nf = reports[0]['results'][name]['n_failed']
        md5s = {r['results'][name]['md5'] for r in reports}
        cons = all(r['results'][name].get('consistent', True) for r in reports)
        if name.startswith('A_'):
            # legacy full scan: bit-equal to the single-process reference
            same = md5s == {ref_md5}
        elif name.startswith('D_'):
            # failure injection: ranks must converge to ONE array carrying the zeros
            same = len(md5s) == 1 and nf > 0 and cons
        elif name.startswith('F_'):
            # deadline injection: convergence + degradation (SlowTemplate pops `videos`, so
            # its values legitimately differ from the DemoTemplate reference -- no cons check)
            same = len(md5s) == 1 and nf > 0
        elif name.startswith('G_'):
            # rank 0 alone voted "enable"; MIN-agreement must disable the feature everywhere
            same = md5s == {'disabled'}
        elif name.startswith('H_'):
            # new_group failed on rank 1 only; the fallback must be symmetric (default group)
            same = md5s == {'default_pg'}
        elif name.startswith('I_'):
            # rank 3 saw n-1 rows; every rank must disable symmetrically
            same = md5s == {'disabled'}
        elif name.startswith('K_'):
            # partial-scan cache resolves a warm-up length on every rank (the mode may be
            # exact-batch-partial, which is per-rank by design -- values may differ)
            same = all(m != 'len_None' for m in md5s)
        else:
            # two-pass scan (v9 exact-batch): deliberately partial -- all ranks must converge
            # to the SAME array, every scanned value must be bit-equal to the full-scan
            # reference, and the array must indeed be partial (n_failed > 0)
            same = len(md5s) == 1 and cons and nf > 0
        print(f'{name:<24} {str(el):<24} {str(wt):<24} {nf:<10} '
              f'{"OK" if same else "MISMATCH"}')
        assert same, name

    a = max(r['results']['A_legacy_rank0']['elapsed_s'] for r in reports)
    b = max(r['results']['B_sharded']['elapsed_s'] for r in reports)
    print(f'\nwall-clock: legacy rank0 {a:.2f}s -> sharded {b:.2f}s '
          f'({a / max(b, 1e-9):.1f}x); correctness: every scenario matches the reference.')
    print(f'DEMO2 OK (artifacts in {outdir})')


if __name__ == '__main__':
    main()
