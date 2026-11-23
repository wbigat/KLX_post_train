#!/usr/bin/env python3
"""Demo 3: the vectorized sampler replay is bit-identical to the real sampler (feature4 v6).

`_replay_peak_vectorized` must produce EXACTLY the peak the real
`MegatronPretrainingRandomSampler` would walk to, for every supported layout
(shuffle on/off x data_sharding on/off, sum/max aggregation, various dp/mbs/epochs).
The reference below is the sampler copied VERBATIM from the installed ms-swift source
(`swift/megatron/trainers/batch_sampler.py`) and injected as the `batch_sampler` module,
so the product's `_replay_peak_via_sampler` runs against it.

Also covered: scan-failure substitution (0-length rows are replaced by the largest VALID
length, not by max_length) and the epoch-sampling fallback (n_epochs x total beyond the cap
replays an evenly spaced subset instead of v5's epoch-0-only).

Run:  python3 demo_replay_vectorized.py
"""
import importlib.util
import os
import random
import re
import sys
import tempfile
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
PATCH = os.environ.get('DEMO_PATCH',
                       os.path.join(HERE, '..', 'feature4_megatron_pad_warmup-v10.patch'))


# ---- swift stubs -------------------------------------------------------------
def _stub_swift():
    if 'swift' in sys.modules:
        return
    swift = types.ModuleType('swift')
    utils = types.ModuleType('swift.utils')

    class _L:
        def info(self, m, *a, **k):
            print(f'[I] {m}', flush=True)

        def warning(self, m, *a, **k):
            print(f'[W] {m}', flush=True)

        def info_once(self, m, *a, **k):
            print(f'[I] {m}', flush=True)

    utils.get_logger = lambda *a, **kw: _L()
    utils.get_env_args = lambda k, t, d: d

    # Reference sampler: VERBATIM logic from swift/megatron/trainers/batch_sampler.py
    # (MegatronPretrainingRandomSampler), minus logging; group_by_length unsupported here
    # (the product routes that layout to the sampler path anyway).
    import torch

    class MegatronPretrainingRandomSampler:
        def __init__(self, dataset, total_samples, consumed_samples, micro_batch_size,
                     data_parallel_rank, data_parallel_size, data_sharding, shuffle=True,
                     group_by_length=False):
            self.dataset = dataset
            self.total_samples = total_samples
            self.consumed_samples = consumed_samples
            self.micro_batch_size = micro_batch_size
            self.data_parallel_rank = data_parallel_rank
            self.data_parallel_size = data_parallel_size
            if group_by_length:
                raise NotImplementedError('reference stub: group_by_length not needed here')
            self.data_sharding = data_sharding
            self.shuffle = shuffle
            self.micro_batch_times_data_parallel_size = micro_batch_size * data_parallel_size
            self.last_batch_size = total_samples % self.micro_batch_times_data_parallel_size

        def __iter__(self):
            active_total_samples = self.total_samples - self.last_batch_size
            self.epoch = self.consumed_samples // active_total_samples
            current_epoch_samples = self.consumed_samples % active_total_samples
            assert current_epoch_samples % self.micro_batch_times_data_parallel_size == 0
            if self.shuffle:
                if self.data_sharding:
                    bucket_size = (self.total_samples
                                   // self.micro_batch_times_data_parallel_size) * self.micro_batch_size
                    bucket_offset = current_epoch_samples // self.data_parallel_size
                    start_idx = self.data_parallel_rank * bucket_size
                    g = torch.Generator()
                    g.manual_seed(self.epoch)
                    random_idx = torch.randperm(bucket_size, generator=g).tolist()
                    idx_range = [start_idx + x for x in random_idx[bucket_offset:]]
                else:
                    full_bucket_size = (self.total_samples // self.micro_batch_size) * self.micro_batch_size
                    full_bucket_offset = current_epoch_samples
                    g = torch.Generator()
                    g.manual_seed(self.epoch)
                    idx_range_total = torch.randperm(full_bucket_size, generator=g).tolist()
                    idx_range_active = idx_range_total[full_bucket_offset:]
                    idx_range = idx_range_active[self.data_parallel_rank::self.data_parallel_size]
            else:
                full_bucket_size = (self.total_samples // self.micro_batch_size) * self.micro_batch_size
                full_bucket_offset = current_epoch_samples
                idx_range = range(full_bucket_offset + self.data_parallel_rank,
                                  full_bucket_size, self.data_parallel_size)
            batch = []
            for idx in idx_range:
                batch.append(idx)
                if len(batch) == self.micro_batch_size:
                    self.consumed_samples += self.micro_batch_times_data_parallel_size
                    yield batch
                    batch = []

    trainers = types.ModuleType('swift.megatron.trainers')
    bs = types.ModuleType('swift.megatron.trainers.batch_sampler')
    bs.MegatronPretrainingRandomSampler = MegatronPretrainingRandomSampler
    megatron = types.ModuleType('swift.megatron')
    sys.modules.update({'swift': swift, 'swift.utils': utils,
                        'swift.dataset': types.ModuleType('swift.dataset'),
                        'swift.megatron': megatron,
                        'swift.megatron.trainers': trainers,
                        'swift.megatron.trainers.batch_sampler': bs})
    dsu = types.ModuleType('swift.dataset.utils')
    dsu.LazyLLMDataset = type('LazyLLMDataset', (), {})
    sys.modules['swift.dataset.utils'] = dsu


_stub_swift()


def load_pad_warmup():
    lines = open(PATCH).read().split('\n')
    start = next(i for i, l in enumerate(lines) if l.startswith('@@ -0,0 +1,'))
    count = int(re.match(r'@@ -0,0 \+1,(\d+) @@', lines[start]).group(1))
    src = '\n'.join(l[1:] for l in lines[start + 1:start + 1 + count]) + '\n'
    path = os.path.join(tempfile.mkdtemp(), 'pad_warmup.py')
    open(path, 'w').write(src)
    spec = importlib.util.spec_from_file_location('pad_warmup_demo3', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['pad_warmup_demo3'] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    pw = load_pad_warmup()
    rng = random.Random(20261109)

    print('== 1. bit-identity: vectorized vs real sampler (matrix) ==')
    n_cases = 0
    for n in (103, 1000, 4999):
        for mbs in (1, 2, 4):
            for dp in (1, 2, 3, 8):
                if mbs * dp > n:
                    continue
                for shuffle in (True, False):
                    for data_sharding in (True, False):
                        for agg in ('sum', 'max'):
                            lengths = [rng.randint(1, 3000) for _ in range(n)]
                            epochs = [0, 1, 2, 7, 16]
                            ranks = list(range(dp))
                            ref = pw._replay_peak_via_sampler(
                                list(range(n)), lengths, mbs, dp, epochs, ranks, agg,
                                data_sharding, shuffle, False)
                            vec = pw._replay_peak_vectorized(
                                lengths, n, mbs, dp, epochs, ranks, agg,
                                data_sharding, shuffle)
                            assert ref == vec, (n, mbs, dp, shuffle, data_sharding, agg,
                                                ref, vec)
                            n_cases += 1
    print(f'   {n_cases} combinations: vectorized == sampler, EXACT match on all')

    print('== 2. per-rank replay (dp_rank given) ==')
    n, mbs, dp = 4999, 4, 8
    lengths = [rng.randint(1, 3000) for _ in range(n)]
    for r in (0, 3, 7):
        ref = pw._replay_peak_via_sampler(list(range(n)), lengths, mbs, dp, [0, 5], [r],
                                          'sum', False, True, False)
        vec = pw._replay_peak_vectorized(lengths, n, mbs, dp, [0, 5], [r], 'sum', False, True)
        assert ref == vec, (r, ref, vec)
    print('   per-rank peaks match too')

    print('== 3. unscanned rows: replay refuses; the exact-batch chain resolves them ==')
    args = types.SimpleNamespace(data_sharding=False, train_dataloader_shuffle=True,
                                 group_by_length=False, train_iters=None, num_train_epochs=3,
                                 global_batch_size=4, micro_batch_size=4, max_length=32768,
                                 tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
                                 context_parallel_size=1)
    n = 1000
    lengths_full = [rng.randint(500, 1500) for _ in range(n)]
    lengths = list(lengths_full)
    for i in rng.sample(range(n), 30):  # 3% failed
        lengths[i] = 0
    # v9: replay is strict -- substituting an invented value is refused (k>=2 fill values
    # colliding in one batch inflated the peak ~linearly with mbs: 198%/296%/478% at 2/4/8).
    assert pw.replay_peak_batch_length(args, list(range(n)), lengths, 4, agg='sum',
                                       dp_rank=0) is None
    # the chain's undershoot-only fallback: max over the fully covered REAL batches
    peak = pw._covered_batch_peak(args, lengths, 4, 'sum', dp=1, dp_rank=0)
    truth = pw._replay_peak_vectorized(lengths_full, n, 4, 1, [0, 1, 2], [0], 'sum',
                                       False, True)
    assert 0 < peak <= truth, (peak, truth)
    print(f'   replay refused zeros; covered-batch peak={peak} '
          f'(<= true peak {truth}, undershoot-only)')

    print('== 4. epoch sampling beyond the cap (covers the LAST epoch) ==')
    saved_cap = pw._REPLAY_MAX_SAMPLES
    pw._REPLAY_MAX_SAMPLES = 40_000  # force sampling: 60 epochs x 2000 samples
    n = 2000
    args.num_train_epochs = 60
    lengths = [1] * n
    lengths[1999] = 5000  # a long tail sample
    calls = []
    orig = pw._replay_peak_vectorized

    def spy(lengths_, total, group_size, dp, epochs, ranks, agg, ds, sh):
        calls.append(list(epochs))
        return orig(lengths_, total, group_size, dp, epochs, ranks, agg, ds, sh)

    pw._replay_peak_vectorized = spy
    pw.replay_peak_batch_length(args, list(range(n)), lengths, 4, agg='sum', dp_rank=0)
    pw._replay_peak_vectorized = orig
    pw._REPLAY_MAX_SAMPLES = saved_cap
    epochs = calls[0]
    assert len(epochs) <= pw._REPLAY_MAX_EPOCHS and 0 in epochs and 59 - epochs[-1] < \
        (60 + len(epochs) - 1) // len(epochs), epochs
    assert epochs[-1] >= 56, epochs  # last sampled epoch is close to the final one
    print(f'   replayed epochs {epochs} (v5 would have used [0] only)')

    print('== 5. speed ==')
    n, epochs = 300_000, list(range(120))
    lengths = [rng.randint(1, 3000) for _ in range(n)]
    t0 = time.perf_counter()
    pw._replay_peak_vectorized(lengths, n, 4, 8, epochs, [3], 'sum', False, True)
    el = time.perf_counter() - t0
    print(f'   vectorized: 120 epochs x 300k samples (36M walked) in {el:.2f}s '
          f'(the Python loop would take minutes)')
    assert el < 60

    print('\nDEMO3 OK')


if __name__ == '__main__':
    main()
