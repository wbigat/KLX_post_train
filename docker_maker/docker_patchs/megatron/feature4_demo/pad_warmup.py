# Copyright (c) ModelScope Contributors. All rights reserved.
"""Pad-warmup for megatron training: pad the first batch to the peak packed length to fight memory
fragmentation when `expandable_segments` is unavailable.

Why
---
With `padding_free`, a micro-batch is one packed row whose length is the *sum* of the sample
lengths, so it varies with the data order. On allocators without `expandable_segments` the caching
allocator cannot grow a freed segment in place: once early (short) batches have carved the memory
into small segments, a later (longer) batch cannot find a contiguous block even though the *total*
free memory is enough:

    torch.OutOfMemoryError: Tried to allocate 15.60 GiB. GPU 3 has a total capacity of 96.00 GiB
    of which 38.53 GiB is free...

The fix: pad the FIRST batch to the peak packed length, so the allocator reserves the big segments
up-front and every later batch -- whose length is <= the warm-up one -- reuses them.

How
---
* **Dummy segment (padding_free).** A dummy sample of ``warmup_len - real_total`` tokens is
  appended before ``packing_row``. It gets ``position_ids = range(pad)``, so it becomes one extra
  segment in ``cu_seqlens``, with ``labels = -100`` / ``loss_scale = 0``: no loss, no gradient,
  and being a separate causal segment it cannot influence the real tokens. In the PADDED layout
  no dummy row is added at all: a row is padded to the batch's longest sample, so the peak shape
  ``(mbs, warmup_len)`` is reached by raising ``padding_to``. An extra row would carve
  ``(mbs+1, warmup_len)`` -- 2x at ``mbs=1``, and the 4-D mask scales with the row count too.
* **Peak, not an upper bound.** Replay `MegatronPretrainingRandomSampler` (deterministic,
  seed = epoch) for every data-parallel rank and every epoch and take the largest micro-batch.
  Exact, with no quantile/coverage magic number. The old ``sum of the top-``micro_batch_size``
  sample lengths`` is *not* used: it assumes the longest samples of the whole dataset land in one
  micro-batch, which random shuffling essentially never produces -- reserving it turns step 0 into
  an artificial global peak. The failure modes are ASYMMETRIC: over-reserving dies early (the
  step-0 batch OOMs by itself -- the feature kills an otherwise healthy run, unacceptable), while
  under-reserving only leaves residual risk (never worse than the feature being off). So: measure
  what the scan actually covered, never estimate; a missed peak degrades the feature, an invented
  one poisons a healthy run.
* **Settle, then carve.** The peak-shaped batch is carved at step 1, but part of the heap's
  *permanent* footprint only materialises around that step: Adam's `exp_avg`/`exp_avg_sq` are
  allocated by the inner optimizer at its first `step()` (megatron-core's `init_state_fn` is
  otherwise only invoked on the checkpoint-load path). Whatever lands after the carve can pin
  and split the pre-carved blocks -- the "reserved but unusable" fragmentation. So before the
  carve, the optimizer states are allocated directly through `init_state_fn` (a pure
  `zeros_like` allocation, idempotent via its `len(state[p]) == 0` guard, so a resumed run is
  unaffected), and a gc + `empty_cache` releases every still-free segment: the carve starts
  from a clean heap wrapped around the stable footprint. A real settle *step* would land the
  shape-dependent workspaces too, but those are transient, while its side effects (RNG drift
  through dropout, Adam moments absorbing the settle gradients) are not.
* **Tail-first scan.** Rows are scanned biggest-media-first (frame count for a path sequence,
  file size otherwise -- the decode-cost proxy), which makes the scan an *anytime* algorithm:
  at any cutoff the unscanned rows are the short ones, so the scanned subset still holds the
  length distribution's upper tail. If the scan completes, the peak is the exact replay of
  this rank's real micro-batches. If it does not, the peak comes from the exact-batch chain:
  proxy-calibrated Top-N real batches when the gates pass, else the max over the fully
  covered real batches (undershoot-only), else the feature turns itself off (an estimate can
  overshoot, and overshoot kills healthy runs). Missing
  rows are NEVER substituted with an invented length: a deterministic fill value colliding
  k>=2 times in one batch inflates the peak ~linearly with mbs (measured 198%/296%/478% at
  mbs 2/4/8), and a too-loose reservation OOMs the step-0 warm-up batch itself -- the one
  failure mode that kills an otherwise healthy run. A closed-loop check in the collator warns
  if any real batch ever exceeds the warm-up length.
* **Exactly once per rank.** `PadWarmupCollator` runs inside the dataloader worker, and a fresh
  worker (with a fresh collator) is forked every epoch when `dataloader_persistent_workers=False`.
  Besides a per-process counter it therefore claims a one-shot flag file, so the whole run pads one
  batch per rank. The flag is deleted when the collator is wrapped, so a flag left behind by an
  earlier run (same `output_dir`, e.g. a resume) can never silently skip the warm-up. Only
  worker 0 may claim it -- a DataLoader returns the batches in worker order, so any other
  worker would pad a batch that is not the FIRST one executed, after the small batches ahead of
  it already carved the heap. The VALIDATION dataloader keeps the unwrapped collator
  (`trainers/base.py`), or an eval batch could consume the claim on a forward-only path. Every
  padding is logged, so the behaviour stays auditable.
* **Scan cache.** Encoding the dataset is the only expensive step and is a pure function of the
  sample plus the tokenizer/template, so the per-sample length array is cached on disk. The key
  holds exactly what can change a sample's token length -- see `_cache_entry`. Every rank reads
  and writes the cache, and a tiny flag rendezvous picks the path: a hit on every rank needs no
  further communication at all, a hit on some ranks is broadcast from the lowest one (the source
  only reads a local file, so nobody waits inside the collective for a slow scan).
* **Fast scan.** Encoding one video sample means decoding the file, and that dominates the
  scan. Rows sharing the same media payload (this dataset: 485 videos x 7 QA pairs) are
  decoded once and the duplicates are recovered from their own text length, and the fork pool
  runs single-threaded workers -- one thread per worker scales far better than a thread pool
  per worker, which only oversubscribes the machine. The dedup plan itself is built
  column-wise: one vectorized read per media column instead of n row random-accesses, with
  Image/Video features read undecoded (arrow `path` child / `decode=False` cast) -- a row-wise
  pass over a decode=True feature would decode every file up-front, and the resulting PIL
  objects could never be frozen into a dedup key at all.
* **Sharded scan.** The scan is distributed over all ranks: the deduplicated media rows are
  dealt out round-robin (rank ``r`` takes ``todo[r::world]`` -- an interleaving that mixes
  short and long media into every shard, so the shards finish within seconds of each other),
  and each rank runs a fork pool sized to its share of the node's CPUs. The wall-clock is
  ``total_decode / (nodes x workers_per_node)`` instead of one node decoding everything while
  the other ranks wait inside a broadcast -- which is exactly what the XCCL watchdog killed on
  slow scans (error 702). The shard results are exchanged through ONE small all_gather (the
  payload is the length array itself, ~n x 8B), so the wait inside the collective is only the
  spread of the shard finish times, not the whole scan. The shard layout is derived from the
  dedup plan, so the ranks first agree on it with a 2-int collective -- otherwise a plan that
  degraded on one rank only would mismatch the result shapes and hang. No file locks and no
  file polling anywhere: a crashed peer fails the collective loudly instead of leaving the
  others blocked on an invisible rendezvous forever.
* **Failure containment.** Every scan-phase collective runs on a dedicated gloo group
  (CPU tensors, hours-long timeout), so a slow scan can neither trip the NCCL/XCCL watchdog
  of the default group nor, on failure, abort the communicators training already built. The
  scan itself is bounded twice -- a per-chunk stall timeout (a corrupt media file can block
  a decoder forever) and an overall deadline; either one turns the missing rows into scan
  failures, and the peak is then resolved from what was actually covered (exact-batch /
  exact-batch-partial), never by substituting invented lengths.
  The enable/disable decision is agreed across ranks (all_reduce MIN) both before the scan
  and after the peak is resolved, so no rank can ever enter a collective alone.
* **Lazy raw allocations.** Anything that allocates OUTSIDE the caching allocator on first use
  must happen before the carve, or it finds the heap already full: the collective backends'
  per-group context (`_prewarm_communicators`, one all_reduce per group), the pipeline group's
  POINT-TO-POINT path (`_prewarm_pipeline_p2p`, one ring `batch_isend_irecv` -- an all_reduce
  does not establish it) and the DeepEP all-to-all buffer (`_prewarm_deepep_buffer`).
* **Self-disabling.** The feature turns itself off (and says why) rather than guess, whenever
  its model of the peak would not hold: `template.mode` outside train/transformers (RLHF/PRM
  index per-segment logits by `seq_lens`), `expandable_segments:True` (the allocator already
  handles fragmentation), streaming datasets, an unusable scan, or no resolvable peak.
  Note that it does NOT disable itself on configurations that are merely unvalidated (CUDA
  graphs, non-causal `task_type`): the collator's try/except already degrades those to the
  unpadded batch, and silently withdrawing a memory guarantee a run depends on is the worse
  failure. Unvalidated combinations are documented, not switched off.

Usage
-----
    SWIFT_PAD_WARMUP=1                 # enable it; off by default
    SWIFT_PAD_WARMUP_CACHE=0           # disable the length cache
    SWIFT_PAD_WARMUP_CACHE_DIR=/tmp/x  # keep the length cache somewhere else (a shared FS
                                       # lets every node reuse one scan)
    SWIFT_PAD_WARMUP_CE_RESERVE=0      # disable the single transparent CE-grad reserve block
    SWIFT_PAD_WARMUP_PP_P2P=1          # opt in to the pipeline p2p pre-warm (PP>1; never run
                                       # against a real backend, and it can HANG -- see
                                       # `_prewarm_pipeline_p2p`)
"""
import array
import base64
import functools
import hashlib
import json
import math
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from swift.utils import get_env_args, get_logger

logger = get_logger()

# A single transparent CE-grad reserve. The logits allocation is left to the normal allocator;
# v7 handles persistent optimizer state and MoE router-cache lifetime independently.
_CE_RESERVE_BUF = None
_CE_RESERVE_SHAPE = None


def acquire_ce_reserve_buf():
    global _CE_RESERVE_BUF
    if _CE_RESERVE_BUF is None and _CE_RESERVE_SHAPE is not None:
        import torch
        _CE_RESERVE_BUF = torch.empty(_CE_RESERVE_SHAPE, dtype=torch.bfloat16, device='cuda')


def release_ce_reserve_buf():
    global _CE_RESERVE_BUF
    _CE_RESERVE_BUF = None


# How many consecutive batches a dataloader worker pads once it has claimed the warm-up.
_WARMUP_BATCHES = 1
# The vectorized replay walks `n_epochs x total` tensor elements (~10^8/s); beyond this many
# combinations, replay an evenly spaced subset of at most `_REPLAY_MAX_EPOCHS` epochs instead
# (still covering the whole run -- v5's epoch-0-only fallback UNDERESTIMATED the peak).
_REPLAY_MAX_SAMPLES = 500_000_000
_REPLAY_MAX_EPOCHS = 16
# Every scan-phase collective runs on a dedicated gloo group (CPU tensors) with its own long
# timeout: a slow scan must neither trip the NCCL/XCCL watchdog of the default group nor, on
# failure, abort the communicators training already built.
_SCAN_PG_TIMEOUT_S = 6 * 3600
# A rank stops scanning after this and contributes zeros; the mode chain then resolves the
# peak from what was actually covered (exact-batch -> partial -> off) instead of the job
# dying on a collective timeout. Sized for the two-pass scan (calib + Top-N rows: tens of
# seconds even at ~20s/video); a full-scan fallback that exceeds it degrades to partial.
_SCAN_DEADLINE_S = 600
# No single decode may take longer than this: a corrupt media file can block a decoder
# (ffmpeg/decord) forever, which would otherwise hang the whole job inside a collective.
_SCAN_ROW_TIMEOUT_S = 600


def _scan_num_proc(args) -> int:
    """Per-node fork-worker budget for the length scan (scales with `dataset_num_proc`).

    Every worker is pinned to one thread (see `_scan_worker_init`): decoding and resizing are
    CPU-bound, and a thread pool per worker only oversubscribes the machine -- 16
    multi-threaded workers reached 6.7x, 64 single-threaded workers gave another ~1.8x and
    the curve is flat beyond that (measured: 64/128/256 within noise). No override knob:
    the two-pass exact-batch scan only touches ~calib+Top-N rows anyway.

    The budget is additionally capped by the host's AVAILABLE memory at ~2 GiB per worker
    (4K frames + decode buffers): the scan runs after `setup_model_training`, so the model
    and the optimizer already hold most of the host RAM, and an OOM-killed worker is
    indistinguishable from a stuck decode (stall timeout -> conservative shutdown).
    """
    budget = max(1, min(64, (getattr(args, 'dataset_num_proc', 1) or 1) * 16))
    try:
        with open('/proc/meminfo') as f:
            avail_kb = next(int(l.split()[1]) for l in f if l.startswith('MemAvailable'))
        budget = max(1, min(budget, avail_kb // (2 * 1024 * 1024)))
    except Exception:
        pass
    return budget


def _local_world_size() -> int:
    """Ranks sharing this node's CPUs (torchrun sets `LOCAL_WORLD_SIZE`)."""
    try:
        value = int(os.environ.get('LOCAL_WORLD_SIZE', 0) or 0)
        if value > 0:
            return value
    except ValueError:
        pass
    try:
        import torch
        return max(1, torch.cuda.device_count())
    except Exception:
        return 1


def _is_local_main() -> bool:
    """True on one rank per node: the only one that needs to write the shared on-disk state.

    The length array is a few MB of JSON for a large dataset (500k samples ~ 3 MB), and every
    rank writing it means `world_size` concurrent writes of an identical file -- pointless load
    on a shared FS. One writer per node covers both layouts: a node-local cache dir gets its own
    copy, a shared one just gets a few redundant (atomic, identical) writes instead of dozens.
    """
    value = os.environ.get('LOCAL_RANK')
    if value is None:
        return True   # launcher did not say; better to write than to lose the cache
    try:
        return int(value) == 0
    except ValueError:
        return True


def _scan_num_proc_sharded(args) -> int:
    """One rank's share of the per-node worker budget, so co-located ranks do not oversubscribe
    the node: the total per node stays `_scan_num_proc`, and the scan's throughput scales with
    the number of *nodes*."""
    return max(1, _scan_num_proc(args) // _local_world_size())


_SCAN_PG = None
_SCAN_PG_FAILED = False


def _scan_process_group():
    """Dedicated gloo group for every scan-phase collective; None falls back to the default group.

    CPU tensors only: the payloads are tiny integer arrays (the largest is the length array
    itself, ~n x 8B), so a GPU backend buys nothing -- while a failure on the *default* group
    would abort the NCCL/XCCL communicators that training already built. Created once and by
    every rank (every rank calls `scan_sample_lengths`), so the rendezvous is symmetric.

    The fallback is symmetric too: a rank that failed to create the group while its peers
    succeeded would wait on a DIFFERENT group than theirs (a hang, not an error), so the
    success flag is agreed with a MIN all_reduce on the default group -- anyone's failure
    moves EVERYONE to the default group.
    """
    global _SCAN_PG, _SCAN_PG_FAILED
    if _SCAN_PG is not None or _SCAN_PG_FAILED:
        return _SCAN_PG
    import datetime
    import torch.distributed as dist
    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
        return None   # nothing to rendezvous: offline / single-rank runs need no group
    pg = None
    try:
        if not dist.is_gloo_available():
            raise RuntimeError('the gloo backend is not available')
        pg = dist.new_group(backend='gloo',
                            timeout=datetime.timedelta(seconds=_SCAN_PG_TIMEOUT_S))
    except Exception as e:
        logger.warning(f'[pad_warmup] Cannot create the dedicated scan process group ({e}).')
    try:
        import torch
        ok = torch.tensor([1 if pg is not None else 0], dtype=torch.long,
                          device=torch.cuda.current_device())
        dist.all_reduce(ok, op=dist.ReduceOp.MIN)
        ok = int(ok.item())
    except Exception:
        ok = 0
    if not ok:
        _SCAN_PG_FAILED = True
        if pg is not None:
            logger.warning('[pad_warmup] Another rank failed to create the scan group; '
                           'falling back to the default group (watchdog timeouts apply).')
        return None
    _SCAN_PG = pg
    return _SCAN_PG


def _scan_worker_init() -> None:
    """Pool initializer: hide the GPU, keep one decode/resize thread per worker.

    The pool is forked AFTER the parent process initialized CUDA (the scan runs past
    `setup_model_training`): a child inheriting that context must never touch the device.
    Decoding is pure CPU work, so hiding it is free insurance against fork-after-CUDA."""
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    import torch
    torch.set_num_threads(1)


def _align(length: int, args) -> int:
    """Round the warm-up length up to the attention-backend alignment (`get_padding_to`)."""
    try:
        from swift.megatron.utils.utils import get_padding_to
        padding_to = get_padding_to(args)
    except Exception:
        padding_to = None
    if padding_to:
        return math.ceil(length / padding_to) * padding_to
    return length


def _cp_pad_length(length: int, cp_size: int) -> int:
    """The per-sample padding the collator applies under context parallelism
    (`_handle_megatron_cp`: every sample becomes a 2*cp multiple, so the zigzag split can
    view each segment as (2*cp, len/(2*cp)))."""
    if cp_size <= 1:
        return length
    m = 2 * cp_size
    return math.ceil(length / m) * m


def _data_parallel_size(args) -> int:
    try:
        import torch.distributed as dist
        world = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1
    except Exception:
        world = 1
    tp = max(1, int(getattr(args, 'tensor_model_parallel_size', 1) or 1))
    pp = max(1, int(getattr(args, 'pipeline_model_parallel_size', 1) or 1))
    cp = max(1, int(getattr(args, 'context_parallel_size', 1) or 1))
    return max(1, world // (tp * pp * cp))


def _num_epochs(args, total_samples: int, step_samples: int) -> int:
    """How many distinct sampler epochs this run will walk through."""
    if step_samples <= 0:
        return 1
    active = (total_samples // step_samples) * step_samples
    if active <= 0:
        return 1
    train_iters = getattr(args, 'train_iters', None)
    if train_iters:
        gbs = max(1, int(getattr(args, 'global_batch_size', 1) or 1))
        return max(1, math.ceil(int(train_iters) * gbs / active))
    return max(1, int(getattr(args, 'num_train_epochs', 1) or 1))


def _current_data_parallel_rank() -> Optional[int]:
    try:
        from megatron.core import parallel_state as mpu
        return mpu.get_data_parallel_rank()
    except Exception:
        return None


def _replay_peak_via_sampler(hf_ds, lengths: List[int], group_size: int, dp: int,
                             epochs: List[int], ranks: List[int], agg: str,
                             data_sharding: bool, shuffle: bool,
                             group_by_length: bool) -> Optional[int]:
    """Replay through the real `MegatronPretrainingRandomSampler`, one batch at a time.

    Used for `group_by_length` (its indices come from transformers' length-grouped sampler,
    which is not worth re-deriving in tensor math) and as the reference implementation for
    the vectorized path's equivalence tests. Zero lengths (failed rows) must have been
    resolved by the caller's mode chain.
    """
    try:
        from swift.megatron.trainers.batch_sampler import MegatronPretrainingRandomSampler
    except Exception as e:
        logger.warning(f'[pad_warmup] Cannot import the batch sampler: {e}')
        return None
    total = len(lengths)
    step_samples = group_size * dp
    active = (total // step_samples) * step_samples
    combine = sum if agg == 'sum' else max
    best = 0
    for epoch in epochs:
        # Cumulative sample count, it legitimately exceeds `total` across epochs -- the real run
        # keeps incrementing `sampler.consumed_samples` and derives `epoch` from it. It must be
        # assigned after construction because the constructor asserts `consumed < total`.
        consumed = epoch * active
        for rank in ranks:
            try:
                sampler = MegatronPretrainingRandomSampler(
                    hf_ds,
                    total_samples=total,
                    consumed_samples=0,
                    micro_batch_size=group_size,
                    data_parallel_rank=rank,
                    data_parallel_size=dp,
                    data_sharding=data_sharding,
                    shuffle=shuffle,
                    group_by_length=group_by_length,
                )
                sampler.consumed_samples = consumed
            except Exception as e:
                logger.warning(f'[pad_warmup] Cannot build the sampler (epoch {epoch}, rank {rank}): {e}')
                return None
            for batch in sampler:
                try:
                    value = combine(lengths[i] for i in batch)
                except Exception:
                    return None
                if value > best:
                    best = value
    return best


def _replay_peak_vectorized(lengths: List[int], total: int, group_size: int, dp: int,
                            epochs: List[int], ranks: List[int], agg: str,
                            data_sharding: bool, shuffle: bool) -> int:
    """Bit-identical, tensorized replay of `MegatronPretrainingRandomSampler` for the plain
    shuffle layouts.

    The sampler is: ``perm = randperm(seed=epoch)`` over the active bucket; this rank's stream
    is ``perm[dp_rank::dp]`` (or its own sub-bucket with `data_sharding`); micro-batches are
    consecutive `group_size` chunks, last partial chunk dropped. Replay always starts each
    epoch at offset 0 (``consumed = epoch * active``), so the sampler's
    ``current_epoch_samples`` is 0 here and every step is tensor math: ~10^8 walked samples
    cost seconds instead of the minutes a Python loop needs.
    """
    import torch
    L = torch.tensor(lengths, dtype=torch.long)
    mbs = group_size
    full_bucket = (total // mbs) * mbs        # sampler: full_bucket_size
    bucket = (total // (mbs * dp)) * mbs      # sampler: bucket_size (data_sharding)
    best = 0
    for epoch in epochs:
        for r in ranks:
            # A FRESH generator per rank: the real sampler builds one per sampler instance
            # (i.e. per rank), so every rank sees the SAME permutation seeded by `epoch` --
            # sharing one generator across ranks would give rank>0 a different permutation
            # than the training run actually uses.
            g = torch.Generator()
            g.manual_seed(epoch)
            if not shuffle:
                idx = torch.arange(r, full_bucket, dp, dtype=torch.long)
            elif data_sharding:
                idx = r * bucket + torch.randperm(bucket, generator=g)
            else:
                idx = torch.randperm(full_bucket, generator=g)[r::dp]
            nb = idx.numel() // mbs
            if nb <= 0:
                continue
            vals = L[idx[:nb * mbs].view(nb, mbs)]
            peak = int(vals.sum(1).max()) if agg == 'sum' else int(vals.max(1).values.max())
            if peak > best:
                best = peak
    return best


def _proxy_from_plan(n: int, dup_of: Dict[int, int], sizes: Dict[int, int],
                     chars: Dict[int, int]) -> Tuple[List[int], List[int]]:
    """(frames, text_chars) for every row; a duplicate takes its representative's media."""
    frames = [int(sizes.get(dup_of.get(i, i), 0) or 0) for i in range(n)]
    return frames, [int(chars.get(i, 0)) for i in range(n)]


def _proxy_arrays(hf_ds, n: int) -> Optional[Tuple[List[int], List[int], List[int], Dict[int, int],
                                                   Dict[int, int]]]:
    """(frames, chars, todo, dup_of, sizes) for the whole dataset, decode-free, or None."""
    try:
        todo, dup_of, sizes = _dedup_plan_or_all(hf_ds, n)
        chars = _text_char_sizes(hf_ds, list(range(n)))
        frames, chars = _proxy_from_plan(n, dup_of, sizes, chars)
        return frames, chars, todo, dup_of, sizes
    except Exception as e:
        logger.info(f'[pad_warmup] exact-batch: cannot build the proxy arrays: {e}')
        return None


def _fit_length_proxy(frames: List[int], chars: List[int], lengths,
                      rows: List[int]) -> Optional[Tuple[float, float, float, float]]:
    """``tokens ~ a*frames + b*chars + c`` by least squares over `rows` (labels: `lengths`).

    Returns ``(a, b, c, mape)`` or None when there are too few labelled rows or the in-sample
    error fails the quality gate (the proxy then does not describe this dataset).
    """
    import torch
    rows = [i for i in rows if lengths[i] > 0]
    if len(rows) < _EXACT_BATCH_MIN_CALIB:
        return None
    A = torch.tensor([[float(frames[i]), float(chars[i]), 1.0] for i in rows],
                     dtype=torch.float64)
    y = torch.tensor([float(lengths[i]) for i in rows], dtype=torch.float64)
    try:
        sol = torch.linalg.lstsq(A, y.unsqueeze(1)).solution.squeeze(1)
    except Exception:
        return None
    mape = float((((A @ sol) - y).abs() / y.clamp(min=1)).mean())
    if not (mape <= _EXACT_BATCH_MAX_MAPE):
        return None
    return float(sol[0]), float(sol[1]), float(sol[2]), mape


def _replay_epochs(args, total: int, group_size: int, dp: int) -> Tuple[List[int], int]:
    """The epoch list the peak replay (and the exact-batch selection) will walk."""
    n_epochs = _num_epochs(args, total, group_size * dp)
    if n_epochs * total > _REPLAY_MAX_SAMPLES and n_epochs > _REPLAY_MAX_EPOCHS:
        # Evenly sampled epochs still cover the WHOLE run, including the last one.
        step = math.ceil(n_epochs / _REPLAY_MAX_EPOCHS)
        return list(range(0, n_epochs, step)), n_epochs
    return list(range(n_epochs)), n_epochs


def _sampler_batches(total: int, group_size: int, dp: int, rank: int, epochs: List[int],
                     data_sharding: bool, shuffle: bool):
    """``(nb, group_size)`` sample-index matrix of one dp rank's micro-batches.

    Same layout math as `_replay_peak_vectorized`: one fresh generator per (epoch, rank),
    ``perm[rank::dp]`` (or the rank's own sub-bucket with `data_sharding`), micro-batches are
    consecutive chunks, the last partial chunk is dropped.
    """
    import torch
    mbs = group_size
    full_bucket = (total // mbs) * mbs
    bucket = (total // (mbs * dp)) * mbs
    out = []
    for epoch in epochs:
        g = torch.Generator()
        g.manual_seed(epoch)
        if not shuffle:
            idx = torch.arange(rank, full_bucket, dp, dtype=torch.long)
        elif data_sharding:
            idx = rank * bucket + torch.randperm(bucket, generator=g)
        else:
            idx = torch.randperm(full_bucket, generator=g)[rank::dp]
        nb = idx.numel() // mbs
        if nb:
            out.append(idx[:nb * mbs].view(nb, mbs))
    return torch.cat(out, 0) if out else torch.zeros(0, mbs, dtype=torch.long)


def _top_batch_rows(args, n: int, group_size: int, agg: str, frames: List[int],
                    chars: List[int], coef: Tuple[float, float, float, float],
                    dp: Optional[int] = None) -> Optional[List[int]]:
    """Rows of the predicted-heaviest `_EXACT_BATCH_TOP_N` real micro-batches of every dp rank,
    ordered by their proxy value (the scan order). None when the selection is too large."""
    import torch
    a, b, c = coef[:3]
    dp = dp or _data_parallel_size(args)
    epochs, _ = _replay_epochs(args, n, group_size, dp)
    data_sharding = getattr(args, 'data_sharding', False)
    shuffle = getattr(args, 'train_dataloader_shuffle', True)
    pred = torch.tensor(frames, dtype=torch.float64) * a \
        + torch.tensor(chars, dtype=torch.float64) * b + c
    rows = set()
    for r in range(dp):
        B = _sampler_batches(n, group_size, dp, r, epochs, data_sharding, shuffle)
        if B.shape[0] == 0:
            continue
        score = pred[B].sum(1) if agg == 'sum' else pred[B].max(1).values
        k = min(_EXACT_BATCH_TOP_N, B.shape[0])
        top = torch.argsort(score, descending=True, stable=True)[:k]
        rows.update(int(i) for i in B[top].flatten().tolist())
    if len(rows) > _EXACT_BATCH_MAX_ROWS:
        return None
    return sorted(rows, key=lambda i: (-(a * frames[i] + b * chars[i]), i))


def _covered_batch_layout(args, lengths: List[int], group_size: int,
                          dp: Optional[int], dp_rank: Optional[int]):
    """``(B, covered, L, dp, dp_rank)`` for one dp rank's real micro-batches, or None.

    Shared preamble of the two batch-level peak paths: the batch matrix ``B`` from the
    deterministic sampler, ``covered`` marking the batches whose rows are ALL scanned, and
    ``L`` = the lengths as a tensor. None when the layout is unavailable
    (`group_by_length`, an unresolvable dp rank) or not a single batch is fully covered.
    """
    import torch
    if getattr(args, 'group_by_length', False):
        return None
    if dp_rank is None:
        dp_rank = _current_data_parallel_rank()
    if dp_rank is None:
        return None
    n = len(lengths)
    dp = dp or _data_parallel_size(args)
    epochs, _ = _replay_epochs(args, n, group_size, dp)
    B = _sampler_batches(n, group_size, dp, dp_rank, epochs,
                         getattr(args, 'data_sharding', False),
                         getattr(args, 'train_dataloader_shuffle', True))
    if B.shape[0] == 0:
        return None
    L = torch.tensor(lengths, dtype=torch.long)
    covered = (L[B] > 0).all(1)
    if not bool(covered.any()):
        return None
    return B, covered, L, dp, dp_rank


def _exact_batch_peak(args, hf_ds, lengths: List[int], group_size: int, agg: str,
                      dp: Optional[int] = None,
                      dp_rank: Optional[int] = None) -> Optional[int]:
    """Peak over THIS dp rank's real micro-batches whose rows are fully scanned.

    Returns None -- forcing the caller to the undershoot-only partial-coverage fallback --
    unless the calibration head is scanned, the proxy fit passes its quality gate, and this
    rank's predicted Top-N batches are entirely covered: a max over a SUBSET of real batches
    can only undershoot, so the Top-N coverage check is what makes the result trustworthy.
    """
    import torch
    # Cheap gates first: `_proxy_arrays` below rebuilds the dedup plan (~24s at 800k rows).
    layout = _covered_batch_layout(args, lengths, group_size, dp, dp_rank)
    if layout is None:
        return None
    B, covered, L, dp, dp_rank = layout
    n = len(lengths)
    proxy = _proxy_arrays(hf_ds, n)
    if proxy is None:
        return None
    frames, chars, todo, dup_of, sizes = proxy
    order1 = _order_tail_first(todo, sizes, {i: chars[i] for i in todo})
    calib = order1[:_EXACT_BATCH_CALIB_ROWS]
    coef = _fit_length_proxy(frames, chars, lengths, calib)
    if coef is None:
        return None
    a, b, c = coef[:3]
    pred = torch.tensor(frames, dtype=torch.float64) * a \
        + torch.tensor(chars, dtype=torch.float64) * b + c
    score = pred[B].sum(1) if agg == 'sum' else pred[B].max(1).values
    k = min(_EXACT_BATCH_TOP_N, B.shape[0])
    top = torch.argsort(score, descending=True, stable=True)[:k]
    if not bool(covered[top].all()):
        return None
    vals = L[B]
    sums = vals.sum(1) if agg == 'sum' else vals.max(1).values
    peak = int(sums[covered].max())
    logger.info(f'[pad_warmup] exact-batch: dp_rank={dp_rank}/{dp}, '
                f'{int(covered.sum())}/{B.shape[0]} real batches fully scanned '
                f'(proxy fit mape={coef[3]:.2%} on {len(calib)} calibration rows), peak={peak}.')
    return peak


def _covered_batch_peak(args, lengths: List[int], group_size: int, agg: str,
                        dp: Optional[int] = None,
                        dp_rank: Optional[int] = None) -> Optional[int]:
    """Max over THIS dp rank's real micro-batches whose rows are ALL scanned.

    Needs no proxy -- only the deterministic sampler layout and the scanned lengths. A max
    over a SUBSET of real batches can only undershoot, never overshoot: a partial warm-up
    may leave residual risk (the same as the feature being off for the missed batches) but
    can never invent a batch larger than anything the run will see, so a too-loose
    reservation -- the failure mode that OOMs an otherwise healthy run at step 0 -- is
    impossible here. This is the fallback when the exact-batch gates (proxy fit, Top-N
    coverage) fail.
    """
    layout = _covered_batch_layout(args, lengths, group_size, dp, dp_rank)
    if layout is None:
        return None
    B, covered, L, dp, dp_rank = layout
    vals = L[B]
    sums = vals.sum(1) if agg == 'sum' else vals.max(1).values
    peak = int(sums[covered].max())
    logger.info(f'[pad_warmup] exact-batch (partial coverage): dp_rank={dp_rank}/{dp}, '
                f'{int(covered.sum())}/{B.shape[0]} real batches fully scanned, peak={peak}.')
    return peak


def replay_peak_batch_length(args, hf_ds, lengths: List[int], group_size: int,
                             agg: str = 'sum', dp_rank: Optional[int] = None) -> Optional[int]:
    """Largest packed length over every micro-batch this run will execute.

    `MegatronPretrainingRandomSampler` is deterministic (seed = epoch), so replaying it yields
    exactly the batches the run will see. `agg` is ``'sum'`` for padding_free (the packed row is
    the concatenation of the samples) and ``'max'`` for the padded layout (the row length is the
    longest sample of the batch).

    With `dp_rank` given, only that rank's micro-batches are replayed: every rank owns its own
    device, so reserving the *global* peak on all of them pads 7 out of 8 ranks with a batch they
    will never see (measured here: a rank's own peak is only 64-100% of the global one).

    `lengths` must be complete: missing rows are the caller's business (the exact-batch chain).
    Substituting an invented value is rejected outright -- k>=2 fill values colliding in one
    batch inflate the peak ~linearly with mbs (measured 198%/296%/478% at mbs 2/4/8), and a
    too-loose reservation OOMs the step-0 warm-up batch of an otherwise healthy run.
    """
    total = len(lengths)
    if total <= 0:
        return None
    try:
        ds_len = len(hf_ds)
    except Exception:
        ds_len = None
    if ds_len is not None and ds_len != total:
        # An index shift would make `lengths[i]` describe the wrong sample.
        logger.warning(f'[pad_warmup] Disabled: the length array has {total} rows but the dataset '
                       f'has {ds_len}.')
        return None
    dp = _data_parallel_size(args)
    group_size = max(1, min(int(group_size or 1), total))

    if any(v <= 0 for v in lengths):
        logger.warning('[pad_warmup] replay called with unscanned rows; refusing '
                       '(missing rows are resolved by the exact-batch chain, not by replay).')
        return None

    ranks = list(range(dp)) if dp_rank is None else [dp_rank]
    data_sharding = getattr(args, 'data_sharding', False)
    shuffle = getattr(args, 'train_dataloader_shuffle', True)
    group_by_length = getattr(args, 'group_by_length', False)

    epochs, n_epochs = _replay_epochs(args, total, group_size, dp)
    if len(epochs) < n_epochs:
        # Evenly sampled epochs still cover the WHOLE run, including the last epoch. (v5
        # replayed epoch 0 only, but per-epoch shuffles differ -- measured per-rank peaks grew
        # ~24% over 60 epochs, so that fallback systematically underestimated the peak.)
        logger.warning(f'[pad_warmup] {n_epochs} epochs x {total} samples is too large to replay '
                       f'fully; replaying {len(epochs)} evenly sampled epochs instead.')

    if group_by_length:
        best = _replay_peak_via_sampler(hf_ds, lengths, group_size, dp, epochs, ranks, agg,
                                        data_sharding, shuffle, group_by_length)
        if best is None:
            return None
    else:
        best = _replay_peak_vectorized(lengths, total, group_size, dp, epochs, ranks, agg,
                                       data_sharding, shuffle)
    logger.info(f'[pad_warmup] Replayed {len(epochs)}/{n_epochs} epoch(s) for dp_rank={ranks}: '
                f'peak {agg} of {group_size} consecutive sample(s) = {best or None}.')
    return best or None


# The template is shared with the scan workers through fork (copy-on-write), because pickling it
# (e.g. via `datasets.map(num_proc=...)`) drops the initialized processor.
_SCAN_TEMPLATE = None


def _scan_row(row: Dict[str, Any]) -> int:
    """Map one dataset row to its token length; 0 for failed samples."""
    try:
        encoded = _SCAN_TEMPLATE.encode(row, return_length=True)
        lengths = encoded.get('lengths') or [len(encoded['input_ids'])]
        return max(lengths)
    except Exception:
        return 0


# Columns whose content decides how many tokens the *media* part of a sample expands to.
_MM_KEY_COLUMNS = ('videos', 'images', 'audios')


def _freeze_media(value: Any) -> Optional[Tuple]:
    """Freeze a media payload reference into a hashable, canonical form.

    Media columns are not always a plain string or a list of strings: HF datasets decode media
    features to dicts (``{'path': ..., 'bytes': ...}``), some datasets nest per-turn lists, and
    mixed structures appear after column mapping. Recursively freezing dicts / lists / sets
    (dict items sorted by key) makes every one of those hashable, so dedup *works* for them
    instead of throwing `unhashable type` and falling back to a full scan.

    Returns ``None`` when the identity cannot be established *exactly*, and such rows are then
    all encoded. Only str / numbers are accepted as leaves, deliberately: ``str()`` of anything
    else is not an identity. A ``PIL.Image`` reprs to ``<PIL.Image.Image ... at 0x7f...>`` --
    an address that the allocator reuses once the row is freed, and a numpy array reprs
    *truncated* (``[1 2 3 ... 998 999]``). Either would merge two different media payloads into
    one group, which silently corrupts the peak; refusing to dedup them only costs time.
    """
    if isinstance(value, dict):
        items = []
        for k, v in value.items():
            fv = _freeze_media(v)
            if fv is None:
                return None
            items.append((str(k), fv))
        return ('dict', tuple(sorted(items)))
    if isinstance(value, (list, tuple)):
        items = []
        for v in value:
            fv = _freeze_media(v)
            if fv is None:
                return None
            items.append(fv)
        return ('list', tuple(items))
    if isinstance(value, (set, frozenset)):
        items = []
        for v in value:
            fv = _freeze_media(v)
            if fv is None:
                return None
            items.append(fv)
        return ('list', tuple(sorted(items)))
    if isinstance(value, str):
        return ('str', value)
    if isinstance(value, (int, float, bool)) or value is None:
        # Tag with the type so that 2 and '2' cannot collide.
        return (type(value).__name__, str(value))
    # bytes / PIL.Image / np.ndarray / anything else: no exact cheap identity -> do not dedup.
    return None


def _media_key(value_of) -> Optional[Tuple]:
    """Identity of a row's multimodal payload across `_MM_KEY_COLUMNS`.

    `value_of(col)` yields the row's value for one media column (None when absent). Rows
    sharing the key decode to exactly the same media tokens, so only one of them has to be
    decoded: this dataset is 485 unique videos x 7 QA pairs, and a full scan decoded every
    single video 7 times. A row whose payload cannot be frozen exactly (raw bytes) gets no
    key and is always encoded.
    """
    parts = []
    for col in _MM_KEY_COLUMNS:
        value = value_of(col)
        if not value:
            continue
        frozen = _freeze_media(value)
        if frozen is None:
            return None
        parts.append((col, frozen))
    return tuple(parts) or None


def _row_media_key(row: Dict[str, Any]) -> Optional[Tuple]:
    """`_media_key` over a row dict."""
    return _media_key(row.get)


def _row_text(row: Dict[str, Any]) -> str:
    """The textual part of a row, used to correct a duplicate by its own text length.

    Message shapes seen in the wild: a list of message dicts (standard), ONE bare dict
    (single-turn rows), a bare string, ShareGPT-style ``from``/``value`` keys, and
    ``content`` as a string, a list of part dicts, or a list of plain strings."""
    messages = row.get('messages')
    if isinstance(messages, dict):
        messages = [messages]
    elif isinstance(messages, str):
        messages = [messages]
    out = []
    for message in (messages or []):
        if isinstance(message, str):
            out.append(message)
            continue
        if not isinstance(message, dict):
            continue
        content = message.get('content')
        if content is None:
            content = message.get('value')   # ShareGPT-style rows
        if isinstance(content, str):
            out.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, str):
                    out.append(part)
                elif isinstance(part, dict) and part.get('type', 'text') == 'text':
                    out.append(part.get('text') or part.get('content') or '')
    return ''.join(out)


def _is_frame_sequence(value) -> bool:
    """A pure list of 2+ strings: one video's frame paths. Its decode-cost proxy is the
    frame COUNT -- statting every frame on NFS can dominate the whole plan build."""
    return isinstance(value, (list, tuple)) and len(value) > 1 \
        and all(isinstance(x, str) for x in value)


@functools.lru_cache(maxsize=262144)
def _stat_size(path: str) -> int:
    """`os.stat().st_size`, or -1; cached per process (the size only decides scan ORDER)."""
    try:
        return os.stat(path).st_size
    except (OSError, ValueError, TypeError):
        return -1


def _prewarm_stat(paths, workers: int = 32) -> None:
    """Stat unique paths with a thread pool, filling `_stat_size`'s cache.

    stat() releases the GIL, so on NFS (~0.35 ms/stat) 32 threads cut 80k stats from ~28 s
    to ~1 s. On a local disk the pool overhead costs ~2 s once -- negligible, and small
    path sets skip the pool entirely.
    """
    uniq = [p for p in dict.fromkeys(paths) if isinstance(p, str) and p]
    if len(uniq) < 256:
        return
    t0 = time.perf_counter()
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(workers, len(uniq))) as ex:
        list(ex.map(_stat_size, uniq))
    logger.info(f'[pad_warmup] stat() x{len(uniq)} prewarmed in {time.perf_counter() - t0:.2f}s.')


def _strings_size(value) -> int:
    """Decode-cost proxy: frame count for a frame sequence, otherwise summed `stat()` sizes.

    Recursive: the frame-count shortcut applies at ANY nesting depth. A nested shape like
    ``videos=[[f1..f180]]`` (an outer len-1 list) must NOT fall through to per-frame
    stats -- millions of cold NFS stats (~22ms each) mean tens of minutes per rank for a
    proxy that was always meant to be the frame count. Return -1 if any required stat
    fails, and 0 when there are no strings."""
    if isinstance(value, str):
        return _stat_size(value)
    if isinstance(value, dict):
        total = 0
        for v in value.values():
            s = _strings_size(v)
            if s < 0:
                return -1
            total += s
        return total
    if isinstance(value, (list, tuple, set)):
        if _is_frame_sequence(value):
            return len(value)
        total = 0
        for v in value:
            s = _strings_size(v)
            if s < 0:
                return -1
            total += s
        return total
    return 0


def _stat_bearing_strings(value):
    """Yield the strings whose stat() the size proxy actually reads; frame sequences are
    proxied by their count at any nesting depth, so their paths are never stat'ed."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _stat_bearing_strings(v)
    elif isinstance(value, (list, tuple, set)):
        if not _is_frame_sequence(value):
            for v in value:
                yield from _stat_bearing_strings(v)


def _row_media_size(row) -> int:
    """Decode-cost proxy for one row: media bytes, or frame count for a path sequence.

    0 for text-only rows, -1 when the size cannot be determined (remote URL, raw bytes, a
    missing/unreadable file). The proxy only needs to be *monotone-ish* with the decoded
    token length: bigger file ≈ longer video ≈ more frames ≈ more tokens.
    """
    if not isinstance(row, dict):
        return -1
    total = 0
    for col in _MM_KEY_COLUMNS:
        value = row.get(col)
        if not value:
            continue
        sz = _strings_size(value)
        if sz < 0:
            return -1
        total += sz
    return total


# ---------------------------------------------------------------------------
# Columnar plan building
# ---------------------------------------------------------------------------
# `_scan_dedup_plan` only needs the media columns, so on a columnar dataset (HF `Dataset`)
# it reads them as WHOLE COLUMNS instead of random-accessing every row: one vectorized
# arrow read per column instead of n x (10-50us) row touches. For Image/Video features
# with decode=True the row access would additionally DECODE every file (minutes, and the
# resulting PIL objects cannot be frozen into a dedup key at all), so those columns are
# read undecoded -- straight off the arrow struct's `path` child when possible, else via a
# metadata-only `cast_column(decode=False)` view.

# Marker standing in for an embedded byte payload after the payload itself is dropped: any
# `bytes` object makes `_freeze_media` refuse the value, so embedded-bytes rows keep the
# row-wise "never dedup by content" behaviour without pinning megabytes in the plan.
_BYTES_MARKER = b'\x00pad_warmup\x00'


def _media_cell(p, bl) -> Tuple[Any, int]:
    """(value, size) for one struct media cell, given its `path` and embedded byte LENGTH.

    A usable path carries the identity alone (the byte payload is dropped -- that is what
    restores dedup for HF Image datasets that embed the file content: row-wise the bytes
    poison every key, and `_freeze_media` must refuse them). Bytes-only cells keep a marker
    so `_freeze_media` still refuses them (a length is not an identity), with the byte
    length as the size proxy. An empty cell formats to None row-wise, so it is None here.
    """
    if p:
        s = _stat_size(p)
        if s >= 0 or not bl:
            return {'path': p, 'bytes': None}, s
        # Path unreadable but content embedded: identity is the bytes -> no dedup.
        return {'path': None, 'bytes': _BYTES_MARKER}, int(bl)
    if bl:
        return {'path': None, 'bytes': _BYTES_MARKER}, int(bl)
    return None, 0


def _is_decoding_feature(feature) -> bool:
    """HF Image/Video-like features carry `decode=True`; Sequence/list wraps one level."""
    if feature is None:
        return False
    if getattr(feature, 'decode', None):
        return True
    inner = getattr(feature, 'feature', None)  # Sequence(Image(...))
    if inner is not None:
        return bool(getattr(inner, 'decode', None))
    if isinstance(feature, (list, tuple)) and len(feature) == 1:  # legacy [Image(...)]
        return bool(getattr(feature[0], 'decode', None))
    return False


def _undecoded_view(hf_ds, cols):
    """A view of `hf_ds` whose decoding media features are cast to `decode=False`.

    `cast_column` is metadata-only (no data copy, indices preserved), so this is cheap.
    None when any cast fails -- the caller then falls back to the row-wise path, whose
    behaviour on such a dataset is exactly today's."""
    features = getattr(hf_ds, 'features', None) or {}
    view = hf_ds
    for col in cols:
        feature = features.get(col)
        if feature is None:
            continue
        try:
            if getattr(feature, 'decode', None):
                view = view.cast_column(col, type(feature)(decode=False))
            else:
                inner = getattr(feature, 'feature', None)
                if inner is not None and getattr(inner, 'decode', None):
                    view = view.cast_column(col, type(feature)(type(inner)(decode=False)))
                elif isinstance(feature, (list, tuple)) and len(feature) == 1 \
                        and getattr(feature[0], 'decode', None):
                    view = view.cast_column(col, [type(feature[0])(decode=False)])
        except Exception:
            return None
    return view


def _arrow_struct_media(arr) -> Optional[list]:
    """[(value, size)] per row of a struct<path: string, bytes: binary> arrow array.

    Only the `path` child is read out, plus the byte LENGTH of the `bytes` child -- the
    embedded payload itself is never materialized. Values are shaped exactly like the
    decode=False row payload so `_freeze_media` treats both access paths identically. Any
    shape beyond the standard two fields -> None, and the caller falls back to a
    full-fidelity column fetch."""
    try:
        names = [f.name for f in arr.type]
        if 'path' not in names or not set(names) <= {'path', 'bytes'}:
            return None
        paths = arr.field('path').to_pylist()
        _prewarm_stat(p for p in paths if p)
        if 'bytes' in names:
            import pyarrow.compute as pc
            blens = pc.binary_length(arr.field('bytes')).to_pylist()
        else:
            blens = [None] * len(paths)
    except Exception:
        return None
    return [_media_cell(p, bl) for p, bl in zip(paths, blens)]


def _arrow_chunk_media(chunk) -> Optional[list]:
    """[(value, size)] for one chunk of a media column: struct rows directly, list rows
    (Sequence(Image)) itemwise. Null list entries come out as None (falsy -> never keyed),
    matching the row value."""
    import pyarrow as pa
    t = chunk.type
    if pa.types.is_struct(t):
        return _arrow_struct_media(chunk)
    if not (pa.types.is_list(t) or pa.types.is_large_list(t)):
        return None
    child = chunk.values  # the unsliced child; `offsets` below index into it absolutely
    if not pa.types.is_struct(child.type):
        return None
    cells = _arrow_struct_media(child)
    if cells is None:
        return None
    offsets = chunk.offsets.to_pylist()
    nulls = chunk.is_null().to_pylist()
    rows = []
    for k, (a, b) in enumerate(zip(offsets, offsets[1:])):
        if nulls[k]:
            rows.append((None, 0))
            continue
        vals = [v for v, _ in cells[a:b]]
        size = 0
        for _, s in cells[a:b]:
            if s < 0:
                size = -1
                break
            size += s
        rows.append((vals, size if vals else 0))
    return rows


def _arrow_column(hf_ds, col, n: int, chunk_fn) -> Optional[list]:
    """Per-LOGICAL-row values of an arrow-backed column; None when unavailable.

    `chunk_fn(chunk) -> list | None` maps one physical chunk. A select()/filter()/shuffle()
    view keeps an indices mapping over the physical table; it is applied here so the result
    follows the logical row order."""
    try:
        table = getattr(hf_ds, 'data', None)
        if table is None or not hasattr(table, 'column'):
            return None
        chunks = getattr(table.column(col), 'chunks', None)
        if chunks is None:
            return None
        values = []
        for chunk in chunks:
            v = chunk_fn(chunk)
            if v is None:
                return None
            values.extend(v)
        if len(values) != table.num_rows:
            return None
        indices = getattr(hf_ds, '_indices', None)
        if indices is not None:
            idx = indices.column(0).to_pylist()
            if len(idx) != n:
                return None
            values = [values[int(i)] for i in idx]
        return values if len(values) == n else None
    except Exception:
        return None


def _arrow_media_column(hf_ds, col, n: int) -> Optional[Tuple[list, list]]:
    """(values, sizes) per LOGICAL row of an arrow-backed media column, or None.

    Reading the struct children straight off the arrow table is what makes the plan cheap
    for HF Image/Video features: no decode, and the embedded byte payload is never
    materialized (only its length is)."""
    rows = _arrow_column(hf_ds, col, n, _arrow_chunk_media)
    if rows is None:
        return None
    values, sizes = zip(*rows) if rows else ([], [])
    return list(values), list(sizes)


def _normalize_media_value(v) -> Tuple[Any, int]:
    """(value, size) for one column-fetched media value.

    The value keeps the exact shape `_freeze_media` would see row-wise, with one deliberate
    reduction: when a usable `path` exists the `bytes` payload is dropped and the path
    alone carries the identity. That is what RESTORES dedup for HF Image datasets that
    embed the file content -- row-wise the bytes poison every key (`_freeze_media` refuses
    them) and every single row gets encoded. When only bytes exist, the marker keeps the
    "never dedup by content" behaviour and the byte length serves as the size proxy."""
    if isinstance(v, dict) and ('path' in v or 'bytes' in v):
        b = v.get('bytes')
        value, s = _media_cell(v.get('path'), len(b) if b is not None else None)
        # Non-payload fields (e.g. fps) stay in the key; the marker form needs none.
        extra = {k: x for k, x in v.items() if k not in ('path', 'bytes')}
        if isinstance(value, dict) and value.get('bytes') is None and extra:
            value.update(extra)
        return value, s
    if isinstance(v, (list, tuple)):
        if _is_frame_sequence(v):
            return list(v), len(v)
        vals, total = [], 0
        for x in v:
            nx, s = _normalize_media_value(x)
            vals.append(nx)
            if s < 0:
                total = -1
            elif total >= 0:
                total += s
        return vals, total
    return v, _strings_size(v)


def _arrow_plain_column(hf_ds, col, n: int) -> Optional[list]:
    """A whole plain column read straight off the arrow table (vectorized). Only for
    non-media-struct columns: a struct<path, bytes> column would materialize the embedded
    payloads and must go through `_arrow_media_column`."""
    return _arrow_column(hf_ds, col, n, lambda chunk: chunk.to_pylist())


def _fetch_media_column(ds, col, n: int, allow_arrow: bool = True
                        ) -> Optional[Tuple[list, list]]:
    """(values, sizes) from one whole-column fetch; None when columnar access fails.

    `allow_arrow=False` forces the dataset's own formatting (used for the decode=False
    cast view, whose whole point is the reformatted values)."""
    raw = _arrow_plain_column(ds, col, n) if allow_arrow else None
    if raw is None:
        try:
            raw = list(ds[col])
        except Exception:
            return None
        if len(raw) != n:
            return None
    # Prewarm only the paths that will actually be stat'ed: a frame sequence is proxied
    # by its frame count and needs no per-frame stat -- at ANY nesting depth.
    _prewarm_stat(s for v in raw for s in _stat_bearing_strings(v))
    values, sizes = [], []
    for v in raw:
        nv, s = _normalize_media_value(v)
        values.append(nv)
        sizes.append(s)
    return values, sizes


def _media_columns(hf_ds, n: int) -> Optional[Tuple[Dict[str, list], Dict[int, int]]]:
    """Column-wise `({col: [payload per row]}, {row: byte-size proxy})`; None -> read rows.

    `values[col][i]` is shaped exactly like the row-wise media value, so the dedup keys are
    identical on both paths. Decoding features (Image/Video with decode=True) are read
    undecoded: off the arrow struct's `path` child when the table is reachable (the byte
    payload is never even materialized), else through a metadata-only decode=False cast --
    which is also what restores dedup for those datasets (row-wise they arrive as PIL
    objects, which can never be frozen into a key)."""
    column_names = getattr(hf_ds, 'column_names', None)
    if not column_names:
        return None
    features = getattr(hf_ds, 'features', None) or {}
    cols = [c for c in _MM_KEY_COLUMNS if c in column_names]
    values: Dict[str, list] = {}
    sizes: Dict[int, int] = {}
    view = None
    for col in cols:
        # Struct media columns first: zero decode AND the embedded byte payload is never
        # materialized (only its length). Plain columns take the vectorized arrow read
        # inside `_fetch_media_column`.
        got = _arrow_media_column(hf_ds, col, n)
        if got is None:
            if _is_decoding_feature(features.get(col)):
                if view is None:
                    view = _undecoded_view(hf_ds, cols)
                    if view is None:
                        return None  # a decoding column we cannot undecode -> row-wise
                got = _fetch_media_column(view, col, n, allow_arrow=False)
            else:
                got = _fetch_media_column(hf_ds, col, n)
        if got is None:
            return None
        values[col], cs = got
        for i, s in enumerate(cs):
            prev = sizes.get(i, 0)
            sizes[i] = -1 if prev < 0 or s < 0 else prev + s
    return values, sizes


def _columnar_media_key(values: Dict[str, list], i: int) -> Optional[Tuple]:
    """`_media_key` over per-column values (identical column order/shape)."""
    return _media_key(lambda col: values[col][i] if col in values else None)


def _text_char_sizes(hf_ds, todo: List[int]) -> Dict[int, int]:
    """Text character counts from one column read; only used to break equal-media ties.

    Always read through the dataset's OWN formatting -- never the raw arrow table: there
    `messages` can be JSON *strings* that only the formatted view restores to list-of-dict,
    so a raw read silently extracted zero (or garbage) text on a real 80k-row dataset --
    equal-frame ties lost their tie-break, the calibration head landed on the wrong rows
    (mape 1.98% -> 8.63%), and the resolved peak undershot by 3.1% (measured). The
    formatted read costs 19.3s at 80k rows, negligible against the scan itself."""
    try:
        text_col = list(hf_ds['messages'])
        if len(text_col) != len(hf_ds):
            return {}
    except Exception:
        return {}
    out = {}
    for i in todo:
        try:
            out[i] = len(_row_text({'messages': text_col[i]}))
        except Exception:
            out[i] = 0
    return out


def _order_tail_first(todo: List[int], sizes: Dict[int, int],
                      text_chars: Optional[Dict[int, int]] = None) -> List[int]:
    """Order likely-longest rows first: media size descending, then text size for ties.

    This makes the scan an *anytime* algorithm for the peak: at ANY cutoff the unscanned
    rows are the short ones, so the scanned subset still holds the length distribution's
    upper tail and `max(scanned)` keeps tracking the true peak. Index order does not have
    this property (datasets are often length-clustered on disk) and neither does random
    order (slow decodes = long rows finish last, biasing any deadline cutoff downwards).
    Unknown media sizes go first because they may hide the long tail.
    """
    text_chars = text_chars or {}
    return sorted(todo, key=lambda i: (0 if sizes.get(i, 0) < 0 else 1,
                                       -max(sizes.get(i, 0), 0), -text_chars.get(i, 0), i))


def _scan_dedup_plan(hf_ds, n: int) -> Tuple[List[int], Dict[int, int], Dict[int, int]]:
    """`(rows_to_encode, duplicate -> representative, row -> media_size)`; rows without
    media are all encoded.

    Column-wise when the dataset supports it (see `_media_columns`): the per-rank O(n) row
    random-access loop becomes one vectorized read per media column, and decoding media
    features are read undecoded -- which both removes the per-rank decode of the whole
    dataset and restores dedup for HF Image/Video-feature datasets. Per-row otherwise.

    A row whose key cannot be computed (odd payload type, a broken row) is simply encoded --
    one bad row must not cost the whole plan.
    """
    representatives = {}
    todo = []
    dup_of = {}
    cols = _media_columns(hf_ds, n)
    if cols is not None:
        values, sizes = cols
        for i in range(n):
            try:
                key = _columnar_media_key(values, i)
            except Exception:
                key = None
            if key is None:
                todo.append(i)
                continue
            rep = representatives.get(key)
            if rep is None:
                representatives[key] = i
                todo.append(i)
            else:
                dup_of[i] = rep
        # Only `todo` rows carry a size (same shape as the row-wise path, so identical
        # data yields identical plan digests on either path).
        return todo, dup_of, {i: sizes.get(i, 0) for i in todo}
    sizes = {}
    for i in range(n):
        try:
            row = hf_ds[i]
        except Exception:
            row = None
        try:
            key = _row_media_key(row) if row is not None else None
        except Exception:
            key = None
        if key is None:
            todo.append(i)
            sizes[i] = _row_media_size(row)
            continue
        rep = representatives.get(key)
        if rep is None:
            representatives[key] = i
            todo.append(i)
            sizes[i] = _row_media_size(row)
        else:
            dup_of[i] = rep
    return todo, dup_of, sizes


def _expand_dedup(hf_ds, todo: List[int], values: List[int], dup_of: Dict[int, int],
                  n: int) -> List[int]:
    """Rebuild the full length array from a deduplicated scan.

    A sample's length is `media_tokens + text_tokens + template_constant`, and within a group
    the media part is identical (same file, same pixel/frame config), so a duplicate differs
    only by its own text: correct it by exactly that delta. Representative rows keep their
    exact encoded value, and a group whose representative failed stays a failure (0) -- the
    conservative direction.
    """
    base = dict(zip(todo, values))
    lengths = [int(base.get(i, 0)) for i in range(n)]
    if not dup_of:
        return lengths
    tokenizer = getattr(_SCAN_TEMPLATE, 'tokenizer', None)
    if tokenizer is None:
        for i, rep in dup_of.items():
            lengths[i] = int(base.get(rep, 0))
        return lengths
    # The text correction reads only the messages column -- fetch it whole. Per-row
    # `hf_ds[i]` would additionally DECODE the media features of every duplicated row
    # (the same cost the columnar plan building just removed).
    text_col = None
    if 'messages' in (getattr(hf_ds, 'column_names', None) or ()):
        try:
            text_col = list(hf_ds['messages'])
            if len(text_col) != n:
                text_col = None
        except Exception:
            text_col = None
    text_len = {}
    for i in set(dup_of) | set(dup_of.values()):
        try:
            row = {'messages': text_col[i]} if text_col is not None else hf_ds[i]
            text_len[i] = len(tokenizer(_row_text(row), add_special_tokens=False)['input_ids'])
        except Exception:
            text_len[i] = 0
    for i, rep in dup_of.items():
        value = int(base.get(rep, 0))
        if value > 0:
            value += text_len.get(i, 0) - text_len.get(rep, 0)
        lengths[i] = max(0, value)
    return lengths


def _dedup_plan_or_all(hf_ds, n: int) -> Tuple[List[int], Dict[int, int], Dict[int, int]]:
    """The dedup plan, degrading to "encode every row" when the plan itself cannot be built."""
    try:
        todo, dup_of, sizes = _scan_dedup_plan(hf_ds, n)
        if todo:
            return todo, dup_of, sizes
    except Exception as e:
        logger.warning(f'[pad_warmup] Cannot build the dedup plan ({e}); scanning every row.')
    return list(range(n)), {}, {}


def _scan_row_indexed(pair) -> Tuple[int, int]:
    """`(index, token_length)`; index-passing variant of `_scan_row` for `imap_unordered`."""
    idx, row = pair
    return idx, _scan_row(row)


def _scan_rows(rows: List[Dict[str, Any]], num_proc: int,
               deadline_s: Optional[float] = None) -> List[int]:
    """Encode `rows` with the fork pool; one value per row, 0 for failed samples.

    Bounded in time twice, because "slow" must never become "the job dies on a collective
    timeout" (a stuck rank keeps every other rank waiting inside the result all_gather):

    * **stall timeout** -- no result for `_SCAN_ROW_TIMEOUT_S`-scaled time means a decode is
      stuck (a corrupt mp4 can block ffmpeg/decord *forever*; a worker killed by the host
      OOM-killer looks the same), so the pool is torn down and the missing rows stay 0;
    * **deadline** -- the whole call stops at `deadline_s` seconds.

    Either bound leaves the remaining values at 0, i.e. scan failures: the caller's mode
    chain (exact-batch -> exact-batch-partial -> off) then resolves the peak from
    what was actually covered.
    """
    if not rows:
        return []
    import multiprocessing as mp
    n = len(rows)
    # chunksize=1 deliberately: some CPython builds (e.g. 3.11.6 TencentOS, and upstream
    # 3.13+) make `imap_unordered` return a plain *generator* when chunksize > 1, which has
    # no `.next(timeout=...)` -- and the timeout is what makes a stuck decode detectable.
    # The per-row IPC overhead is negligible next to a decode.
    values = [0] * n
    t0 = time.perf_counter()
    done = 0
    last_progress = t0
    next_report = t0 + 60
    with mp.get_context('fork').Pool(num_proc, initializer=_scan_worker_init) as pool:
        it = pool.imap_unordered(_scan_row_indexed, enumerate(rows), chunksize=1)
        while done < n:
            try:
                idx, v = it.next(timeout=60)
            except mp.TimeoutError:
                now = time.perf_counter()
                if deadline_s is not None and now - t0 > deadline_s:
                    pool.terminate()
                    logger.warning(f'[pad_warmup] Scan deadline ({deadline_s}s) reached at '
                                   f'{done}/{n} rows; the rest count as failed samples.')
                    break
                if now - last_progress > _SCAN_ROW_TIMEOUT_S:
                    pool.terminate()
                    logger.warning(f'[pad_warmup] A row stalled the scan for {now - last_progress:.0f}s '
                                   f'(corrupt media? killed worker?); {done}/{n} rows done, '
                                   f'the rest count as failed samples.')
                    break
                if now >= next_report:
                    logger.info(f'[pad_warmup] Scanning: {done}/{n} rows, '
                                f'{now - t0:.0f}s elapsed.')
                    next_report = now + 60
                continue
            values[idx] = int(v) if v else 0
            done += 1
            now = time.perf_counter()
            last_progress = now
            if now >= next_report:
                logger.info(f'[pad_warmup] Scanning: {done}/{n} rows, {now - t0:.0f}s elapsed.')
                next_report = now + 60
            if deadline_s is not None and now - t0 > deadline_s:
                pool.terminate()
                logger.warning(f'[pad_warmup] Scan deadline ({deadline_s}s) reached at '
                               f'{done}/{n} rows; the rest count as failed samples.')
                break
    return values


def _exact_batch_scan_plan(args, n: int, todo: List[int], dup_of: Dict[int, int],
                           sizes: Dict[int, int], chars_map: Dict[int, int],
                           order1: List[int]):
    """``(calib_rows, select_fn)`` for the two-pass exact-batch scan, or None to keep the plain
    single-pass tail-first scan. ``select_fn(coef)`` returns the pass-2 representative rows
    (proxy-ordered, calibration rows excluded)."""
    if args is None or getattr(args, 'group_by_length', False) or not order1:
        return None
    group_size = max(1, int(getattr(args, 'micro_batch_size', 1) or 1))
    agg = 'sum' if getattr(args, 'padding_free', False) else 'max'
    frames, chars = _proxy_from_plan(n, dup_of, sizes, chars_map)
    calib = list(order1[:_EXACT_BATCH_CALIB_ROWS])
    calib_set = set(calib)

    def select_fn(coef):
        raw = _top_batch_rows(args, n, group_size, agg, frames, chars, coef)
        if raw is None:
            return None
        reps, seen = [], set(calib_set)
        for i in raw:
            rep = dup_of.get(i, i)
            if rep not in seen:
                seen.add(rep)
                reps.append(rep)
        return reps

    return calib, select_fn


def _exact_batch_calib_budget() -> float:
    """Time budget for the calibration head: a bounded slice of the overall scan deadline."""
    return max(30.0, min(300.0, _SCAN_DEADLINE_S / 6))


def _exact_batch_fit_select(plan, n: int, dup_of: Dict[int, int], sizes: Dict[int, int],
                            chars_map: Dict[int, int], calib_values: Dict[int, int]):
    """Fit the length proxy on the calibration head's scanned values, run the pass-2
    selection. Returns ``(sel, coef)``; ``sel`` None when the fit fails its quality gate
    (the caller then falls back to the plain tail-first scan)."""
    calib, select_fn = plan
    labels = [0] * n
    for i, v in calib_values.items():
        labels[i] = v
    frames, chars = _proxy_from_plan(n, dup_of, sizes, chars_map)
    coef = _fit_length_proxy(frames, chars, labels, calib)
    return (select_fn(coef) if coef else None), coef


def _scan_lengths(hf_ds, n: int, num_proc: int, args=None) -> List[int]:
    """Encode rows with a fork pool; rows sharing a media payload are decoded once.

    Two passes when the exact-batch plan applies: a small calibration head fits the length
    proxy, then only the rows of the predicted-heaviest REAL micro-batches are scanned. Any
    failure degrades to the plain tail-first scan, whose deadline cutoff still leaves the
    length distribution's upper tail in the scanned subset.
    """
    todo, dup_of, sizes = _dedup_plan_or_all(hf_ds, n)
    chars_map = _text_char_sizes(hf_ds, list(range(n)))
    todo = _order_tail_first(todo, sizes, chars_map)
    if len(todo) < n:
        logger.info(f'[pad_warmup] Scanning {len(todo)}/{n} samples (num_proc={num_proc}, '
                    f'dedup by media payload)...')
    else:
        logger.info(f'[pad_warmup] Scanning {n} samples (num_proc={num_proc})...')
    t0 = time.perf_counter()
    plan = _exact_batch_scan_plan(args, n, todo, dup_of, sizes, chars_map, todo)
    if plan is not None:
        calib, _ = plan
        values = _scan_rows([hf_ds[i] for i in calib], num_proc,
                            deadline_s=_exact_batch_calib_budget())
        sel, coef = _exact_batch_fit_select(plan, n, dup_of, sizes, chars_map,
                                            dict(zip(calib, values)))
        if sel is not None:
            logger.info(f'[pad_warmup] exact-batch scan: calibration={len(calib)} rows '
                        f'(proxy fit mape={coef[3]:.2%}), scanning {len(sel)}/{len(todo)} '
                        f'selected rows (num_proc={num_proc})...')
            remaining = max(0.0, _SCAN_DEADLINE_S - (time.perf_counter() - t0))
            values2 = _scan_rows([hf_ds[i] for i in sel], num_proc, deadline_s=remaining)
            rows = calib + sel
            return _expand_dedup(hf_ds, rows, values + values2, dup_of, n)
        logger.info('[pad_warmup] exact-batch scan unavailable; falling back to the '
                    'tail-first scan.')
    remaining = max(0.0, _SCAN_DEADLINE_S - (time.perf_counter() - t0))
    values = _scan_rows([hf_ds[i] for i in todo], num_proc, deadline_s=remaining)
    return _expand_dedup(hf_ds, todo, values, dup_of, n)


# ---------------------------------------------------------------------------
# Length scan and its on-disk cache
# ---------------------------------------------------------------------------
# Bumping this invalidates every existing cache entry at once.
# v3: `lengths` is a base64-encoded little-endian int32 array instead of a JSON list --
# at 10M rows a JSON list is ~60 MB of text, seconds to parse per rank and >10x memory
# amplification; the binary blob decodes in milliseconds into a flat buffer.
# Deliberately still 3: `_dataset_fingerprint` is unchanged, so entries written by the
# previous release stay valid. Any edit to that function changes every key and costs a full
# re-scan (~105 min measured on 80450 videos) -- that price has to buy something bigger than
# the blind spot documented there.
_LENGTH_CACHE_VERSION = 3
# Old cache entries are pruned on every store, keeping only the newest few: the cache key
# changes with the dataset/config, so entries would otherwise pile up forever.
_LENGTH_CACHE_KEEP = 8
# Batch-level tail-first ("exact-batch"): rather than ESTIMATING the peak of a partially
# scanned dataset from synthetic batch combinations, replay this dp rank's REAL micro-batches,
# rank them by a calibrated decode-free proxy (tokens ~ a*frames + b*text_chars + c), and scan
# exactly the rows of the predicted-heaviest handful of batches. The peak is then the max over
# the fully scanned REAL batches: per-dp-rank by construction, exact whenever the proxy keeps
# the true peak batch inside the Top-N (measured 0.00% error at Top-N=200 for mbs 1/2/4/8 on
# the reference dataset), and identical to the `exact` replay at full coverage. The scan runs
# in two passes: a small calibration head (fits the proxy), then the selected batch rows.
_EXACT_BATCH_CALIB_ROWS = 100
# Fewer scanned calibration rows than this -> the proxy fit cannot be trusted.
_EXACT_BATCH_MIN_CALIB = 50
# Quality gate: in-sample MAPE of the proxy fit on the calibration rows. Beyond it the proxy
# does not describe this dataset (mixed text-only rows, per-resolution images, ...), so the
# selection would be arbitrary and the estimate could UNDERshoot -- fall back instead.
_EXACT_BATCH_MAX_MAPE = 0.10
# Predicted-heaviest batches per dp rank whose rows get the exact scan. 200 was measured to
# leave a >30x margin on the worst rank position of the true peak batch (mbs=1: @6).
_EXACT_BATCH_TOP_N = 200
# Beyond this selection size the plain tail-first scan is the cheaper strategy.
_EXACT_BATCH_MAX_ROWS = 24_576
# Environment variables that decide how many tokens one multimodal (image/video) sample expands to.
_MM_LENGTH_ENV_VARS = ('MAX_PIXELS', 'MIN_PIXELS', 'VIDEO_MAX_PIXELS', 'VIDEO_MIN_PIXELS',
                       'VIDEO_TOTAL_PIXELS', 'FPS', 'FPS_MAX_FRAMES')
# Arguments that decide how long a sample is *after* `template.encode`.
#
# Deliberately NOT in the key (they only decide which samples land in the same micro-batch and in
# which order, never how long a single sample is): `train_iters`, `num_train_epochs`,
# `global_batch_size`, `micro_batch_size`, `data_parallel_size` / `data_parallel_rank`,
# `data_sharding`, `train_dataloader_shuffle`, `group_by_length`, `seq_length`,
# `dataset_num_proc`, `output_dir`.
_LENGTH_TEMPLATE_ARGS = ('template', 'system', 'max_length', 'truncation_strategy', 'max_pixels',
                         'agent_template', 'norm_bbox', 'use_chat_template', 'padding_free',
                         'template_backend', 'response_prefix', 'enable_thinking',
                         'preserve_thinking', 'add_non_thinking_prefix')


def _swift_version() -> str:
    try:
        from swift import __version__
        return str(__version__)
    except Exception:
        return 'unknown'


def _state_dir() -> Optional[str]:
    """Directory for pad-warmup's on-disk state: the length cache and the one-shot warm-up flag.

    It defaults to the working directory instead of ``$HOME``: both are throwaway artifacts, and
    inside the workspace they disappear together with it rather than piling up forever.
    """
    d = os.environ.get('SWIFT_PAD_WARMUP_CACHE_DIR') or os.path.join(os.getcwd(), '.pad_warmup_cache')
    try:
        os.makedirs(d, exist_ok=True)
    except Exception as e:
        logger.warning(f'[pad_warmup] Cannot create {d}: {e}')
        return None
    return d


def _cache_dir() -> Optional[str]:
    """`_state_dir()`, or None when the length cache is switched off."""
    if os.environ.get('SWIFT_PAD_WARMUP_CACHE', '1').lower() in ('0', 'off', 'false', 'no'):
        return None
    return _state_dir()


def _dataset_fingerprint(args) -> List[Any]:
    """Fingerprint the data *content* (size + mtime_ns), so editing a local file invalidates the
    cache. A remote dataset is identified by its id alone.

    Known limitation (deliberately kept): for a directory dataset only the first 4096 ``os.walk``
    entries are recorded, so editing a file past them leaves the cached lengths in place. The
    consequence is bounded -- the cache only holds the per-sample token lengths, which feed the
    PEAK RESERVATION and nothing else, so a stale entry can only make the reservation too small
    (residual OOM risk, never worse than the feature being off) or too large (wasted memory). It
    can NOT make training consume wrong data. Tightening it was tried and reverted: a full stat
    walk costs minutes on NFS for a frame-extracted tree, and any change to this function's
    output invalidates every existing entry, which costs a full re-scan (~105 min measured on
    80450 videos) -- a far larger regression than the blind spot it closes.
    """
    out = []
    for spec in (getattr(args, 'dataset', None) or []):
        path = str(spec).split('#', 1)[0]
        try:
            st = os.stat(path)
        except OSError:
            out.append([str(spec), None])
            continue
        if os.path.isfile(path):
            out.append([str(spec), [[os.path.basename(path), st.st_size, st.st_mtime_ns]]])
            continue
        items = []
        for root, dirs, files in os.walk(path):
            dirs.sort()
            files.sort()
            for fn in files:
                fp = os.path.join(root, fn)
                try:
                    s = os.stat(fp)
                except OSError:
                    continue
                items.append([os.path.relpath(fp, path), s.st_size, s.st_mtime_ns])
            if len(items) > 4096:
                items = items[:4096]
                break
        out.append([str(spec), items])
    return out


def _cache_entry(args, template, n: int) -> Optional[Tuple[str, Dict[str, Any]]]:
    """``(path, key)`` of the length cache entry for this dataset and config; None if caching is off.

    The key holds everything that can change a sample's token length, and nothing else: arguments
    that only decide which samples share a micro-batch (see `_LENGTH_TEMPLATE_ARGS`) are excluded,
    so changing them does not force a re-scan.
    """
    cache_dir = _cache_dir()
    if not cache_dir:
        return None
    key = {
        'version': _LENGTH_CACHE_VERSION,
        # what the samples are: content and which subset is selected
        'dataset': [str(x) for x in (getattr(args, 'dataset', None) or [])],
        'fingerprint': _dataset_fingerprint(args),
        'split_dataset_ratio': getattr(args, 'split_dataset_ratio', None),
        'columns': getattr(args, 'columns', None),
        'custom_dataset_info': [str(x) for x in (getattr(args, 'custom_dataset_info', None) or [])],
        'dataset_shuffle': getattr(args, 'dataset_shuffle', None),
        'n_samples': n,
        # tokenizer and multimodal processor come from the model
        'model': getattr(args, 'model', None),
        'model_revision': getattr(args, 'model_revision', None),
        # multimodal: pixels / frames
        'mm_env': {k: os.environ.get(k) for k in _MM_LENGTH_ENV_VARS},
        # safety net: a swift / processor upgrade invalidates every entry
        'swift_version': _swift_version(),
        'template_cls': type(template).__name__ if template is not None else None,
        'processor_cls': type(getattr(template, 'processor', None)).__name__ if template else None,
    }
    for name in _LENGTH_TEMPLATE_ARGS:
        key[f'tmpl.{name}'] = getattr(args, name, None)
    src = json.dumps(key, sort_keys=True, default=str)
    path = os.path.join(cache_dir,
                        f'lengths_v{_LENGTH_CACHE_VERSION}_{hashlib.md5(src.encode()).hexdigest()[:16]}.json')
    return path, key


def _lengths_to_blob(lengths: List[int]) -> str:
    """base64 of the little-endian int32 array (token lengths are far below 2**31)."""
    a = array.array('i', lengths)
    if sys.byteorder == 'big':
        a.byteswap()
    return base64.b64encode(a.tobytes()).decode('ascii')


def _lengths_from_blob(blob: str, n: int) -> Optional[List[int]]:
    """Inverse of `_lengths_to_blob`; None on any shape mismatch."""
    try:
        a = array.array('i')
        a.frombytes(base64.b64decode(blob))
        if sys.byteorder == 'big':
            a.byteswap()
        return a.tolist() if len(a) == n else None
    except Exception:
        return None


def _length_cache_load(path: str, n: int) -> Optional[Tuple[List[int], int]]:
    """Read a cache entry, or None when it is missing, stale, or does not match ``n``."""
    try:
        with open(path) as f:
            payload = json.load(f)
    except Exception:
        return None
    if not isinstance(payload, dict) or payload.get('version') != _LENGTH_CACHE_VERSION:
        return None
    lengths = _lengths_from_blob(payload.get('lengths') or '', n)
    if lengths is None:
        logger.info(f'[pad_warmup] Discarding the cached lengths: '
                    f'blob rows != {n} ({path}).')
        return None
    return lengths, int(payload.get('n_failed', 0) or 0)


def _length_cache_store(path: str, key: Dict[str, Any], lengths: List[int], n_failed: int) -> None:
    """Write a cache entry atomically, so a crash or a concurrent run cannot leave a partial file."""
    tmp = f'{path}.{os.getpid()}.tmp'
    try:
        with open(tmp, 'w') as f:
            json.dump(
                {
                    'version': _LENGTH_CACHE_VERSION,
                    'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'key': key,
                    'n': len(lengths),
                    'n_failed': n_failed,
                    'lengths': _lengths_to_blob(lengths),
                }, f)
        os.replace(tmp, path)
    except Exception as e:
        logger.warning(f'[pad_warmup] Cannot store the cached lengths: {e}')
        try:
            os.remove(tmp)
        except OSError:
            pass
        return
    _length_cache_gc(os.path.dirname(path))


def _length_cache_gc(cache_dir: str) -> None:
    """Keep only the newest `_LENGTH_CACHE_KEEP` entries; never fatal."""
    try:
        entries = [os.path.join(cache_dir, f) for f in os.listdir(cache_dir)
                   if f.startswith('lengths_v') and f.endswith('.json')]
        entries.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        for old in entries[_LENGTH_CACHE_KEEP:]:
            os.remove(old)
    except Exception:
        pass


def scan_sample_lengths(dataset, template, args) -> Optional[Tuple[List[int], int]]:
    """Encode the dataset once and return every sample's token length (same value on all ranks).

    Returns ``(lengths, n_failed)`` where ``lengths[i]`` is the token length of sample ``i``. The
    array always has exactly one slot per sample: a row that fails to encode keeps its slot as
    ``0`` instead of being dropped, so indices never shift. Uses a precomputed ``length(s)`` column
    when available, otherwise the ranks scan the dataset cooperatively (sharded by deduplicated
    media rows) and exchange the result through one all_gather -- see "Sharded scan" in the
    module docstring.
    """
    if dataset is None:
        return None
    if getattr(args, 'streaming', False):
        # An IterableDataset is neither materialized nor re-iterable, so a stable per-sample array
        # does not exist -- computing or caching one would be wrong.
        logger.info_once('[pad_warmup] Length scan disabled: streaming=True.')
        return None
    try:
        n = len(dataset)
    except Exception:
        return None
    if n == 0:
        return None

    from swift.dataset.utils import LazyLLMDataset
    hf_ds = dataset.dataset if isinstance(dataset, LazyLLMDataset) else dataset

    # Fast path: a precomputed length column (lazy_tokenize=False). It is merged into the
    # rendezvous below as just another kind of hit instead of returning early: a rank whose
    # column read transiently fails must NOT diverge from the ranks that succeeded, or the
    # others would wait for it inside a collective that never comes.
    fast = None
    for col in ('lengths', 'length'):
        try:
            vals = hf_ds[col]
            if vals is not None and len(vals) == n:
                fast = ([(max(v) if isinstance(v, (list, tuple)) else int(v)) for v in vals], 0)
                break
        except Exception:
            pass

    import torch
    import torch.distributed as dist
    dist_inited = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if dist_inited else 0
    world = dist.get_world_size() if dist_inited else 1
    if not dist_inited or world == 1:
        return fast if fast is not None else _scan_rank0(hf_ds, template, args, n)

    pg = _scan_process_group()
    entry = _cache_entry(args, template, n)
    cached = fast if fast is not None else (_length_cache_load(entry[0], n) if entry else None)
    # One tiny rendezvous: which ranks already hold the full cache? Everyone arrives here
    # quickly (a column read or a cache load at most), so the collective itself is never the
    # timeout's problem -- and it runs on the dedicated scan group anyway. `n` rides along:
    # if the ranks disagree on the dataset size, `lengths[i]` would describe different samples
    # on different ranks and nothing downstream can be consistent -- disable symmetrically
    # (every rank sees the same gather).
    flags = [torch.zeros(2, dtype=torch.long) for _ in range(world)]
    dist.all_gather(flags, torch.tensor([1 if cached is not None else 0, n], dtype=torch.long),
                    group=pg)
    ns = {int(f[1].item()) for f in flags}
    flags = [int(f[0].item()) for f in flags]
    if len(ns) > 1:
        logger.warning(f'[pad_warmup] The ranks disagree on the dataset size ({sorted(ns)}); '
                       f'disabling pad-warmup.')
        return None

    if all(flags):
        # Every rank has its own copy (cache entries are content-keyed, so they are identical):
        # nothing left to communicate.
        return cached
    if any(flags):
        # At least one rank holds the cache: broadcast it from the lowest such rank. The source
        # only reads a local file, so nobody waits inside the collective for a slow scan.
        src = flags.index(1)
        result = cached if rank == src else None
        if rank == src and result is None:
            result = _length_cache_load(entry[0], n)
        result = _broadcast_lengths(result, src, pg)
        if result and cached is None and entry and _is_local_main():
            # This node had no entry (node-local cache dir): store it so the next run is a
            # hit on *every* rank and needs no collective at all.
            _length_cache_store(*entry, result[0], result[1])
        return result

    # Sharded scan: every rank decodes an interleaved slice of the deduplicated media rows, so
    # the wall-clock is total_decode / (nodes x per-node workers) and no rank idles.
    return _scan_sharded(hf_ds, template, args, n, rank, world, entry, pg)


def _broadcast_lengths(result: Optional[Tuple[List[int], int]], src: int, pg=None,
                       ) -> Optional[Tuple[List[int], int]]:
    """Broadcast ``(lengths, n_failed)`` from `src` (None broadcasts failure); all ranks call it.
    CPU tensors on the scan group, so a failure here cannot abort the training communicators."""
    import torch
    import torch.distributed as dist
    size = torch.tensor([len(result[0]) + 2 if result else 0], dtype=torch.long)
    dist.broadcast(size, src=src, group=pg)
    n_len = int(size.item())
    if not n_len:
        return None
    # One collective for [n, n_failed, *lengths].
    values = [len(result[0]), result[1]] + result[0] if result else [0] * n_len
    buf = torch.tensor(values[:n_len], dtype=torch.long)
    dist.broadcast(buf, src=src, group=pg)
    v = buf.tolist()
    result = ([int(x) for x in v[2:]], int(v[1]))
    return result if result[0] else None


def _agree_dedup_plan(hf_ds, n: int, todo: List[int], dup_of: Dict[int, int],
                      sizes: Dict[int, int], world: int,
                      pg=None) -> Optional[Tuple[List[int], Dict[int, int], Dict[int, int]]]:
    """Make sure every rank shards the *same* plan, falling back to "encode everything".

    The shard layout (`counts`, and which global row a gathered value belongs to) is derived
    from `todo`, so the plans must match bit for bit. They normally do -- the plan is a
    deterministic function of the dataset -- but `_dedup_plan_or_all` degrades to
    ``range(n)`` when a row cannot be read, and that can happen on one rank only (a transient
    IO error). The shapes would then differ and the result all_gather would mismatch, i.e.
    hang or abort mid-scan. The tail-first sizes ride along in the digest for the same
    reason (a node whose `stat` fails would order its shard differently). One extra 3-int
    collective rules both out up front.

    Returns ``None`` when the ranks disagree on `n` itself: the "encode everything" fallback
    would then use each rank's LOCAL `n`, the shard shapes would still mismatch, and
    `lengths[i]` has no consistent meaning anyway -- pad-warmup must disable, symmetrically
    (every rank sees the same gathered vectors, so every rank returns None here).
    """
    import torch
    import torch.distributed as dist
    import zlib
    digest = zlib.crc32(repr((todo, sorted(dup_of.items()), sorted(sizes.items()))).encode()) \
        & 0x7FFFFFFF
    mine = torch.tensor([n, len(todo), digest], dtype=torch.long)
    out = [torch.zeros(3, dtype=torch.long) for _ in range(world)]
    dist.all_gather(out, mine, group=pg)
    ns = {int(t[0].item()) for t in out}
    seen = {(int(t[1].item()), int(t[2].item())) for t in out}
    if len(ns) > 1:
        logger.warning(f'[pad_warmup] The ranks disagree on the dataset size ({sorted(ns)}); '
                       f'disabling pad-warmup (the length array would be misaligned).')
        return None
    if len(seen) == 1:
        return todo, dup_of, sizes
    logger.warning(f'[pad_warmup] The ranks disagree on the dedup plan ({sorted(seen)}); '
                   f'scanning every row instead (slower, but the shards stay consistent).')
    return list(range(n)), {}, {}


def _exchange_shards_via_gather(counts: List[int], values: List[int], pg=None) -> List[List[int]]:
    """all_gather the shard values, padded to equal length, on the scan group (CPU tensors).
    The wait inside the collective is only the spread of the shard finish times, not the whole
    scan -- and with the dedicated group even an extreme spread degrades to a timeout of THIS
    group, leaving the training communicators untouched."""
    import torch
    import torch.distributed as dist
    world = len(counts)
    width = max(counts)
    buf = torch.tensor(values + [0] * (width - len(values)), dtype=torch.long)
    out = [torch.zeros(width, dtype=torch.long) for _ in range(world)]
    dist.all_gather(out, buf, group=pg)
    return [[int(x) for x in t.tolist()[:counts[r]]] for r, t in enumerate(out)]


def _agree_scan_selection(calib: List[int], sel: List[int], world: int, pg=None) -> bool:
    """One tiny collective: every rank must scan the SAME pass-2 selection, or the shard
    mapping of the result all_gather would be garbage. A disagreement degrades ALL ranks to
    the plain tail-first scan, symmetrically (every rank sees the same gathered digests)."""
    import torch
    import torch.distributed as dist
    import zlib
    digest = zlib.crc32(repr((calib, sel)).encode()) & 0x7FFFFFFF
    mine = torch.tensor([len(calib), len(sel), digest], dtype=torch.long)
    out = [torch.zeros(3, dtype=torch.long) for _ in range(world)]
    dist.all_gather(out, mine, group=pg)
    if len({(int(t[0].item()), int(t[1].item()), int(t[2].item())) for t in out}) == 1:
        return True
    logger.warning('[pad_warmup] The ranks disagree on the exact-batch selection; falling '
                   'back to the tail-first scan.')
    return False


def _scan_sharded(hf_ds, template, args, n: int, rank: int, world: int, entry, pg=None,
                  ) -> Optional[Tuple[List[int], int]]:
    """Scan the dataset cooperatively: rank ``r`` encodes the deduplicated rows
    ``todo[r::world]``, the per-shard values are exchanged through one all_gather, and every
    rank rebuilds the full length array locally (and stores it, so the next run needs no scan
    at all). No file locks / file polling: a crashed peer fails the collective loudly instead
    of leaving the rest blocked on an invisible rendezvous forever."""
    t0 = time.perf_counter()
    todo, dup_of, sizes = _dedup_plan_or_all(hf_ds, n)
    logger.info(f'[pad_warmup] Dedup plan: rank {rank}/{world}, {len(todo)}/{n} rows, '
                f'{len(dup_of)} duplicates, took {time.perf_counter() - t0:.2f}s.')
    # The shard layout is derived from `todo`, so all ranks must hold the very same plan.
    plan = _agree_dedup_plan(hf_ds, n, todo, dup_of, sizes, world, pg)
    if plan is None:
        # The ranks disagree on `n` itself: disabling, symmetrically.
        return None
    todo, dup_of, sizes = plan
    # Tail-first: each rank's interleaved shard is already graded big -> small, so a deadline
    # cutoff anywhere leaves the SHORT rows unscanned on every rank.
    order_t0 = time.perf_counter()
    chars_map = _text_char_sizes(hf_ds, list(range(n)))
    todo = _order_tail_first(todo, sizes, chars_map)
    logger.info(f'[pad_warmup] Tail-first order ready in {time.perf_counter() - order_t0:.2f}s; '
                f'head={todo[:3]}.')
    num_proc = _scan_num_proc_sharded(args)
    global _SCAN_TEMPLATE
    _SCAN_TEMPLATE = template

    def _one_pass(rows: List[int], deadline_s: float) -> Dict[int, int]:
        """Scan ``rows[rank::world]`` and all_gather the shards; ``{row: value}`` everywhere."""
        mine = rows[rank::world]
        logger.info(f'[pad_warmup] Sharded scan: rank {rank}/{world} encodes '
                    f'{len(mine)}/{len(rows)} deduplicated rows (of {n} samples, '
                    f'num_proc={num_proc})...')
        try:
            values = _scan_rows([hf_ds[i] for i in mine], num_proc,
                                deadline_s=deadline_s) if mine else []
        except Exception as e:
            # Contribute zeros rather than tear down the rendezvous: a fully failed shard
            # leaves its rows unscanned, and the peak-resolution chain degrades
            # (exact-batch -> partial coverage -> off) accordingly.
            logger.warning(f'[pad_warmup] Rank {rank} failed to scan its shard: {e}')
            values = [0] * len(mine)
        counts = [(len(rows) - r + world - 1) // world for r in range(world)]
        gathered = _exchange_shards_via_gather(counts, values, pg)
        base = {}
        for r in range(world):
            for j, v in enumerate(gathered[r]):
                base[rows[r + j * world]] = v
        return base

    t_scan = time.perf_counter()
    two_pass = _exact_batch_scan_plan(args, n, todo, dup_of, sizes, chars_map, todo)
    if two_pass is not None:
        calib, _ = two_pass
        # Pass 1: the calibration head fits the length proxy (identical on every rank after
        # the gather, so the pass-2 selection is identical too).
        base = _one_pass(calib, _exact_batch_calib_budget())
        sel, coef = _exact_batch_fit_select(two_pass, n, dup_of, sizes, chars_map, base)
        if sel is not None and not _agree_scan_selection(calib, sel, world, pg):
            # Asymmetric proxy/plan state would misalign the pass-2 shards: degrade together.
            sel = None
        if sel is None:
            logger.info('[pad_warmup] exact-batch scan unavailable; falling back to the '
                        'tail-first scan.')
            two_pass = None
        else:
            # Pass 2: only the rows of the predicted-heaviest real micro-batches.
            logger.info(f'[pad_warmup] exact-batch scan: calibration={len(calib)} rows '
                        f'(proxy fit mape={coef[3]:.2%}), {len(sel)}/{len(todo)} selected rows.')
            remaining = max(0.0, _SCAN_DEADLINE_S - (time.perf_counter() - t_scan))
            base.update(_one_pass(sel, remaining))
            rows = calib + sel
            lengths = _expand_dedup(hf_ds, rows, [base.get(i, 0) for i in rows], dup_of, n)
    if two_pass is None:
        remaining = max(0.0, _SCAN_DEADLINE_S - (time.perf_counter() - t_scan))
        base = _one_pass(todo, remaining)
        lengths = _expand_dedup(hf_ds, todo, [base.get(i, 0) for i in todo], dup_of, n)
    n_failed = sum(1 for v in lengths if v <= 0)
    ok = [v for v in lengths if v > 0]
    logger.info(f'[pad_warmup] Scan finished: {n} rows, {len(ok)} valid, {n_failed} failed, '
                f'mean={sum(ok) / max(1, len(ok)):.0f}, max={max(ok) if ok else 0}, '
                f'took {time.perf_counter() - t0:.2f}s.')
    if entry and _is_local_main():
        # One writer per node: every rank rebuilt the identical array, so `world_size`
        # concurrent writes of the same multi-MB file would be pure waste.
        _length_cache_store(*entry, lengths, n_failed)
    return lengths, n_failed


def _scan_rank0(hf_ds, template, args, n: int) -> Optional[Tuple[List[int], int]]:
    """Scan (on rank 0) for every sample's token length, using the on-disk cache if present."""
    t0 = time.perf_counter()
    entry = _cache_entry(args, template, n)
    if entry:
        cached = _length_cache_load(entry[0], n)
        if cached is not None:
            lengths, n_failed = cached
            logger.info(f'[pad_warmup] Using the cached sample lengths: {n} rows, {n_failed} failed '
                        f'({time.perf_counter() - t0:.2f}s, {entry[0]}).')
            return lengths, n_failed

    global _SCAN_TEMPLATE
    _SCAN_TEMPLATE = template
    try:
        lengths = _scan_lengths(hf_ds, n, _scan_num_proc(args), args)
    except Exception as e:
        logger.warning(f'[pad_warmup] Failed to scan the dataset: {e}')
        return None
    n_failed = sum(1 for v in lengths if v <= 0)
    ok = [v for v in lengths if v > 0]
    logger.info(f'[pad_warmup] Scan finished: {n} rows, {len(ok)} valid, {n_failed} failed, '
                f'mean={sum(ok) / max(1, len(ok)):.0f}, max={max(ok) if ok else 0}, '
                f'took {time.perf_counter() - t0:.2f}s.')
    if entry:
        _length_cache_store(*entry, lengths, n_failed)
    return lengths, n_failed


def _packing_bin_peak(dataset, args, mbs: int, cp_size: int) -> Optional[int]:
    """Exact peak of a packed micro-batch, read from the REAL bin layout.

    `sft.py` wraps the training dataset in `PackingDataset` before the trainer ever sees it, and
    that constructor has already bin-packed the whole dataset -- `packed_idx` (sample indices per
    bin) and `packed_length` (raw length per bin) -- and broadcast it to every rank. Reusing that
    layout is both free and exact: re-packing here could drift from upstream (it packs in
    `PACKING_BATCH_SIZE` chunks across `packing_num_proc` workers, holding back the last
    unfinished bin), and any drift would make this an estimate again.

    Under CP the per-SAMPLE 2*cp rounding (`_handle_megatron_cp`, applied to each sample BEFORE
    `packing_row`) can push a full bin PAST `packing_length` -- the flat `mbs * packing_length`
    cap under-reserves there by roughly `cp / mean_sample_len` (measured: 3.9% at cp=8 on
    80-400-token samples, 21.5% on 20-50-token samples). The rounding is per-SAMPLE, so the bin
    lengths must be re-derived from `packed_idx` + the sample lengths, not read off
    `packed_length` (a raw bin total loses that information: two bins of ~1024 raw tokens can
    differ by 77% after the rounding).

    The peak itself comes in three layers, each falling back to the next:

    1. **Sampler replay over the bin sequence.** The dataloader samples PACKED BINS through
       `MegatronPretrainingRandomSampler`, so the bin lengths are the exact analogue of the
       sample-level replay used elsewhere; per-rank like that path. This is tight even when one
       sample fills a whole bin (long samples), where a top-`mbs` bound degrades (~14% loose).
    2. **Top-`mbs` sum.** Pure math: every micro-batch is `mbs` bins, so the sum of the `mbs`
       largest bounds EVERY layout, shuffled or not. Cannot under-reserve, but stays tight only
       while bin lengths are uniform (bin-packing keeps them so except in the one-sample-per-bin
       regime).
    3. The caller's flat cap. Never reached from here (this returns None instead).

    Returns None when the real layout is unavailable (streaming's `IterablePackingDataset`, a
    non-packed dataset, or a rank that never received the broadcast); the caller falls back to
    the flat cap, which is the safe direction except under CP (see above), so the caller logs a
    warning when cp>1.
    """
    packed_idx = getattr(dataset, 'packed_idx', None)
    if not packed_idx:
        return None
    try:
        lengths = dataset.dataset['lengths']
        per = [sum(x) if isinstance(x, list) else int(x) for x in lengths]
        if cp_size > 1:
            bin_lens = [sum(_cp_pad_length(per[i], cp_size) for i in bin) for bin in packed_idx]
        else:
            bin_lens = getattr(dataset, 'packed_length', None)
            if bin_lens is None or len(bin_lens) != len(packed_idx):
                bin_lens = [sum(per[i] for i in bin) for bin in packed_idx]
        bin_lens = [int(x) for x in bin_lens if x]
        if not bin_lens:
            return None
    except Exception as e:
        logger.info(f'[pad_warmup] Packing bin layout unavailable ({e}); using the flat cap.')
        return None
    try:
        peak = replay_peak_batch_length(args, dataset, bin_lens, mbs, agg='sum',
                                        dp_rank=_current_data_parallel_rank())
    except Exception as e:
        logger.info(f'[pad_warmup] Bin replay unavailable ({e}); using the top-bin bound.')
        peak = None
    if peak:
        return int(peak)
    return int(sum(sorted(bin_lens, reverse=True)[:max(1, mbs)]))


def get_pad_warmup_length(args, dataset, template) -> Optional[int]:
    """Resolve the warm-up peak length; ``None`` disables it.

    One dummy-segment scheme either way -- only the aggregation differs:
    * padding_free: a row is the concatenation of its samples, so the peak is the largest *sum*
      over any micro-batch;
    * otherwise: a row is padded to its longest sample, so the peak is the largest *max*.
    """
    if not get_env_args('SWIFT_PAD_WARMUP', bool, False):
        return None
    alloc_conf = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '').lower().replace(' ', '')
    if 'expandable_segments:true' in alloc_conf:
        # With expandable segments the allocator already handles fragmentation; a
        # peak-shaped reservation on top only wastes memory.
        logger.info_once('[pad_warmup] Disabled: PYTORCH_CUDA_ALLOC_CONF has '
                         'expandable_segments:True (fragmentation is already handled).')
        return None
    if getattr(template, 'mode', 'train') not in ('transformers', 'train'):
        # RLHF/PRM trainers index per-segment logits via packed_seq_params.seq_lens; an extra
        # dummy segment would shift the segment count.
        logger.info_once(f'[pad_warmup] Disabled: template.mode={getattr(template, "mode", None)!r}.')
        return None

    max_length = getattr(args, 'max_length', None) or getattr(args, 'seq_length', None)
    if not max_length or max_length <= 0:
        return None
    max_length = int(max_length)
    cp_size = max(1, int(getattr(args, 'context_parallel_size', 1) or 1))
    padding_free = bool(getattr(args, 'padding_free', False))

    def _finalize(x: int) -> int:
        # `_align` only guarantees the attention-backend multiple (cp, not 2*cp, without
        # fp8/fused); the collator pads every sample to a 2*cp multiple under CP and would
        # re-ceil the dummy past the target. Keep the target itself 2*cp-aligned so the
        # warm-up batch lands on it exactly.
        return _cp_pad_length(_align(x, args), cp_size)

    if getattr(args, 'packing', False):
        # Packing bins are capped at `packing_length` (`PackingDataset`: binpacking
        # `to_constant_volume(packing_length)`, defaulting to `template.max_length`), and ONE
        # micro-batch carries `micro_batch_size` bins -- `Template.data_collator` flattens them
        # with `batch = sum(batch, start=[])`. So with padding_free the packed row this rank
        # sees is at most `mbs * packing_length` (a HARD cap of the layout, not an estimate),
        # while the padded layout keeps the row at `max(bins) <= packing_length`. v9 returned
        # `max_length` in both cases: an mbs-fold UNDER-reservation of the padding_free peak
        # (dummy_len <= 0, i.e. the feature silently did nothing) whenever mbs > 1, and it also
        # ignored `packing_length` when that argument raises the cap above `max_length`.
        # CP: `_handle_megatron_cp` pads EVERY sample of a bin to a 2*cp multiple before
        # `packing_row`, so a bin that is full in RAW length ends up LONGER than
        # `packing_length`. Measured against the real collator (demo12 §G): with cp=8 on a
        # typical t2t length mix, 20/23 micro-batches exceed the flat cap, by up to 3.9% --
        # i.e. the flat cap alone leaves the peak un-reserved exactly when CP is on. Rather
        # than invent slack (the per-bin sample count is unknown without the bins), the real
        # bin layout is read off the `PackingDataset` and the CP-rounded top-`mbs` bins are
        # summed: still an upper bound, but an exact one, and tighter than the flat cap even
        # at cp=1 because not every bin is full.
        # (`megatron_base_args`: `if self.packing: self.padding_free = True`, so the padded
        # branch below is unreachable under megatron today -- kept because the cap it encodes
        # is the correct one for that layout, not because the layout is expected.)
        packing_length = int(getattr(args, 'packing_length', None) or max_length)
        if padding_free:
            mbs = max(1, int(getattr(args, 'micro_batch_size', 1) or 1))
            exact = _packing_bin_peak(dataset, args, mbs, cp_size)
            if exact is not None:
                return _finalize(exact)
            if cp_size > 1:
                # The flat cap ignores the per-sample CP rounding and under-reserves by roughly
                # cp / mean_sample_len (measured up to 21.5%); the real layout was unavailable
                # (streaming, a non-packed dataset, ...) so this is the best that can be said.
                logger.info_once('[pad_warmup] Packing + CP: the real bin layout is unavailable; '
                                 'the flat cap may under-reserve the peak (residual risk).')
            packing_length *= mbs
        return _finalize(packing_length)

    from swift.dataset.utils import LazyLLMDataset
    hf_ds = dataset.dataset if isinstance(dataset, LazyLLMDataset) else dataset
    scanned = scan_sample_lengths(dataset, template, args)
    if not scanned:
        logger.warning('[pad_warmup] Disabled: the sample lengths are unavailable.')
        return None
    lengths, n_failed = scanned
    if cp_size > 1:
        # Under CP the collator pads EVERY sample to a 2*cp multiple (`_handle_megatron_cp`),
        # so the real packed length is a sum of PADDED lengths. Peaks computed from the raw
        # lengths would under-reserve by up to group_size * (2*cp - 1) tokens -- the fatal
        # direction, and invisible to a raw-length overshoot check.
        lengths = [_cp_pad_length(v, cp_size) if v > 0 else 0 for v in lengths]
        max_length = _cp_pad_length(max_length, cp_size)
    n = len(lengths)
    valid = [v for v in lengths if v > 0]
    agg = 'sum' if padding_free else 'max'
    group_size = max(1, int(getattr(args, 'micro_batch_size', 1) or 1))

    if not n_failed and valid:
        # Full scan: exact replay of this rank's real micro-batches, no modelling at all.
        mode = 'exact'
        peak = replay_peak_batch_length(args, hf_ds, lengths, group_size, agg=agg,
                                        dp_rank=_current_data_parallel_rank())
    else:
        # Missing rows. The design rule is ASYMMETRIC: a too-loose reservation OOMs the
        # step-0 warm-up batch itself (the feature kills an otherwise healthy run), while
        # an under-reservation only leaves residual risk -- never worse than the feature
        # being off. So: no substitution (a deterministic fill value colliding k>=2 times
        # in one batch inflates the peak ~linearly with mbs -- measured 198%/296%/478% at
        # mbs 2/4/8); measure what the scan actually covered instead.
        peak = None
        if len(valid) >= _EXACT_BATCH_MIN_CALIB:
            # Batch-level tail-first: the exact peak of THIS dp rank, from the fully scanned
            # real micro-batches. Returns None unless the proxy fit passes its quality gate
            # AND the predicted Top-N batches are entirely scanned.
            try:
                peak = _exact_batch_peak(args, hf_ds, lengths, group_size, agg)
            except Exception as e:
                logger.warning(f'[pad_warmup] exact-batch peak failed ({e}).')
                peak = None
        if peak:
            mode = 'exact-batch'
        else:
            # Gates failed (proxy unfit / Top-N not fully covered): the max over this rank's
            # fully covered REAL batches is exact for what it covers and undershoot-only --
            # the direction the asymmetry rule prefers over any estimate.
            try:
                peak = _covered_batch_peak(args, lengths, group_size, agg)
            except Exception as e:
                logger.warning(f'[pad_warmup] covered-batch peak failed ({e}).')
                peak = None
            if peak:
                mode = 'exact-batch-partial'
        if not peak:
            # Not one fully covered real batch -- an ESTIMATE (Monte Carlo over the scanned
            # subset) is the only remaining option, and an estimate errs in BOTH directions.
            # Overshoot OOMs the step-0 warm-up batch of an otherwise healthy run, which is
            # the one unacceptable direction -- so the feature turns itself off instead.
            # (The two-pass scan makes this corner rare: the scan is only ~calib+Top-N rows,
            # so a deadline cutoff or a cache written for a different mbs are the realistic
            # triggers.)
            logger.warning(f'[pad_warmup] Disabled: scan incomplete ({n - n_failed}/{n} rows) '
                           f'and no fully covered real batch to measure.')
            return None
    logger.info(f'[pad_warmup] Peak mode: {mode} '
                f'(length coverage {n - n_failed}/{n} = {(n - n_failed) / max(1, n):.0%}).')
    if not peak:
        logger.warning('[pad_warmup] Disabled: the peak batch length is unavailable.')
        return None
    # `max_length` still truncates every sample, so it remains a hard ceiling.
    return _finalize(min(peak, group_size * max_length if padding_free else max_length))


def _prewarm_communicators() -> None:
    """Force the lazy communicator (XCCL/BKCL) setup while the device still has free memory.

    The collective backends build their context on the **first** collective of a process group,
    and they allocate through the raw device API instead of the torch caching allocator. Without
    pad-warmup that happens on the small step-0 batch; with pad-warmup step 0 *is* the
    peak-shaped batch, so the device is already full and the backend cannot obtain its buffers:

        [ERR][XCCL][kl3_init.cpp:237] param 10 data alloc failed
        ValueError: [RUNTIME ERROR]: error code= 2, out of memory (at .../xccl/XCCLUtils.cpp:28)

    This is not hypothetical: with 60 epochs the per-rank peaks grow (rank4 19908 -> 24773) and
    step 1 dies this way. Running one tiny all-reduce per group up-front moves the allocation to
    a point where the memory footprint is still small. Best-effort: never raises.
    """
    try:
        import torch
        import torch.distributed as dist
        if not (dist.is_available() and dist.is_initialized()):
            return

        # Every process group this run can use. Enumerated from `parallel_state` rather than a
        # hand-written list, so groups added later (e.g. `get_expert_model_parallel_group`, which
        # is the EP group and is easy to miss) are covered automatically.
        groups = [(None, 'default')]  # (group, name); None means the default (world) group
        seen = set()
        try:
            from megatron.core import parallel_state as mpu
        except Exception:
            mpu = None
        for name in sorted(dir(mpu)) if mpu else []:
            if not (name.startswith('get_') and name.endswith('_group')):
                continue
            fn = getattr(mpu, name, None)
            if not callable(fn):
                continue
            try:
                group = fn()
                # Skip world_size 1: megatron short-circuits its collectives, so no context is
                # ever built for them. Dedup on the object, not on the rank set: two groups over
                # the same ranks are still two distinct communicators, each lazily creating its
                # own context.
                if group is None or dist.get_world_size(group) <= 1 or id(group) in seen:
                    continue
            except Exception:
                continue
            seen.add(id(group))
            groups.append((group, name))

        buf = torch.zeros(1, dtype=torch.float32, device=torch.cuda.current_device())
        failed = []
        for group, name in groups:
            try:
                dist.all_reduce(buf, group=group)
            except Exception as e:
                failed.append(f'{name}: {e}')
        torch.cuda.synchronize()
        if failed:
            logger.warning(f'[pad_warmup] Failed to warm up {len(failed)} communicator(s): '
                           f'{failed}')
        logger.info_once(f'[pad_warmup] Pre-warmed {len(groups)} communicator(s) before the '
                         f'peak-shaped batch: ' + ', '.join(f'{n}({dist.get_world_size(g)})'
                                                            for g, n in groups))
    except Exception as e:
        logger.warning(f'[pad_warmup] Communicator pre-warm failed: {e}')


def _prewarm_pipeline_p2p(args) -> None:
    """Force the lazy point-to-point setup of the pipeline group(s) before the carve.

    `_prewarm_communicators` only issues an `all_reduce` per group. On the collective
    backends the *point-to-point* path (`p2p_communication.send_forward/recv_forward`,
    i.e. `isend`/`irecv`) can build its own connection and buffers on FIRST USE, outside
    the torch caching allocator -- the same failure class as the lazy XCCL context (error
    702) and the DeepEP buffer, and the only one of the three that neither pre-warm
    covers. With PP>1 that first use is the step-0 forward, which under pad-warmup is the
    PEAK-shaped batch on an already-carved heap.

    One 1-element ring exchange (each rank sends to its pipeline successor and receives
    from its predecessor, via `batch_isend_irecv`, so there is no ordering deadlock) is
    enough to establish it. Every rank of the job runs this -- the enable decision is
    agreed before -- so no rank can be left alone in a p2p op. Best-effort: never raises.

    OPT-IN (`SWIFT_PAD_WARMUP_PP_P2P=1`), unlike the other pre-warms, because this one has
    never run against a real backend and its failure mode is the worst available: if one
    rank leaves the loop early (a raise inside the try) while its peers are already in
    `req.wait()`, the job HANGS instead of OOMing, and there is no timeout here. The
    fragmentation it protects against is a risk; a hang is a certainty once it triggers.
    Enable it deliberately on a PP>1 run, after checking the log line it emits.
    """
    if os.environ.get('SWIFT_PAD_WARMUP_PP_P2P', '0').lower() in ('0', 'off', 'false', 'no'):
        return
    if max(1, int(getattr(args, 'pipeline_model_parallel_size', 1) or 1)) <= 1:
        return
    try:
        import torch
        import torch.distributed as dist
        from megatron.core import parallel_state as mpu
        if not (dist.is_available() and dist.is_initialized()):
            return
        group = mpu.get_pipeline_model_parallel_group()
        # VPP / newer megatron may hand back a list of pipeline groups; the layout is
        # symmetric across ranks, so iterating it in order keeps every rank in step.
        groups = list(group) if isinstance(group, (list, tuple)) else [group]
        me = dist.get_rank()
        done = 0
        for g in groups:
            if g is None or dist.get_world_size(g) <= 1:
                continue
            ranks = dist.get_process_group_ranks(g)
            if me not in ranks:
                continue
            i = ranks.index(me)
            nxt, prv = ranks[(i + 1) % len(ranks)], ranks[(i - 1) % len(ranks)]
            send = torch.zeros(1, dtype=torch.float32, device=torch.cuda.current_device())
            recv = torch.empty_like(send)
            reqs = dist.batch_isend_irecv([dist.P2POp(dist.isend, send, nxt, g),
                                           dist.P2POp(dist.irecv, recv, prv, g)])
            for req in reqs:
                req.wait()
            done += 1
        if done:
            torch.cuda.synchronize()
            logger.info_once(f'[pad_warmup] Pre-warmed the point-to-point path of {done} '
                             f'pipeline group(s) before the peak-shaped batch.')
    except Exception as e:
        logger.warning(f'[pad_warmup] Pipeline p2p pre-warm failed: {e}')


def _prewarm_deepep_buffer(args) -> None:
    """Pre-create the DeepEP all-to-all buffer while the device still has free memory.

    With `moe_enable_deepep` (flex dispatcher, deepep backend) the first MoE dispatch
    lazily creates a global `deep_ep.Buffer` (`fused_a2a.get_buffer`): a RAW device
    allocation plus the IBGDA/RDMA setup, outside the torch caching allocator. Without
    pad-warmup that happens on a small early batch; with pad-warmup the first dispatch is
    on the PEAK-shaped batch, the heap is already carved full, and the raw allocation
    fails exactly like the lazy XCCL context did (the class of failure
    `_prewarm_communicators` fixes -- but a DeepEP buffer is not a dist collective, so it
    is not covered there). Creating it here moves the allocation to a point where the
    footprint is still small; the buffer persists in a module global and every later
    dispatch just reuses it. Best-effort: any failure leaves the lazy path unchanged.
    """
    if not getattr(args, 'moe_enable_deepep', False):
        return
    try:
        from megatron.core import parallel_state as mpu
        from megatron.core.transformer.moe import fused_a2a
        group = mpu.get_expert_tensor_and_model_parallel_group()
        # get_hidden_bytes(x): hidden_size x max(element_size, 2) -- hidden*2 for bf16/fp8.
        hidden_bytes = int(getattr(args, 'hidden_size')) * 2
        fused_a2a.get_buffer(group, hidden_bytes)
        logger.info(f'[pad_warmup] DeepEP buffer pre-created before the peak-shaped batch '
                    f'(hidden_bytes={hidden_bytes}).')
    except Exception as e:
        logger.warning(f'[pad_warmup] DeepEP buffer pre-creation failed ({e}); the first '
                       'MoE dispatch will create it lazily as before.')


def _settle_allocator_state(optimizer) -> None:
    """Land the lazily-created *persistent* allocations, then release every free segment.

    See "Settle, then carve" in the module docstring. `optimizer` is the trainer's (possibly
    chained) megatron optimizer; its `init_state_fn` allocates Adam's `exp_avg`/`exp_avg_sq`
    via `zeros_like` -- exactly what the first `step()` would do, but without consuming the
    RNG stream (dropout) or letting the moments absorb a settle gradient. A failing
    sub-optimizer only loses its own settle. With `optimizer=None` (or a non-megatron
    optimizer) only the heap clean-up runs. Best-effort: never raises.
    """
    try:
        n_state = 0
        opts = getattr(optimizer, 'chained_optimizers', None) or ([optimizer] if optimizer else [])
        for opt in opts:
            try:
                inner = getattr(opt, 'optimizer', None)
                init_state_fn = getattr(opt, 'init_state_fn', None)
            except Exception:
                continue
            if inner is None or not callable(init_state_fn):
                continue
            try:
                init_state_fn(inner, getattr(opt, 'config', None))
                n_state += len(inner.state)
            except Exception as e:
                logger.warning(f'[pad_warmup] Cannot materialise the optimizer state of '
                               f'{type(opt).__name__}: {e}')
        import gc
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
            mem = (f'allocated={torch.cuda.memory_allocated() / 2**30:.2f} GiB, '
                   f'reserved={torch.cuda.memory_reserved() / 2**30:.2f} GiB')
        except Exception:
            mem = 'memory stats unavailable'
        logger.info(f'[pad_warmup] Allocator settled before the peak-shaped carve: optimizer '
                    f'state materialised for {n_state} param shard(s); free segments released '
                    f'({mem}).')
    except Exception as e:
        logger.warning(f'[pad_warmup] Allocator settle failed: {e}')


def _patch_moe_dispatcher_release(dispatcher_cls=None) -> bool:
    """Release the AlltoAll dispatcher's per-dispatch router state after combine.

    Backport of upstream Megatron-LM #8297/#8231 (both merged 2026-10, after this image's
    megatron-core 0.18.0). In 0.18.0, `dispatch_preprocess` stores `self.probs` (the router
    output, [num_tokens, num_experts]) and `self.routing_map` on the module and never clears
    them: the attribute pins the tensor -- and with it the router's whole autograd subgraph,
    which under full recompute survives into the remaining backward -- until the next step's
    same-layer forward overwrites it. The per-layer set stays alive across steps while
    drifting position every step (allocate-new then free-old), punching long-lived holes into
    the warm-up-carved blocks. In this version the last consumers are `combine_preprocess`
    (probs, and only for `.dtype`) and `combine_postprocess` itself (routing_map for the
    unpermute), so releasing both at the end of `combine_postprocess` is safe; recompute
    simply re-runs dispatch and re-populates them. Applied as a method wrapper so the fix
    lives in this file; idempotent. `dispatcher_cls` is injectable for tests. Best-effort:
    any failure returns False instead of raising.
    """
    try:
        if dispatcher_cls is None:
            from megatron.core.transformer.moe.token_dispatcher import (
                MoEAlltoAllTokenDispatcher as dispatcher_cls)
        if getattr(dispatcher_cls.combine_postprocess, '_pad_warmup_release', False):
            return True
        orig = dispatcher_cls.combine_postprocess

        @functools.wraps(orig)
        def combine_postprocess(self, *a, **kw):
            out = orig(self, *a, **kw)
            # combine is the last consumer of the router state stored at dispatch time; retaining
            # it pins the recomputed MoE forward graph until the next iteration (upstream #8297).
            self.probs = None
            self.routing_map = None
            return out

        combine_postprocess._pad_warmup_release = True
        dispatcher_cls.combine_postprocess = combine_postprocess
        logger.info_once('[pad_warmup] MoEAlltoAllTokenDispatcher will release router '
                         'probs/routing_map after combine (backport of upstream #8297/#8231).')
        return True
    except Exception as e:
        logger.warning(f'[pad_warmup] MoE dispatcher release patch failed: {e}')
        return False


def _prewarm_runtime(args, optimizer) -> None:
    """Pre-carve runtime preparation, while the device still has free memory (see
    "Settle, then carve" in the module docstring). Order matters: communicators and the
    DeepEP buffer first (they need free memory), then the allocator settle, so the
    peak-shaped batch is carved on a clean heap wrapped around the stable footprint.
    Every piece guards itself internally (best-effort, never raises).
    """
    _prewarm_communicators()
    _prewarm_pipeline_p2p(args)
    _prewarm_deepep_buffer(args)
    if optimizer is None:
        logger.warning('[pad_warmup] wrap_pad_warmup() called without optimizer=...: the '
                       'allocator settle skips the optimizer states (only the heap clean-up '
                       'runs). Pass optimizer=self.optimizer (trainers/base.py) for the full '
                       'settle -- a bare pad_warmup.py replacement gets everything else.')
    _settle_allocator_state(optimizer)
    _patch_moe_dispatcher_release()


class PadWarmupCollator:
    """Wrap a data collator so that the first batch of the run is padded to `length`.

    `num_batches` is per process (every dataloader worker owns a copy of the collator), and with
    `dataloader_persistent_workers=False` a new worker -- with a fresh copy -- is forked every
    epoch. `flag_path` is therefore claimed atomically as well, so that the whole run pads exactly
    one batch per rank. Every padding is logged so the behaviour stays auditable.
    """

    def __init__(self, collator: Callable[..., Dict[str, Any]], length: int, num_batches: int = 1,
                 flag_path: Optional[str] = None, padding_free: bool = True, cp_size: int = 1):
        self.collator = collator
        self.length = length
        self.num_batches = num_batches
        self.flag_path = flag_path
        self.padding_free = padding_free
        self.cp_size = max(1, int(cp_size or 1))
        self._overshoot_warned = False

    def _batch_length(self, batch: List[Dict[str, Any]]) -> Optional[int]:
        """The batch's packed length in the space the model actually sees: under CP every
        sample is padded to a 2*cp multiple by the collator (`_handle_megatron_cp`), so the
        per-sample lengths are padded before aggregating."""
        try:
            row_lens = [int(r.get('length') or len(r['input_ids'])) for r in batch]
            if self.cp_size > 1:
                row_lens = [_cp_pad_length(v, self.cp_size) for v in row_lens]
            return sum(row_lens) if self.padding_free else max(row_lens)
        except Exception:
            return None

    def _check_overshoot(self, batch: List[Dict[str, Any]]) -> None:
        """Closed-loop guard: warn if a real batch ever exceeds the warm-up length.

        The warm-up length is a *tight upper bound* of the peak; everything that can make it
        an UNDERestimate (a lying size proxy truncating the scan early, a replay/sampler drift,
        a stale cache) is otherwise silent until the OOM. One len() per row per batch is the
        entire cost. The reservation cannot be fixed retroactively (the heap is already
        carved), but the WARN pinpoints the cause instead of a mysterious late OOM.
        """
        if self._overshoot_warned:
            return
        actual = self._batch_length(batch)
        if actual is None:
            return
        if actual > self.length:
            self._overshoot_warned = True
            logger.warning(f'[pad_warmup] A batch exceeded the warm-up length '
                           f'({actual} > {self.length}): the peak was underestimated, the '
                           f'reserved segments will not cover this batch.')

    @staticmethod
    def _is_first_worker() -> bool:
        """True in the dataloader worker that produces the FIRST batch of the epoch.

        With `dataloader_num_workers > 1` every worker holds its own collator copy, and the
        claim used to be first-come-first-served: whichever worker finished collating first
        padded ITS batch. But a DataLoader hands the batches back in worker order (worker i
        produces batch i), so the padded batch could be batch 2 or 3 -- the small batches
        ahead of it already carve the heap into the small segments the warm-up exists to
        prevent. Worker 0 always produces batch 0, so the claim is restricted to it.
        """
        try:
            from torch.utils.data import get_worker_info
            info = get_worker_info()
        except Exception:
            return True
        return info is None or info.id == 0

    def _claimed(self) -> bool:
        """Atomically claim the one-shot warm-up for this rank; False if already taken."""
        if not self._is_first_worker():
            return False
        if not self.flag_path:
            return True
        try:
            fd = os.open(self.flag_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return True
        except FileExistsError:
            return False
        except Exception as e:
            logger.warning(f'[pad_warmup] Cannot claim {self.flag_path} ({e}); warm up anyway.')
            return True

    def __call__(self, batch: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        self._check_overshoot(batch)
        if self.num_batches <= 0:
            return self.collator(batch, **kwargs)
        self.num_batches -= 1
        if not self._claimed():
            # Already warmed up in an earlier epoch (a fresh worker was forked), or this is
            # not the worker that produces the first batch.
            logger.info(f'[pad_warmup] Warm-up not claimed here, skip padding '
                        f'(pid {os.getpid()}).')
            return self.collator(batch, **kwargs)
        logger.info(f'[pad_warmup] Padding the first batch to {self.length} (pid {os.getpid()}).')
        n0 = len(batch)
        try:
            return self.collator(batch, pad_to_length=self.length, **kwargs)
        except Exception as e:
            # Best-effort feature: a dummy that some template field cannot accommodate must
            # degrade to the unpadded batch, never take training down at step 0. The dummy is
            # only ever appended; drop it before the retry (in-place cp-padding of the real
            # rows is idempotent and matches the normal collate).
            del batch[n0:]
            logger.warning(f'[pad_warmup] Warm-up padding failed ({e}); forwarding the batch '
                           f'unpadded.')
            return self.collator(batch, **kwargs)

    def __repr__(self) -> str:
        return (f'PadWarmupCollator(length={self.length}, num_batches={self.num_batches}, '
                f'flag_path={self.flag_path}, collator={self.collator})')


def _warmup_flag_path(args) -> Optional[str]:
    """Per-(output_dir, rank) one-shot flag; `wrap_pad_warmup` deletes it up front so a
    leftover from an earlier run can never suppress this run's warm-up.

    Keyed by the GLOBAL rank, not the local one: on a shared `SWIFT_PAD_WARMUP_CACHE_DIR`
    (NFS), every node's `local_rank=0` would otherwise collide on ONE flag file and only a
    single node would ever warm up."""
    state_dir = _state_dir()
    if not state_dir:
        return None
    rank = os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0'))
    key = f'{getattr(args, "output_dir", "")}:{rank}'
    return os.path.join(state_dir, f'warmed_{hashlib.md5(key.encode()).hexdigest()[:16]}.flag')


def _agree_enable(flag: int) -> bool:
    """All ranks must make the SAME enable decision; disagreement disables everywhere.

    A rank that skips a collective its peers enter (the scan's rendezvous, or the
    communicator warm-up) hangs the whole job, so "one rank says no" must mean "everybody
    says no". Called unconditionally at two points: before the scan (so a rank whose
    `SWIFT_PAD_WARMUP` differs cannot strand the minority inside the scan's rendezvous)
    and after the peak length is resolved (so a rank that failed to compute it cannot
    leave the others waiting in `_prewarm_communicators`).
    """
    try:
        import torch
        import torch.distributed as dist
        if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
            return bool(flag)
        mine = torch.tensor([int(flag)], dtype=torch.long, device=torch.cuda.current_device())
        out = [torch.zeros(1, dtype=torch.long, device=mine.device)
               for _ in range(dist.get_world_size())]
        dist.all_gather(out, mine)
        votes = {int(t[0].item()) for t in out}
        if votes == {1}:
            return True
        if votes != {0}:
            logger.warning(f'[pad_warmup] The ranks disagree on enabling pad-warmup '
                           f'(flags={sorted(votes)}); disabling it everywhere.')
        return False
    except Exception as e:
        logger.warning(f'[pad_warmup] Cannot agree on the enable flag ({e}); disabling.')
        return False


def wrap_pad_warmup(data_collator: Callable[..., Dict[str, Any]],
                    args,
                    dataset=None,
                    template=None,
                    optimizer=None) -> Callable[..., Dict[str, Any]]:
    """Pad the first batch of the run to the peak packed length, see the module docstring."""
    try:
        want = 1 if get_env_args('SWIFT_PAD_WARMUP', bool, False) else 0
    except Exception:
        want = 0
    if not _agree_enable(want):
        return data_collator
    if want:
        # Create (and agree on) the scan group NOW, before any code path that could return
        # early on one rank only -- `new_group` rendezvouses all ranks. Self-guarded:
        # any failure falls back to the default group, agreed across ranks (MIN all_reduce).
        _scan_process_group()
    try:
        length = get_pad_warmup_length(args, dataset, template)
    except Exception as e:
        # This file is a best-effort addition: nothing in it may ever take training down.
        logger.warning(f'[pad_warmup] Disabled: error while resolving the warm-up length: {e}')
        length = None
    if not _agree_enable(1 if length is not None else 0):
        return data_collator
    # Pre-carve runtime preparation: communicators, DeepEP buffer, allocator settle,
    # dispatcher release -- all while the device still has free memory.
    _prewarm_runtime(args, optimizer)
    # Single transparent CE-grad reserve: hold one (length, vocab // tp) bf16 block
    # so the allocator's peak-segment shape is carved before training; base.py registers
    # forward pre/post hooks on wrapped_models[0] that re-acquire the block at every forward
    # start and release it at every forward end for the CE backward's gradient allocation.
    # No length slack: in the exact modes the resolved length is already >= the true peak
    # (alignment rounds UP); partial-mode undershoot simply degrades the reserve for the
    # missed batches (never fatal), and `_check_overshoot` is the guard that reports it.
    # Default ON (run37 production-validated); SWIFT_PAD_WARMUP_CE_RESERVE=0 opts out.
    # Residual risk of the default: a run that barely fits WITHOUT fragmentation protection
    # now carries the reserve through forward -- if that alone OOMs, opt out with =0.
    if os.environ.get('SWIFT_PAD_WARMUP_CE_RESERVE', '1') != '0':
        global _CE_RESERVE_BUF, _CE_RESERVE_SHAPE
        try:
            # The CE backward lives on the LAST pipeline stage only: holding the block on
            # any other stage would be pure waste (and could OOM a stage that never needed
            # it). VPP interleaves chunk forwards, where this single-block hook protocol is
            # unvalidated -- skip there too. When the stage cannot be told (megatron not
            # initialized, e.g. offline), holding stays the protective default.
            if getattr(args, 'virtual_pipeline_model_parallel_size', None):
                hold = False
            else:
                from megatron.core import parallel_state as mpu
                hold = bool(mpu.is_pipeline_last_stage())
        except Exception:
            hold = True
        if not hold:
            logger.info_once('[pad_warmup] CE reserve skipped: not the last pipeline '
                             'stage (or VPP).')
        else:
            vocab = getattr(args, 'padded_vocab_size', None) or getattr(args, 'vocab_size', None) or 248320
            tp = getattr(args, 'tensor_model_parallel_size', 1) or 1
            cp = max(1, int(getattr(args, 'context_parallel_size', 1) or 1))
            # The block must match the logits/CE-gradient shape THIS rank allocates:
            # * `tp` shards the vocab dimension;
            # * `cp` shards the SEQUENCE dimension -- a CP rank holds `seq/cp` tokens
            #   (`reconstruct_tensor_cp`: "[batch_size, seq_len/cp_size] -> [batch_size,
            #   seq_len]"), so sizing on the full length over-reserved cp-fold (v9 bug: at
            #   run37 shapes that is ~18 GiB of pure waste per extra cp rank, in the
            #   "overshoot kills a healthy run" direction);
            # * the padded layout feeds `mbs` rows of `length` (padding_free packs one row),
            #   so the token count scales with the row count.
            rows = 1 if getattr(args, 'padding_free', False) \
                else max(1, int(getattr(args, 'micro_batch_size', 1) or 1))
            _CE_RESERVE_SHAPE = (max(1, length // cp * rows), vocab // tp)
            _CE_RESERVE_BUF = None
            try:
                acquire_ce_reserve_buf()
            except Exception as e:
                # Best-effort: a reserve that cannot be allocated (or no CUDA device, e.g. an
                # offline demo run) must not take training down -- disable the reserve entirely
                # so the forward hooks stay inert no-ops.
                _CE_RESERVE_SHAPE = None
                logger.warning(f'[pad_warmup] CE reserve allocation failed ({e}); reserve disabled.')
            if _CE_RESERVE_SHAPE is not None:
                logger.info_once(f'[pad_warmup] Holding single CE-grad reserve {_CE_RESERVE_SHAPE} bf16; '
                                 f'each forward end releases it for the CE backward.')
    logger.info_once(f'[pad_warmup] warmup: pad the first {_WARMUP_BATCHES} batch(es) '
                     f'per dataloader worker to {length} '
                     f'(padding_free={getattr(args, "padding_free", False)}).')
    flag_path = _warmup_flag_path(args)
    if flag_path:
        try:
            # A flag left behind by an EARLIER run (same output_dir, e.g. a resume) would make
            # this run skip the warm-up entirely; the claim must live and die with the run.
            os.remove(flag_path)
        except OSError:
            pass
    return PadWarmupCollator(data_collator, length, _WARMUP_BATCHES, flag_path=flag_path,
                             padding_free=bool(getattr(args, 'padding_free', False)),
                             cp_size=max(1, int(getattr(args, 'context_parallel_size', 1) or 1)))
