#!/usr/bin/env python3
"""Executable equivalence test: REAL mcore_bridge.split_cp_inputs vs the demo mirror.

demo_cp.py verifies the CP zigzag semantics against a *mirror* reimplementation. This test
closes the gap by driving the actual `mcore_bridge.utils.megatron_utils.split_cp_inputs`
(CPU: mpu is stubbed, Tensor.cuda is a no-op) on randomized segment layouts, checking:

1. shape balance: every rank gets total/cp tokens;
2. mirror equivalence: the real function returns exactly the demo's zigzag_split output;
3. invertibility: gathering the cp ranks' shards reconstructs the original packed row
   bit-exactly (the property the demo only proved for the mirror).

Run inside the training image:  python test_cp_split_real.py
"""
import random
import sys
import types

import torch

# CPU-only test box: neutralize every CUDA touchpoint BEFORE importing mcore_bridge
# (its module import probes torch.cuda.get_device_name).
torch.Tensor.cuda = lambda self, *a, **kw: self
import torch.cuda
torch.cuda._lazy_init = lambda *a, **kw: None
torch.cuda.get_device_name = lambda *a, **kw: 'cpu-stub'
torch.cuda.get_device_capability = lambda *a, **kw: (8, 0)
_orig_tensor = torch.tensor
torch.tensor = lambda *a, **kw: _orig_tensor(*a, **{k: v for k, v in kw.items()
                                                   if k != 'pin_memory'})


def install_stubs(cp_size: int, cp_rank: int):
    """Stub parallel_state's CP accessors (CPU-only test box)."""
    import mcore_bridge.utils.megatron_utils as mu
    mu.mpu = types.SimpleNamespace(
        get_context_parallel_world_size=lambda: cp_size,
        get_context_parallel_rank=lambda: cp_rank)
    return mu.split_cp_inputs


def mirror_zigzag(x, segs, cp, rank):
    """demo_cp.py's reference: per segment view (2cp, len/2cp), chunks (rank, 2cp-1-rank)."""
    out = []
    for a, b in segs:
        v = x[a:b].view(2 * cp, -1)
        out += [v[rank], v[2 * cp - 1 - rank]]
    return torch.cat(out)


def main():
    random.seed(20261112)
    n_cases = 0
    for cp in (2, 4, 8):
        for trial in range(50):
            n_seg = random.randint(1, 6)
            segs, total = [], 0
            for _ in range(n_seg):
                # random segment lengths, each an exact 2*cp multiple (the collator's guarantee)
                seg = random.randint(1, 40) * 2 * cp
                segs.append((total, total + seg))
                total += seg
            x = torch.randint(0, 32000, (total,), dtype=torch.long)
            cu = torch.tensor([0] + [b for _, b in segs], dtype=torch.int32)
            shards = []
            for rank in range(cp):
                real = install_stubs(cp, rank)
                got = real(x, cu, dim=0)
                want = mirror_zigzag(x, segs, cp, rank)
                assert got.shape == (total // cp,), (cp, trial, rank, got.shape)
                assert torch.equal(got, want), f'real != mirror (cp={cp}, trial={trial}, rank={rank})'
                shards.append(got)
                n_cases += 1
            # invertibility: regroup the shards back into the original order
            recon = torch.empty(total, dtype=x.dtype)
            off = {r: 0 for r in range(cp)}
            for a, b in segs:
                c = (b - a) // (2 * cp)
                for rank in range(cp):
                    first = shards[rank][off[rank]:off[rank] + c]
                    second = shards[rank][off[rank] + c:off[rank] + 2 * c]
                    recon[a + rank * c:a + (rank + 1) * c] = first
                    recon[a + (2 * cp - 1 - rank) * c:a + (2 * cp - rank) * c] = second
                    off[rank] += 2 * c
            assert torch.equal(recon, x), f'reconstruction failed (cp={cp}, trial={trial})'
    print(f'TEST OK: real split_cp_inputs == mirror, invertible, balanced '
          f'({n_cases} rank-shards over cp in (2,4,8) x 50 random layouts)')


if __name__ == '__main__':
    sys.exit(main())
