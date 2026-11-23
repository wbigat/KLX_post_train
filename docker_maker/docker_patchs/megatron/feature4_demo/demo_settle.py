#!/usr/bin/env python3
"""Demo 7: settle-then-carve (v7) -- `_settle_allocator_state`.

Before the peak-shaped carve, v7 lands the lazily-created *persistent* allocations that would
otherwise materialise around step 1 and pin/split the pre-carved blocks: Adam's
`exp_avg`/`exp_avg_sq` are allocated by the inner optimizer at its first `step()` in a fresh run
(megatron-core's `init_state_fn` is otherwise only invoked on the checkpoint-load path). This
demo verifies the settle machinery with a real torch.optim.Adam and megatron-shaped wrappers:

1. a fresh (megatron-shaped) optimizer gets its state materialised WITHOUT any step() --
   zeros, no RNG consumed, no gradient involved;
2. a ChainedOptimizer (dense + expert, megatron's MoE layout) settles every sub-optimizer;
3. idempotency: a pre-populated state (the resume-from-checkpoint case) is left untouched;
4. a sub-optimizer whose `init_state_fn` raises only loses its own settle, the others and the
   heap clean-up still run;
5. optimizer=None / a plain torch optimizer (no `init_state_fn`) degrade to the clean-up only.

Run:  python3 demo_settle.py            (loads pad_warmup.py -- the patch's build source)
      DEMO_PATCH=1 python3 demo_settle.py   (extracts pad_warmup.py from the patch instead)
"""
import importlib.util
import os
import sys
import tempfile
import types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))

WARNINGS = []
INFOS = []


def _stub_swift():
    if 'swift' in sys.modules:
        return
    swift = types.ModuleType('swift')
    utils = types.ModuleType('swift.utils')

    class _L:
        def info(self, m, *a, **k):
            INFOS.append(m)

        def warning(self, m, *a, **k):
            WARNINGS.append(m)

        def info_once(self, m, *a, **k):
            INFOS.append(m)

    utils.get_logger = lambda *a, **kw: _L()
    utils.get_env_args = lambda k, t, d: d
    dsu = types.ModuleType('swift.dataset.utils')
    dsu.LazyLLMDataset = type('LazyLLMDataset', (), {})
    sys.modules.update({'swift': swift, 'swift.utils': utils,
                        'swift.dataset': types.ModuleType('swift.dataset'),
                        'swift.dataset.utils': dsu})


_stub_swift()


def load_pad_warmup():
    if os.environ.get('DEMO_PATCH'):
        import re
        lines = open(os.path.join(HERE, '..', 'feature4_megatron_pad_warmup-v10.patch')).read().split('\n')
        start = next(i for i, l in enumerate(lines) if l.startswith('@@ -0,0 +1,'))
        count = int(re.match(r'@@ -0,0 \+1,(\d+) @@', lines[start]).group(1))
        src = '\n'.join(l[1:] for l in lines[start + 1:start + 1 + count]) + '\n'
        path = os.path.join(tempfile.mkdtemp(), 'pad_warmup.py')
        open(path, 'w').write(src)
    else:
        path = os.path.join(HERE, 'pad_warmup.py')
    spec = importlib.util.spec_from_file_location('pad_warmup_demo7', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['pad_warmup_demo7'] = mod
    spec.loader.exec_module(mod)
    return mod


def megatron_init_state_fn(opt, config=None):
    """megatron-core's init_state_fn for Adam (optimizer/__init__.py): zeros_like per param,
    guarded so an already-loaded state is never re-allocated."""
    for group in opt.param_groups:
        for p in group['params']:
            if len(opt.state[p]) == 0:
                opt.state[p]['exp_avg'] = torch.zeros_like(p.data)
                opt.state[p]['exp_avg_sq'] = torch.zeros_like(p.data)


def make_megatron_optimizer(n_params=4, shape=(16, 16)):
    """A megatron-shaped wrapper: .optimizer (inner torch optimizer) + .init_state_fn + .config."""
    params = [torch.randn(*shape, requires_grad=True) for _ in range(n_params)]
    inner = torch.optim.Adam(params, lr=1e-3)
    wrapper = types.SimpleNamespace(optimizer=inner, init_state_fn=megatron_init_state_fn,
                                    config=None)
    return wrapper, params


def main():
    pw = load_pad_warmup()

    print('== 1. fresh optimizer: state materialised without any step() ==')
    opt, params = make_megatron_optimizer()
    assert len(opt.optimizer.state) == 0, 'fresh inner optimizer must have empty state'
    rng_before = torch.random.get_rng_state()
    pw._settle_allocator_state(opt)
    state = opt.optimizer.state
    assert len(state) == len(params), f'{len(state)} != {len(params)}'
    for p in params:
        assert torch.equal(state[p]['exp_avg'], torch.zeros_like(p))
        assert torch.equal(state[p]['exp_avg_sq'], torch.zeros_like(p))
    assert torch.equal(rng_before, torch.random.get_rng_state()), 'settle must not consume RNG'
    assert any('Allocator settled' in m and '4 param shard(s)' in m for m in INFOS), INFOS
    print(f'   {len(state)} params settled, all zeros, RNG untouched  OK')

    print('== 2. ChainedOptimizer (dense + expert): every sub-optimizer settles ==')
    dense, dp = make_megatron_optimizer(3)
    expert, ep = make_megatron_optimizer(5)
    chained = types.SimpleNamespace(chained_optimizers=[dense, expert])
    INFOS.clear()
    pw._settle_allocator_state(chained)
    assert len(dense.optimizer.state) == 3 and len(expert.optimizer.state) == 5
    assert any('8 param shard(s)' in m for m in INFOS), INFOS
    print('   3 + 5 = 8 shards settled  OK')

    print('== 3. idempotency: a pre-populated state (resume) is untouched ==')
    for p in params:
        state[p]['exp_avg'].fill_(1.0)
        state[p]['exp_avg_sq'].fill_(2.0)
    pw._settle_allocator_state(opt)
    for p in params:
        assert torch.all(state[p]['exp_avg'] == 1.0) and torch.all(state[p]['exp_avg_sq'] == 2.0)
    print('   loaded values survive the settle  OK')

    print('== 4. a failing sub-optimizer only loses its own settle ==')
    WARNINGS.clear()
    broken, _ = make_megatron_optimizer(2)

    def boom(inner, config=None):
        raise RuntimeError('no state for you')

    broken.init_state_fn = boom
    chained2 = types.SimpleNamespace(chained_optimizers=[broken, dense])
    pw._settle_allocator_state(chained2)
    assert len(broken.optimizer.state) == 0, 'broken one stays unsettled'
    assert len(dense.optimizer.state) == 3, 'the healthy one still settles'
    assert any('Cannot materialise the optimizer state' in m for m in WARNINGS), WARNINGS
    print('   broken skipped with a warning, healthy settled  OK')

    print('== 5. graceful degradation: None / plain torch optimizer ==')
    pw._settle_allocator_state(None)
    plain = torch.optim.SGD([torch.randn(4)], lr=0.1)   # no .optimizer/.init_state_fn attrs
    pw._settle_allocator_state(plain)
    assert len(plain.state) == 0
    print('   no crash, clean-up only  OK')

    print('== 6. dispatcher release (backport of upstream #8297/#8231) ==')
    import weakref

    class FakeAlltoAllDispatcher:
        def combine_postprocess(self, tokens):
            return tokens.view(-1)  # 0.18.0 body: unpermute/reshape; router state untouched

    assert pw._patch_moe_dispatcher_release(FakeAlltoAllDispatcher) is True
    assert pw._patch_moe_dispatcher_release(FakeAlltoAllDispatcher) is True  # idempotent
    disp = FakeAlltoAllDispatcher()
    probs = torch.randn(8, 4, requires_grad=True).softmax(-1)  # carries an autograd subgraph
    disp.probs = probs
    disp.routing_map = torch.rand(8, 4) > 0.5
    probs_ref = weakref.ref(probs)
    out = disp.combine_postprocess(torch.randn(2, 4))
    assert out.shape == (8,), 'wrapped method must return the original result unchanged'
    assert disp.probs is None and disp.routing_map is None, 'router state must be released'
    del probs
    assert probs_ref() is None, 'tensor + autograd subgraph must be fully freed'
    pw._patch_moe_dispatcher_release(None)  # no megatron on the host -> False, no crash
    print('   probs/routing_map released after combine, result unchanged, weakref freed  OK')

    print('== 7. DeepEP buffer pre-warm (moe_enable_deepep) ==')
    # The first MoE dispatch lazily creates a raw-device deep_ep.Buffer on top of the
    # peak-shaped batch; the pre-warm must create it with the tp_ep group and hidden*2,
    # and never crash when the flag is off or the backend fails.
    mpu = types.ModuleType('megatron.core.parallel_state')
    group = object()
    mpu.get_expert_tensor_and_model_parallel_group = lambda: group
    fused = types.ModuleType('megatron.core.transformer.moe.fused_a2a')
    calls = []
    fused.get_buffer = lambda g, h: calls.append((g, h))
    for name in ('megatron', 'megatron.core', 'megatron.core.transformer',
                 'megatron.core.transformer.moe'):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules['megatron.core.parallel_state'] = mpu
    sys.modules['megatron.core.transformer.moe.fused_a2a'] = fused
    args7 = types.SimpleNamespace(moe_enable_deepep=True, hidden_size=4096)
    pw._prewarm_deepep_buffer(args7)
    assert calls == [(group, 8192)], calls
    calls.clear()
    pw._prewarm_deepep_buffer(types.SimpleNamespace(moe_enable_deepep=False,
                                                    hidden_size=4096))
    assert not calls, 'flag off must not touch the backend'
    WARNINGS.clear()

    def boom(g, h):
        raise RuntimeError('rdma boom')

    fused.get_buffer = boom
    pw._prewarm_deepep_buffer(args7)
    assert any('DeepEP buffer pre-creation failed' in m for m in WARNINGS), WARNINGS
    print('   created with (tp_ep group, hidden*2); flag off / backend failure degrade '
          'safely  OK')

    print('\nDEMO7 OK')


if __name__ == '__main__':
    main()
