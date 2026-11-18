#!/usr/bin/env python3
"""Demo 1: robustness of the media-payload dedup key (feature4, no torch needed).

The dedup key decides which rows are decoded once. Media columns are NOT always a plain
string or a list of strings -- HF datasets decode media features to dicts, some datasets
nest per-turn lists -- and the old `tuple(value)` key threw `unhashable type` on those,
silently falling back to a full scan. This demo exercises `_freeze_media` /
`_row_media_key` / `_scan_dedup_plan` from the *shipped patch* against every shape.

Run:  python3 demo_dedup_key.py
"""
import importlib.util
import os
import re
import sys
import tempfile
import types

PATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                     'feature4_megatron_pad_warmup-v9.patch')


def load_pad_warmup():
    """Import pad_warmup.py from the patch (single source of truth), swift deps stubbed."""
    swift = types.ModuleType('swift')
    utils = types.ModuleType('swift.utils')

    class _L:
        def __getattr__(self, k):
            return lambda *a, **kw: None
    utils.get_logger = lambda *a, **kw: _L()
    utils.get_env_args = lambda k, t, d: d
    dsu = types.ModuleType('swift.dataset.utils')
    dsu.LazyLLMDataset = type('LazyLLMDataset', (), {})
    sys.modules.update({'swift': swift, 'swift.utils': utils,
                        'swift.dataset': types.ModuleType('swift.dataset'),
                        'swift.dataset.utils': dsu})
    lines = open(PATCH).read().split('\n')
    start = next(i for i, l in enumerate(lines) if l.startswith('@@ -0,0 +1,'))
    count = int(re.match(r'@@ -0,0 \+1,(\d+) @@', lines[start]).group(1))
    src = '\n'.join(l[1:] for l in lines[start + 1:start + 1 + count]) + '\n'
    path = os.path.join(tempfile.mkdtemp(), 'pad_warmup.py')
    open(path, 'w').write(src)
    spec = importlib.util.spec_from_file_location('pad_warmup_demo', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['pad_warmup_demo'] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    pw = load_pad_warmup()
    K = pw._row_media_key
    ok = 0

    def check(name, cond):
        nonlocal ok
        assert cond, name
        ok += 1
        print(f'  ok {name}')

    # plain shapes
    check('str 与 list-of-str 是不同键（不混）', K({'videos': '/v1.mp4'}) != K({'videos': ['/v1.mp4']}))
    check('同值同键', K({'videos': ['/v1.mp4']}) == K({'videos': ['/v1.mp4']}))
    # dict shape (HF media feature), key order must not matter
    check('dict 形态乱序同键',
          K({'videos': {'path': '/v1.mp4', 'fps': 2}}) == K({'videos': {'fps': 2, 'path': '/v1.mp4'}}))
    check('dict 与 str 不同键', K({'videos': {'path': '/v1.mp4', 'fps': 2}}) != K({'videos': '/v1.mp4'}))
    # nested list-list / list-of-dict
    check('list-list 同值同键',
          K({'videos': [['/a.mp4', '/b.mp4'], ['/c.mp4']]}) == K({'videos': [['/a.mp4', '/b.mp4'], ['/c.mp4']]}))
    check('list-list 内容不同不同键',
          K({'videos': [['/a.mp4', '/b.mp4'], ['/c.mp4']]}) != K({'videos': [['/a.mp4', '/b.mp4'], ['/d.mp4']]}))
    check('list-of-dict 同键', K({'images': [{'path': '/i1.png'}]}) == K({'images': [{'path': '/i1.png'}]}))
    # raw bytes: identity too expensive -> never dedup (safe direction)
    check('bytes 载荷不去重', K({'videos': b'raw'}) is None)
    check('dict 内含 bytes 不去重', K({'images': [{'bytes': b'x', 'path': None}]}) is None)
    # tuple / numbers mixed
    check('tuple+数字同键', K({'videos': ('/v1.mp4', 3)}) == K({'videos': ('/v1.mp4', 3)}))
    check('数字 2 与字符串 "2" 不同键（无类型碰撞）',
          K({'videos': [2]}) != K({'videos': ['2']}))

    # objects whose str() is NOT an identity must never be deduped: a PIL-like repr carries a
    # reused memory address and numpy reprs are truncated -> either would merge different media.
    class FakeImage:
        def __repr__(self):
            return '<PIL.Image.Image image mode=RGB size=224x224 at 0x7f0000000000>'

    class FakeArray:
        def __repr__(self):
            return '[1 2 3 ... 998 999]'
    check('PIL 风格对象不去重（地址会被复用）', K({'images': [FakeImage()]}) is None)
    check('ndarray 风格对象不去重（repr 被截断）', K({'images': [FakeArray()]}) is None)
    same_repr = [FakeArray(), FakeArray()]
    check('两个 repr 相同但不同的对象不会被合并',
          K({'images': [same_repr[0]]}) is None and K({'images': [same_repr[1]]}) is None)
    # no media / empty -> no key, row is always encoded
    check('无媒体 -> None', K({'messages': []}) is None)
    check('空媒体列 -> None', K({'videos': []}) is None)
    # multi-column identity
    check('多列组合与列序无关',
          K({'videos': ['/v1.mp4'], 'images': ['/i1.png']}) == K({'images': ['/i1.png'], 'videos': ['/v1.mp4']}))

    # end-to-end: the plan dedups dict-shaped rows and survives odd rows
    ds = [{'videos': {'path': f'/v{v}.mp4', 'fps': 2}, 'messages': [{'content': f'q{q}'}]}
          for v in range(10) for q in range(3)]
    ds.append({'videos': b'raw-bytes', 'messages': [{'content': 'x'}]})   # bytes -> always encoded
    ds.append({'videos': ['/v0.mp4'], 'messages': [{'content': 'plain'}]})  # different shape -> own key
    todo, dup_of, sizes = pw._scan_dedup_plan(ds, len(ds))
    check('端到端: 30 dict 行 -> 10 代表 + bytes + 异形行',
          len(todo) == 12 and len(dup_of) == 20)

    # frame sequences: a pure path list is proxied by its FRAME COUNT -- zero per-frame
    # stat (v9 replaced the sampled-stat scheme; 3.5M frames = 0 stats).
    import os
    import tempfile
    tmp = tempfile.mkdtemp(prefix='pw_frames_')
    frames = []
    for i in range(100):
        p = os.path.join(tmp, f'f{i:03d}.jpg')
        with open(p, 'wb') as f:
            f.truncate(1000 + i)  # near-uniform frame sizes 1000..1099
        frames.append(p)
    pw._stat_size.cache_clear()
    sz_list = pw._strings_size(frames)
    check('100 帧序列零 stat（帧数即代理）', pw._stat_size.cache_info().misses == 0)
    check('size == 帧数', sz_list == 100)
    vals, sz_norm = pw._normalize_media_value(frames)
    check('columnar 路径同值同尺寸（digest 一致）', vals == frames and sz_norm == sz_list)
    check('row-wise 路径同尺寸（digest 一致）', pw._row_media_size({'videos': frames}) == sz_list)
    pw._stat_size.cache_clear()
    check('短序列同规则（帧数、零 stat）', pw._strings_size(frames[:4]) == 4
          and pw._stat_size.cache_info().misses == 0)
    mixed = frames + [{'path': frames[0]}]
    pw._stat_size.cache_clear()
    pw._strings_size(mixed)
    check('混合列表退化为逐路径 stat（保守）', pw._stat_size.cache_info().misses == 100)

    # NESTED frame sequences: videos=[[f1..f100]] (an outer len-1 list). The v8/v9 shortcut
    # only recognized FLAT lists, so this shape fell through to per-frame stats -- 3.5M cold
    # NFS stats (~22ms each) = tens of minutes per rank, all for a proxy meant to be the
    # frame count. The recursive fix takes the shortcut at any depth.
    nested = [frames]
    pw._stat_size.cache_clear()
    check('嵌套帧序列 size==帧数（任意深度递归捷径）', pw._strings_size(nested) == 100)
    check('嵌套帧序列零 stat', pw._stat_size.cache_info().misses == 0)
    check('嵌套形状行式/列式尺寸一致',
          pw._row_media_size({'videos': nested}) == pw._normalize_media_value(nested)[1] == 100)
    other = os.path.join(tmp, 'single.mp4')
    with open(other, 'wb') as f:
        f.truncate(1234)
    check('stat_bearing_strings 跳过嵌套帧序列、保留普通路径',
          list(pw._stat_bearing_strings({'videos': nested, 'audios': [other]})) == [other])
    pw._stat_size.cache_clear()
    pw._prewarm_stat(s for v in [{'videos': nested}] for s in pw._stat_bearing_strings(v))
    check('嵌套形状 prewarm 零 stat（不再预热无用帧路径）',
          pw._stat_size.cache_info().misses == 0)

    # _prewarm_stat with >=256 paths (below the threshold it is a no-op): fills the
    # `_stat_size` cache via the thread pool and logs the elapsed time (regression: the
    # log line used to reference an undefined `t0`, crashing on the first big dataset).
    many = []
    for i in range(300):
        p = os.path.join(tmp, f'pre_{i:03d}.jpg')
        with open(p, 'wb') as f:
            f.truncate(10)
        many.append(p)
    pw._stat_size.cache_clear()
    pw._prewarm_stat(iter(many))     # generator input, 300 unique paths
    pw._stat_size(many[0])
    check('prewarm(300 路径) 灌满 stat 缓存且不炸', pw._stat_size.cache_info().misses == 300
          and pw._stat_size.cache_info().hits == 1)
    pw._prewarm_stat(many[:10])      # below the 256 threshold: no-op, no pool
    check('prewarm(<256 路径) 跳过线程池', pw._stat_size.cache_info().misses == 300)

    # regression: an all-unique dataset must still get its sizes (tail-first order); the old
    # `0 < len(todo) < n` condition silently dropped the plan when nothing was duplicated.
    ds2 = [{'videos': [f'/unique_{i}.mp4'], 'messages': []} for i in range(50)]
    todo2, dup2, sizes2 = pw._dedup_plan_or_all(ds2, 50)
    check('零重复数据集仍保留 sizes（tail-first 不退化为索引序）',
          len(todo2) == 50 and not dup2 and len(sizes2) == 50)

    # cache v3: lengths travel as a base64 int32 blob (10M rows: 60MB JSON + seconds of
    # parse + >10x memory amplification -> 40MB flat decode in ms), and stale entries are
    # garbage-collected (the key changes with the config, so they would pile up forever).
    import json as _json
    import time as _time
    cache_dir = tempfile.mkdtemp(prefix='pw_cache_')
    lengths = [0, 123, 131072, 5] * 100
    blob = pw._lengths_to_blob(lengths)
    check('blob 往返逐位一致', pw._lengths_from_blob(blob, len(lengths)) == lengths)
    check('blob 行数不匹配拒收', pw._lengths_from_blob(blob, len(lengths) + 1) is None)
    check('blob 非法内容拒收', pw._lengths_from_blob('!!!not-b64!!!', 4) is None)
    for i in range(10):
        pw._length_cache_store(os.path.join(cache_dir, f'lengths_v3_{i:016x}.json'),
                               {'version': 3}, lengths, 1)
        _time.sleep(0.01)   # 保证 mtime 可分辨
    kept = sorted(f for f in os.listdir(cache_dir) if f.startswith('lengths_v'))
    check('GC 只保留最新 8 个缓存', len(kept) == pw._LENGTH_CACHE_KEEP)
    loaded = pw._length_cache_load(os.path.join(cache_dir, kept[-1]), len(lengths))
    check('v3 store/load 往返（含 n_failed）', loaded == (lengths, 1))
    check('v3 行数不匹配拒收', pw._length_cache_load(os.path.join(cache_dir, kept[-1]),
                                                    len(lengths) + 1) is None)
    legacy = os.path.join(cache_dir, 'lengths_v2_legacy.json')
    with open(legacy, 'w') as f:
        _json.dump({'version': 2, 'n': len(lengths), 'n_failed': 0, 'lengths': lengths}, f)
    check('v2 旧格式缓存被拒收（版本失效）',
          pw._length_cache_load(legacy, len(lengths)) is None)

    print(f'DEMO1 OK: {ok} 项去重键健壮性检查全部通过')


if __name__ == '__main__':
    main()
