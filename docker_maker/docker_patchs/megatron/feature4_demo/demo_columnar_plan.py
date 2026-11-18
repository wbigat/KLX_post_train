#!/usr/bin/env python3
"""Demo 5: columnar dedup-plan building + decode=False (feature4 v6, batch 4).

Runs against REAL `datasets`/`pyarrow` (in the training image): the dedup plan must be
built from whole-column reads instead of n row random-accesses, and decoding media
features (HF Image/Video with decode=True) must be read UNDECODED -- which both removes
the per-rank full-dataset decode and restores dedup for those datasets (row-wise they
arrive as PIL objects, which `_freeze_media` must refuse, so today every row is encoded).

Scenarios:
  A  string media columns (the ms-swift shape): columnar plan == the row-wise reference,
     zero row reads, zero image decodes;
  B  Image(decode=True), path-based: plan built off the arrow struct with ZERO PIL decodes
     and zero row reads, dedup restored (50 unique x 4 QA -> 50 representatives);
  C  Image with embedded bytes (no path): never deduped (bytes are not an identity), but
     the byte LENGTH becomes the tail-first size proxy;
  D  Sequence(Image): per-row image lists dedup by the path list, sizes summed;
  E  select() view (indices mapping): the plan follows the LOGICAL row order;
  F  text-only dataset: O(1) plan, every row encoded, zero row reads;
  G  end-to-end `_scan_lengths` on a decode=True dataset: dedup decodes each unique image
     exactly once, lengths match the reference, `_expand_dedup` reads text column-wise;
  H  performance: 50k-row plan build, columnar vs the simulated row-wise loop.

Run (inside the training image):  python3 demo_columnar_plan.py
"""
import importlib.util
import io
import os
import re
import sys
import tempfile
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
PATCH = os.environ.get('DEMO_PATCH', os.path.join(HERE, '..', 'feature4_megatron_pad_warmup-v9.patch'))


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
    if os.environ.get('PW_SOURCE', 'local') == 'local':
        src_path = os.path.join(HERE, 'pad_warmup.py')
    else:  # exercise the shipped patch itself
        lines = open(PATCH).read().split('\n')
        start = next(i for i, l in enumerate(lines) if l.startswith('@@ -0,0 +1,'))
        count = int(re.match(r'@@ -0,0 \+1,(\d+) @@', lines[start]).group(1))
        src = '\n'.join(l[1:] for l in lines[start + 1:start + 1 + count]) + '\n'
        src_path = os.path.join(tempfile.mkdtemp(), 'pad_warmup.py')
        open(src_path, 'w').write(src)
    spec = importlib.util.spec_from_file_location('pad_warmup_demo5', src_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['pad_warmup_demo5'] = mod  # the fork pool unpickles _scan_row by module ref
    spec.loader.exec_module(mod)
    return mod


class CountedDS:
    """Delegating wrapper exposing the HF Dataset interface; counts ROW accesses."""

    def __init__(self, ds):
        self._ds = ds
        self.row_reads = 0

    def __len__(self):
        return len(self._ds)

    def __getattr__(self, k):  # column_names / features / data / _indices / ...
        return getattr(self._ds, k)

    def cast_column(self, *a, **k):
        return self._ds.cast_column(*a, **k)

    def __getitem__(self, k):
        if isinstance(k, str):
            return self._ds[k]
        self.row_reads += 1
        return self._ds[k]


class PilCounter:
    """Context manager counting real image decodes (HF Image decode goes through PIL.open)."""

    def __enter__(self):
        import PIL.Image
        self.n = 0
        self._real = PIL.Image.open

        def counting(*a, **k):
            self.n += 1
            return self._real(*a, **k)
        PIL.Image.open = counting
        return self

    def __exit__(self, *a):
        import PIL.Image
        PIL.Image.open = self._real


def ref_rowwise_plan(pw, rows):
    """The pre-change per-row algorithm, run on plain row dicts (the reference)."""
    reps, todo, dup_of, sizes = {}, [], {}, {}
    for i, row in enumerate(rows):
        key = pw._row_media_key(row)
        if key is None:
            todo.append(i)
            sizes[i] = pw._row_media_size(row)
            continue
        rep = reps.get(key)
        if rep is None:
            reps[key] = i
            todo.append(i)
            sizes[i] = pw._row_media_size(row)
        else:
            dup_of[i] = rep
    return todo, dup_of, sizes


def make_images(tmp, n, base=8):
    from PIL import Image as PILImage
    os.makedirs(tmp, exist_ok=True)
    paths = []
    for i in range(n):
        p = os.path.join(tmp, f'im{i:04d}.png')
        # noise pixels: the file size grows strictly with the dimensions
        w = base + i
        PILImage.frombytes('RGB', (w, base), os.urandom(w * base * 3)).save(p)
        paths.append(p)
    return paths


def msgs(i, tag='q'):
    return [{'role': 'user', 'content': f'{tag}{i}'}]


OK = 0


def check(name, cond, detail=''):
    global OK
    assert cond, f'{name} {detail}'
    OK += 1
    print(f'  ok {name} {detail}')


def scenario_a(pw, tmp):
    """String media columns: columnar == row-wise reference, zero row reads."""
    from datasets import Dataset
    paths = make_images(tmp, 25)
    rows = [{'videos': [paths[i % 25], paths[(i + 7) % 25]], 'messages': msgs(i)}
            for i in range(100)] + [{'videos': None, 'messages': msgs(1000 + i, 't')}
                                     for i in range(10)]
    ds = Dataset.from_list(rows)
    ref = ref_rowwise_plan(pw, rows)
    cds = CountedDS(ds)
    with PilCounter() as pil:
        plan = pw._scan_dedup_plan(cds, len(ds))
    check('A: 列式计划与行式参考完全一致', plan == ref)
    check('A: 计划构建零行访问', cds.row_reads == 0, f'(row_reads={cds.row_reads})')
    check('A: 计划构建零解码', pil.n == 0)
    check('A: 去重生效', len(plan[0]) < len(rows) and len(plan[1]) > 0,
          f'(todo={len(plan[0])}/{len(rows)})')


def scenario_b(pw, tmp):
    """Image(decode=True), path-based: zero-decode plan off the arrow struct, dedup restored."""
    from datasets import Dataset, Image
    paths = make_images(os.path.join(tmp, 'b'), 50)
    rows = [{'images': paths[i % 50], 'messages': msgs(i)} for i in range(200)]
    ds = Dataset.from_list(rows).cast_column('images', Image())
    assert ds.features['images'].decode  # the pathology: row access would DECODE
    # The arrow fast path must serve this column directly.
    got = pw._arrow_media_column(ds, 'images', len(ds))
    check('B: arrow 零解码通道可用', got is not None)
    cds = CountedDS(ds)
    with PilCounter() as pil:
        todo, dup_of, sizes = pw._scan_dedup_plan(cds, len(ds))
    check('B: 零 PIL 解码', pil.n == 0, f'(opens={pil.n})')
    check('B: 零行访问', cds.row_reads == 0)
    check('B: 去重恢复: 200 行 -> 50 代表', len(todo) == 50 and len(dup_of) == 150)
    check('B: 大小代理 == 文件字节', all(sizes[i] == os.path.getsize(paths[i]) for i in todo))
    # Same plan as the row-wise reference over the decode=False view.
    view = ds.cast_column('images', Image(decode=False))
    ref = ref_rowwise_plan(pw, [view[i] for i in range(len(ds))])
    check('B: 与 decode=False 视图的行式参考一致', (todo, dup_of, sizes) == ref)
    # Row-wise on the decode=True dataset could not dedup at all (PIL is unfreezable).
    check('B: 行式旧路径无去重能力（对照）', pw._row_media_key(ds[0]) is None)


def scenario_c(pw, tmp):
    """Embedded bytes (no path): never deduped; the byte length is the size proxy."""
    from datasets import Dataset, Image
    from PIL import Image as PILImage
    raws = []
    for i in range(4):
        buf = io.BytesIO()
        w = 16 + 16 * i
        PILImage.frombytes('RGB', (w, 16), os.urandom(w * 16 * 3)).save(buf, format='PNG')
        raws.append(buf.getvalue())
    rows = [{'images': raws[i % 4], 'messages': msgs(i)} for i in range(8)]
    ds = Dataset.from_list(rows).cast_column('images', Image())
    got = pw._arrow_media_column(ds, 'images', len(ds))
    check('C: arrow 通道可用', got is not None)
    values, sizes = got
    check('C: bytes 载荷被摘除（marker 占位）',
          all(v == {'path': None, 'bytes': pw._BYTES_MARKER} for v in values))
    check('C: 大小代理 == 内嵌字节数', sizes == [len(raws[i % 4]) for i in range(8)])
    todo, dup_of, _ = pw._scan_dedup_plan(CountedDS(ds), len(ds))
    check('C: 内嵌字节行全部照扫（不以长度冒充身份）', len(todo) == 8 and not dup_of)
    ordered = pw._order_tail_first(todo, _)
    check('C: tail-first 把大行排前', ordered[0] == 3 and ordered[-1] == 4)


def scenario_d(pw, tmp):
    """Sequence(Image): lists of images dedup by the path list."""
    from datasets import Dataset, Image, Sequence
    paths = make_images(os.path.join(tmp, 'd'), 6)
    rows = [{'images': [paths[i % 6], paths[(i + 1) % 6]], 'messages': msgs(i)}
            for i in range(12)]
    ds = Dataset.from_list(rows).cast_column('images', Sequence(Image()))
    cds = CountedDS(ds)
    with PilCounter() as pil:
        todo, dup_of, sizes = pw._scan_dedup_plan(cds, len(ds))
    check('D: Sequence(Image) 零解码零行访问', pil.n == 0 and cds.row_reads == 0)
    check('D: 12 行 -> 6 代表', len(todo) == 6 and len(dup_of) == 6)
    check('D: 大小为两图之和',
          sizes[0] == os.path.getsize(paths[0]) + os.path.getsize(paths[1]))


def scenario_e(pw, tmp):
    """select() view: the plan follows the logical row order through the indices mapping."""
    from datasets import Dataset, Image
    paths = make_images(os.path.join(tmp, 'e'), 20)
    rows = [{'images': paths[i % 20], 'messages': msgs(i)} for i in range(80)]
    ds = Dataset.from_list(rows).cast_column('images', Image())
    view = ds.shuffle(seed=42).select(range(60))
    assert view._indices is not None
    cds = CountedDS(view)
    with PilCounter() as pil:
        plan = pw._scan_dedup_plan(cds, len(view))
    check('E: 视图计划零解码零行访问', pil.n == 0 and cds.row_reads == 0)
    dv = view.cast_column('images', Image(decode=False))
    ref = ref_rowwise_plan(pw, [dv[i] for i in range(len(view))])
    check('E: 计划与视图行式参考一致（逻辑序）', plan == ref)


def scenario_f(pw):
    """Text-only dataset: O(1) plan."""
    from datasets import Dataset
    ds = Dataset.from_list([{'messages': msgs(i)} for i in range(1000)])
    cds = CountedDS(ds)
    todo, dup_of, sizes = pw._scan_dedup_plan(cds, len(ds))
    check('F: 纯文本计划 O(1)', todo == list(range(1000)) and not dup_of
          and cds.row_reads == 0)


def scenario_g(pw, tmp):
    """End-to-end `_scan_lengths` on a decode=True dataset."""
    from datasets import Dataset, Image
    paths = make_images(os.path.join(tmp, 'g'), 10, base=16)

    class DemoTemplate:
        tokenizer = None

        def encode(self, row, return_length=True):
            media = 0
            for im in ([row['images']] if not isinstance(row['images'], list) else row['images']):
                media += 1000 + im.size[0] * im.size[1]
            text = sum(len(m.get('content', '')) for m in row.get('messages') or [])
            return {'lengths': [media + text]}

    rows = [{'images': paths[i % 10], 'messages': msgs(i)} for i in range(40)]
    ds = Dataset.from_list(rows).cast_column('images', Image())
    pw._SCAN_TEMPLATE = DemoTemplate()
    with PilCounter() as pil:
        lengths = pw._scan_lengths(CountedDS(ds), len(ds), num_proc=4)
    check('G: 每个唯一图像恰好解码一次', pil.n == 10, f'(opens={pil.n})')
    tpl = DemoTemplate()
    ref = [max(tpl.encode(ds[i])['lengths']) for i in range(len(ds))]
    # dups inherit their representative's value (tokenizer=None -> no text delta)
    view = ds.cast_column('images', Image(decode=False))
    _, dup_of, _ = ref_rowwise_plan(pw, [view[i] for i in range(len(ds))])
    for i, rep in dup_of.items():
        ref[i] = ref[rep]
    check('G: 端到端长度与参考一致', lengths == ref)

    # `_expand_dedup` with a tokenizer: the text delta applies, and the messages column is
    # fetched column-wise (zero row reads).
    class TokTemplate(DemoTemplate):
        @staticmethod
        def tokenizer(text, add_special_tokens=False):
            return {'input_ids': list(range(len(text)))}

    pw._SCAN_TEMPLATE = TokTemplate()
    todo, dup_of, sizes = ref_rowwise_plan(pw, [view[i] for i in range(len(ds))])
    base = dict(zip(todo, pw._scan_rows([ds[i] for i in todo], 4)))
    cds = CountedDS(ds)
    out = pw._expand_dedup(cds, todo, [base[i] for i in todo], dup_of, len(ds))
    check('G: expand 零行访问（messages 整列取）', cds.row_reads == 0)
    exp = {i: base[i] for i in todo}
    for i, rep in dup_of.items():
        exp[i] = max(0, base[rep] + len(f'q{i}') - len(f'q{rep}'))
    check('G: 文本增量修正正确', out == [exp[i] for i in range(len(ds))])


def scenario_h(pw, tmp):
    """50k-row plan build: columnar vs the simulated row-wise loop."""
    from datasets import Dataset
    n = 50_000
    ds = Dataset.from_list([{'videos': f'/v/{i % 5000}.mp4', 'messages': msgs(i)}
                            for i in range(n)])
    t0 = time.perf_counter()
    plan = pw._scan_dedup_plan(CountedDS(ds), n)
    t_col = time.perf_counter() - t0
    t0 = time.perf_counter()
    ref = ref_rowwise_plan(pw, [ds[i] for i in range(n)])
    t_row = time.perf_counter() - t0
    check('H: 5 万行计划一致', plan == ref)
    check('H: 列式显著更快', t_col < t_row / 5,
          f'(columnar {t_col:.3f}s vs row-wise {t_row:.3f}s)')


def scenario_i(pw):
    """`_text_char_sizes` correctness (2026-11-16 regression): the v9 raw-arrow fast path
    bypassed dataset formatting -- on a real 80k-row dataset `messages` are JSON strings in
    the raw table and only the formatted view restores list-of-dict, so zero text was
    extracted on EVERY row, the equal-frame tie-break died, the calibration head misplaced,
    and the resolved peak undershot by 3.1% (measured). Fix: read the formatted column
    only. These checks pin the values against the formatted row-by-row reference."""
    from datasets import Dataset

    def ref(ds, i):
        return len(pw._row_text({'messages': ds[i].get('messages')}))

    rows = [{'messages': [{'role': 'user', 'content': f'question {i} ' + 'x' * (i % 7)}]}
            for i in range(30)]
    # shuffle view: counts must follow the LOGICAL row order
    ds = Dataset.from_list(rows).shuffle(seed=42)
    got = pw._text_char_sizes(ds, list(range(len(ds))))
    check('I: shuffle 视图挂对行（_indices 映射）',
          all(got[i] == ref(ds, i) for i in range(len(ds))))
    # select view (split_dataset_ratio path)
    ds = Dataset.from_list(rows).select(list(range(7, 30)))
    got = pw._text_char_sizes(ds, list(range(len(ds))))
    check('I: select 视图挂对行', all(got[i] == ref(ds, i) for i in range(len(ds))))
    # rare message shapes -- each with its OWN dataset (arrow cannot unify mixed shapes):
    # ShareGPT from/value, bare dict, bare string, content as a list of plain strings.
    # The parser must not silently extract zero text.
    weird_sets = [
        [{'messages': [{'from': 'human', 'value': f'sg text {i}'}]} for i in range(10)],
        [{'messages': {'role': 'user', 'content': f'bare dict {i}'}} for i in range(10)],
        [{'messages': f'bare string {i}'} for i in range(10)],
        [{'messages': [{'role': 'user', 'content': [f'part {i}', 'more']}]}
         for i in range(10)],
    ]
    for k, weird in enumerate(weird_sets):
        ds = Dataset.from_list(weird)
        got = pw._text_char_sizes(ds, list(range(len(ds))))
        check(f'I: 异形 schema{k} 解析出文本且与参考一致',
              all(got[i] == ref(ds, i) and got[i] > 0 for i in range(len(ds))),
              f'(head={got[0]})')
    # genuinely text-free rows: zeros are correct, no crash
    ds = Dataset.from_list([{'messages': [{'role': 'user', 'content': None}]}] * 5)
    got = pw._text_char_sizes(ds, list(range(5)))
    check('I: 无文本行返回 0 且不炸', all(v == 0 for v in got.values()))
    # The incident shape itself: the raw arrow table stores messages as JSON *strings* and
    # only the formatted view parses them into list-of-dict. Any raw-table read extracts
    # zero (or garbage) text; the fixed function reads the formatted column only.
    import json as _json
    base = Dataset.from_list(
        [{'messages': [_json.dumps({'role': 'user', 'content': f'js {i}'})]}
         for i in range(20)])

    class JsonDS:
        """messages stored as JSON strings; formatted access parses them (user's shape)."""
        def __init__(self, d):
            self._d = d

        def __len__(self):
            return len(self._d)

        def __getitem__(self, k):
            if k == 'messages':
                return [[_json.loads(m) for m in row] for row in self._d['messages']]
            if isinstance(k, str):
                return self._d[k]
            return self._d[int(k)]

    ds = JsonDS(base)
    got = pw._text_char_sizes(ds, list(range(len(ds))))
    check('I: JSON 串消息经格式化路径正确解析（事故形状）',
          all(got[i] == len(f'js {i}') for i in range(len(ds))))
    # no messages column at all -> empty map, no crash
    ds = Dataset.from_list([{'videos': ['/v.mp4']}] * 3)
    check('I: 无 messages 列返回空表', pw._text_char_sizes(ds, [0, 1, 2]) == {})


def scenario_j(pw, tmp):
    """Nested frame sequences (videos=[[f1..fK]], hotfix 1): the frame-count shortcut must
    apply at depth on BOTH plan paths, so columnar and row-wise agree on keys AND sizes,
    and no frame path is ever stat'ed (the prewarm filter skips them at any depth)."""
    from datasets import Dataset
    fdir = os.path.join(tmp, 'j')
    os.makedirs(fdir, exist_ok=True)
    vids = []
    for v in range(6):
        frames = []
        for k in range(5):
            p = os.path.join(fdir, f'v{v}_f{k}.jpg')
            with open(p, 'wb') as f:
                f.truncate(100 + v * 10 + k)
            frames.append(p)
        vids.append(frames)
    scalar = os.path.join(fdir, 'scalar.mp4')
    with open(scalar, 'wb') as f:
        f.truncate(7777)
    # Arrow cannot unify list<list<str>> with list<str> in one column, so the shapes are
    # separated by column: videos = nested frame sequences, audios = a flat scalar path.
    # 24 rows: 6 nested videos x 4 QA (dup 4:1), plus 2 rows reusing video 0.
    rows = [{'videos': [vids[i % 6]], 'audios': [scalar], 'messages': msgs(i)}
            for i in range(24)] + [
        {'videos': [vids[0]], 'audios': [scalar], 'messages': msgs(100)},
        {'videos': [vids[0]], 'audios': [scalar], 'messages': msgs(101)}]
    ds = Dataset.from_list(rows)   # videos: list<list<str>>, audios: list<str>
    ref = ref_rowwise_plan(pw, rows)
    pw._stat_size.cache_clear()
    plan = pw._scan_dedup_plan(CountedDS(ds), len(ds))
    info = pw._stat_size.cache_info()
    check('J: 嵌套帧序列列式计划 == 行式参考（键+尺寸逐位）', plan == ref)
    check('J: 帧路径零 stat（只 stat 了标量路径）', info.misses == 1,
          f'(misses={info.misses}, hits={info.hits})')
    todo, dup_of, sizes = plan
    check('J: 去重 26 行 -> 6 代表', len(todo) == 6 and len(dup_of) == 20)
    check('J: 尺寸 == 帧数 + 标量字节（两列求和）',
          sizes[0] == 5 + 7777, f'({sizes[0]})')


def main():
    pw = load_pad_warmup()
    import datasets
    print(f'datasets {datasets.__version__}')
    tmp = tempfile.mkdtemp(prefix='pw_demo5_')
    scenario_a(pw, os.path.join(tmp, 'a'))
    scenario_b(pw, tmp)
    scenario_c(pw, tmp)
    scenario_d(pw, tmp)
    scenario_e(pw, tmp)
    scenario_f(pw)
    scenario_g(pw, tmp)
    scenario_h(pw, tmp)
    scenario_i(pw)
    scenario_j(pw, tmp)
    print(f'DEMO5 OK: {OK} checks passed')


if __name__ == '__main__':
    main()
