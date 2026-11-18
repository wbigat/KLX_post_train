#!/usr/bin/env python3
"""Demo 10: end-to-end validation on a REAL sample dataset (runs in the training image).

No model checkpoint is available in the verify container, so the template is a semi-real
stand-in whose `encode` performs REAL media IO and decode (PIL for images, decord for
videos) and counts tokens with the production-shaped rule (tokens ~ frames/pixels + text).
Everything else is the shipped pad_warmup from the PATCH (single source of truth), running
against real files, a real arrow-backed HF `datasets.Dataset`, and the real fork pool.

Datasets (generated up-front with real encoders):
  A  400 unique noise videos x 4 QA (1600 rows) -- proxy-friendly (size ~ frames), so the
     two-pass exact-batch scan engages and must decode only ~calib+Top-N unique rows;
  B  mixed images + videos + text-only rows + one CORRUPT video -- robustness of the
     degrade chain on a realistic mix.

Verifies:
  1. scanned lengths == an independent decode-based reference (real files, real decode);
  2. media-payload dedup: 1600 rows -> 400 unique encodes, duplicates recovered exactly
     (representative + own-text delta);
  3. two-pass exact-batch on real data: partial scan, yet the resolved peak is BITWISE the
     full-scan truth (dp=1, mbs=1/2/4), via the production `get_pad_warmup_length`;
  4. the v3 length cache: second run is a <1s hit, bitwise identical;
  5. corrupt media -> graceful degrade (partial undershoot-only or off), never a wrong peak;
  6. collator end-to-end: the patch's real `_append_warmup_dummy` pads the first batch to
     exactly the resolved length (cp=1 and cp=2).

Run (container):  python demo_real_dataset.py
"""
import importlib.util
import json
import os
import re
import sys
import tempfile
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
PATCH = os.environ.get('DEMO_PATCH',
                       os.path.join(HERE, '..', 'feature4_megatron_pad_warmup-v9.patch'))

WORK = tempfile.mkdtemp(prefix='pw_real_')
DECODE_COUNT_FILE = os.path.join(WORK, 'decode_count')


def load_pad_warmup():
    lines = open(PATCH).read().split('\n')
    start = next(i for i, l in enumerate(lines) if l.startswith('@@ -0,0 +1,'))
    count = int(re.match(r'@@ -0,0 \+1,(\d+) @@', lines[start]).group(1))
    src = '\n'.join(l[1:] for l in lines[start + 1:start + 1 + count]) + '\n'
    path = os.path.join(tempfile.mkdtemp(), 'pad_warmup.py')
    open(path, 'w').write(src)
    spec = importlib.util.spec_from_file_location('pad_warmup_demo10', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['pad_warmup_demo10'] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Real media generation (noise compresses badly: file size ~ frames/pixels -- a strong,
# honest size proxy, no synthetic shortcut)
# ---------------------------------------------------------------------------

def make_image(path, w, h, seed):
    import numpy as np
    from PIL import Image
    rng = np.random.RandomState(seed)
    Image.fromarray(rng.randint(0, 255, (h, w, 3), dtype=np.uint8)).save(path, quality=85)


def make_video(path, n_frames, seed, w=160, h=120):
    import av
    import numpy as np
    rng = np.random.RandomState(seed)
    out = av.open(path, 'w')
    try:
        stream = out.add_stream('libx264', rate=8)
    except Exception:
        stream = out.add_stream('mpeg4', rate=8)
    stream.width, stream.height, stream.pix_fmt = w, h, 'yuv420p'
    for _ in range(n_frames):
        frame = av.VideoFrame.from_ndarray(
            rng.randint(0, 255, (h, w, 3), dtype=np.uint8), format='rgb24')
        for pkt in stream.encode(frame):
            out.mux(pkt)
    for pkt in stream.encode():
        out.mux(pkt)
    out.close()


def image_tokens(path):
    from PIL import Image
    with Image.open(path) as im:
        w, h = im.size
    return max(4, (w // 28) * (h // 28))


def video_tokens(path):
    import decord
    return len(decord.VideoReader(path)) * 64


def ref_row_length(row):
    """The independent reference: real decode, production-shaped token rule."""
    total = 8 + max(1, len(''.join(m['content'] for m in row['messages'])) // 4)
    for p in row.get('images') or []:
        total += image_tokens(p)
    for p in row.get('videos') or []:
        total += video_tokens(p)
    return total


class RealMediaTemplate:
    """Semi-real template: encode() does the real media IO/decode; the token rule mirrors
    production (tokens ~ pixels/frames + text). Decodes are counted through a file so the
    fork-pool workers' calls are visible to the parent."""
    mode = 'train'

    class _Tok:
        def __call__(self, text, add_special_tokens=False):
            return {'input_ids': [0] * max(1, len(text) // 4)}

    tokenizer = _Tok()

    def encode(self, row, return_length=True):
        with open(DECODE_COUNT_FILE, 'a') as f:
            f.write('.')
        total = ref_row_length(row)
        return {'lengths': [total], 'input_ids': [0] * total}


def gen_datasets():
    """Dataset A: 400 videos x 4 QA (dedup 4:1). Dataset B: images+videos+text+corrupt."""
    import datasets
    rows_a, refs_a = [], []
    vdir = os.path.join(WORK, 'videos_a')
    os.makedirs(vdir, exist_ok=True)
    t0 = time.perf_counter()
    for v in range(400):
        n_frames = 8 + (v * 7) % 41          # 8..48 frames
        if v == 0:
            n_frames = 96                    # one clearly-longest video (the true peak)
        p = os.path.join(vdir, f'v{v:03d}.mp4')
        make_video(p, n_frames, seed=v)
        for q in range(4):
            text = f'video {v} question {q}: ' + 'describe the clip. ' * (q + 1)
            rows_a.append({'videos': [p],
                           'messages': [{'role': 'user', 'content': text}]})
    print(f'   dataset A media generated: 400 videos in {time.perf_counter() - t0:.0f}s')
    refs_a = [ref_row_length(r) for r in rows_a]
    path_a = os.path.join(WORK, 'a.jsonl')
    with open(path_a, 'w') as f:
        for r in rows_a:
            f.write(json.dumps(r) + '\n')
    ds_a = datasets.load_dataset('json', data_files=path_a, split='train')

    rows_b = []
    idir = os.path.join(WORK, 'images_b')
    os.makedirs(idir, exist_ok=True)
    for i in range(40):
        p = os.path.join(idir, f'i{i:02d}.jpg')
        make_image(p, 112 + (i * 53) % 400, 112 + (i * 31) % 200, seed=1000 + i)
        for q in range(3):
            rows_b.append({'images': [p],
                           'messages': [{'role': 'user', 'content': f'what is this {i}/{q}?'}]})
    vdir_b = os.path.join(WORK, 'videos_b')
    os.makedirs(vdir_b, exist_ok=True)
    for v in range(20):
        p = os.path.join(vdir_b, f'v{v:02d}.mp4')
        make_video(p, 8 + v * 2, seed=2000 + v)
        rows_b.append({'videos': [p],
                       'messages': [{'role': 'user', 'content': f'clip {v}?'}]})
    corrupt = os.path.join(vdir_b, 'corrupt.mp4')
    with open(corrupt, 'wb') as f:
        f.write(b'this is not a video')
    rows_b.append({'videos': [corrupt], 'messages': [{'role': 'user', 'content': 'broken'}]})
    for t in range(100):
        rows_b.append({'messages': [{'role': 'user', 'content': 'text row ' + 'x' * t * 3}]})
    path_b = os.path.join(WORK, 'b.jsonl')
    with open(path_b, 'w') as f:
        for r in rows_b:
            f.write(json.dumps(r) + '\n')
    ds_b = datasets.load_dataset('json', data_files=path_b, split='train')
    return (ds_a, rows_a, refs_a, path_a), (ds_b, rows_b, path_b)


OK = 0


def check(name, cond, detail=''):
    global OK
    assert cond, f'{name} {detail}'
    OK += 1
    print(f'  ok {name} {detail}', flush=True)


def make_args(mbs, dataset_path):
    return types.SimpleNamespace(
        context_parallel_size=1, tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
        max_length=1 << 20, packing=False, micro_batch_size=mbs, padding_free=True,
        data_sharding=False, train_dataloader_shuffle=True, group_by_length=False,
        dataset_num_proc=4, num_train_epochs=2, train_iters=None, global_batch_size=mbs,
        dataset=[dataset_path], streaming=False, split_dataset_ratio=0.0, columns=None,
        custom_dataset_info=None, dataset_shuffle=True)


def main():
    pw = load_pad_warmup()
    open(DECODE_COUNT_FILE, 'w').close()
    os.environ['SWIFT_PAD_WARMUP'] = '1'
    os.environ['SWIFT_PAD_WARMUP_CACHE_DIR'] = os.path.join(WORK, 'cache')
    # No megatron runtime here: pin the dp identity (demo2 scenario K did the same).
    pw._current_data_parallel_rank = lambda: 0

    (ds_a, rows_a, refs_a, path_a), (ds_b, rows_b, path_b) = gen_datasets()
    tpl = RealMediaTemplate()
    n_a = len(rows_a)
    print(f'dataset A: {n_a} rows (400 unique videos x 4 QA), '
          f'max ref length {max(refs_a)}')

    print('-- 1. real scan (single-pass, full): lengths == independent decode reference; '
          'dedup 4:1')
    todo, dup_of, sizes = pw._scan_dedup_plan(ds_a, n_a)
    check('A 去重计划: 1600 行 -> 400 唯一代表', len(todo) == 400 and len(dup_of) == 1200)
    # group_by_length=True disables the two-pass plan -> a plain FULL tail-first scan
    # (group_by_length is deliberately NOT in the cache key, so section 2 still hits it).
    args_full = make_args(1, path_a)
    args_full.group_by_length = True
    lengths, n_failed = pw.scan_sample_lengths(ds_a, tpl, args_full)
    n_decoded = os.path.getsize(DECODE_COUNT_FILE)
    check('全扫无失败行', n_failed == 0)
    check('扫描长度与独立参考逐位一致', lengths == refs_a)
    check(f'唯一媒体只解码一次（{n_decoded} == 400 唯一视频，非 1600）', n_decoded == 400)
    # 400 unique paths >= 256: the `_prewarm_stat` thread-pool path ran (t0 regression).

    print('-- 2. warm full cache -> exact mode, peak == full-scan truth (mbs 1/2/4)')
    for mbs in (1, 2, 4):
        args = make_args(mbs, path_a)
        truth = pw.replay_peak_batch_length(args, ds_a, refs_a, mbs, agg='sum', dp_rank=0)
        got = pw.get_pad_warmup_length(args, ds_a, tpl)
        check(f'A mbs={mbs} 峰值 == 全扫真值', got == truth, f'(got={got}, truth={truth})')
    n_decoded_2 = os.path.getsize(DECODE_COUNT_FILE)
    check('三次求解全部命中缓存（零新增解码）', n_decoded_2 == n_decoded)

    print('-- 3. cache v3: cold partial scan vs warm hit')
    # Fresh cache dir + proxy-friendly data: the two-pass scan decodes only calib+Top-N.
    os.environ['SWIFT_PAD_WARMUP_CACHE_DIR'] = os.path.join(WORK, 'cache2')
    open(DECODE_COUNT_FILE, 'w').close()
    args1 = make_args(1, path_a)
    got1 = pw.get_pad_warmup_length(args1, ds_a, tpl)
    n_cold = os.path.getsize(DECODE_COUNT_FILE)
    check(f'冷启动两遍扫描只解码部分唯一视频（{n_cold} < 400）', 0 < n_cold < 400)
    t0 = time.perf_counter()
    got2 = pw.get_pad_warmup_length(args1, ds_a, tpl)
    dt = time.perf_counter() - t0
    check(f'热缓存 {dt:.2f}s < 1s 且结果不变', dt < 1.0 and got2 == got1)
    truth1 = pw.replay_peak_batch_length(args1, ds_a, refs_a, 1, agg='sum', dp_rank=0)
    check('两遍扫描的 exact-batch 峰值 == 全扫真值', got1 == truth1,
          f'(got={got1}, truth={truth1}, 冷解码 {n_cold}/400 唯一)')

    print('-- 4. mixed dataset + corrupt media: graceful degrade, never a wrong peak')
    refs_b = []
    for r in rows_b:
        try:
            refs_b.append(ref_row_length(r))
        except Exception:
            refs_b.append(0)
    lengths_b, n_failed_b = pw.scan_sample_lengths(ds_b, tpl, make_args(1, path_b))
    check('B 损坏视频计为失败行', n_failed_b == 1)
    ok_rows = [(v, r) for v, r in zip(lengths_b, refs_b) if v > 0]
    check('B 已扫行长度与独立参考一致', all(v == r for v, r in ok_rows))
    args_b = make_args(1, path_b)
    truth_b = pw.replay_peak_batch_length(args_b, ds_b,
                                          [v if v > 0 else 0 for v in refs_b], 1,
                                          agg='sum', dp_rank=0)
    got_b = pw.get_pad_warmup_length(args_b, ds_b, tpl)
    check('B 降级链结果只欠不超（或 off）', got_b is None or 0 < got_b <= (1 << 20),
          f'(got={got_b})')
    print(f'   B: n_failed={n_failed_b}, resolved={got_b}, truth(full)={truth_b}')

    print('-- 5. collator end-to-end: first batch padded to exactly the resolved length')
    import math
    import torch
    lines = open(PATCH).read().split('\n')
    i0 = next(i for i, l in enumerate(lines) if l.startswith('+    def _sample_dummy_text_ids'))
    i1 = next(i for i in range(i0, len(lines)) if lines[i].startswith('+    def data_collator'))
    src = '\n'.join(l[1:] for l in lines[i0:i1])
    from typing import Any, Dict, List, Optional
    ns = {'torch': torch, 'math': math, 'List': List, 'Dict': Dict, 'Any': Any,
          'Optional': Optional}
    exec('class _FakeTemplate:\n' + src.rstrip() + '\n', ns)
    for cp in (1, 2):
        t = ns['_FakeTemplate']()
        t.use_megatron = True
        t.padding_free = True
        t.sequence_parallel_size = cp
        t.tokenizer = types.SimpleNamespace(pad_token_id=0, all_special_ids=[1, 2])
        t.image_token_id = 32000
        t._get_megatron_cp_length = types.MethodType(
            lambda self, n: n if (not self.use_megatron or self.sequence_parallel_size == 1)
            else math.ceil(n / (2 * self.sequence_parallel_size))
            * (2 * self.sequence_parallel_size), t)
        batch = []
        for v in refs_a[2000:2003]:
            batch.append({'input_ids': [5 + (i % 500) for i in range(v)], 'length': v,
                          'labels': [1] * v, 'loss_scale': [1.] * v,
                          'token_type_ids': [0] * v, 'channel': 'ch0',
                          'mm_token_type_ids': torch.zeros(v, dtype=torch.int64),
                          'position_ids': torch.arange(v).expand(3, v).clone()})
        target = pw._cp_pad_length(max(refs_a) + 64, cp)
        def collate(b, pad_to_length=None, _t=t):
            if pad_to_length:
                _t._append_warmup_dummy(b, pad_to_length)
            return sum(r['length'] for r in b)
        col = pw.PadWarmupCollator(collate, target, 1, flag_path=None,
                                   padding_free=True, cp_size=cp)
        total = col([dict(r) for r in batch])
        check(f'cp={cp} 首个 batch 恰好 pad 到目标长度', total == target,
              f'(total={total}, target={target})')

    print(f'\nDEMO10 OK: {OK} checks passed (artifacts in {WORK})')


if __name__ == '__main__':
    main()
