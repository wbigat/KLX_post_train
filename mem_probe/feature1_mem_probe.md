# [STASHED] 特性一：Per-rank CUDA 显存历史采集

该特性已从 `src/llamafactory/train/tuner.py` 中移出，暂存于此。本文件包含全部代码与恢复步骤，
按下面 3 步粘回 `tuner.py` 即可完全还原（不依赖 git）。

## 用法（恢复后）

```bash
LLAMAFACTORY_MEM_PROBE=1 llamafactory-cli train RENDERED_YAML.yaml
```

可选环境变量：

| 环境变量 | 默认 | 说明 |
| --- | --- | --- |
| `LLAMAFACTORY_MEM_PROBE_DIR` | `mem_probe` | 输出目录 |
| `LLAMAFACTORY_MEM_PROBE_MAX_ENTRIES` | `150000` | 每卡记录的分配事件上限 |
| `LLAMAFACTORY_MEM_PROBE_CONTEXT` | `all` | `all` / `alloc` / `state` |
| `LLAMAFACTORY_MEM_PROBE_STACKS` | `all` | `all` / `python` |
| `LLAMAFACTORY_MEM_PROBE_INTERVAL` | `2` | timeline 采样间隔（秒），`0` 关闭 |
| `LLAMAFACTORY_MEM_PROBE_DUMP_ON_EXIT` | `0` | 正常结束也 dump 一份基线 |

每卡产物：`{tag}_{reason}.pickle`（拖到 https://docs.pytorch.org/memory_viz）、
`{tag}_{reason}_summary.txt`、`{tag}_{reason}_stats.json`、`{tag}_timeline.jsonl`。

## 恢复步骤 1/3：`tuner.py` 顶部 import

把
```python
import os
import shutil
from typing import TYPE_CHECKING, Any, Optional
```
替换为
```python
import json
import os
import shutil
import signal
import socket
import threading
import time
from typing import TYPE_CHECKING, Any, Optional
```

## 恢复步骤 2/3：在 `logger = logging.get_logger(__name__)` 之后插入

```python
# ==================================================================================================
# [TEMPORARY] Per-rank CUDA memory probe for OOM diagnosis.
#
# Usage (single node, 8 GPUs):
#   LLAMAFACTORY_MEM_PROBE=1 llamafactory-cli train RENDERED_YAML.yaml
#
# Optional env vars:
#   LLAMAFACTORY_MEM_PROBE_DIR          output dir, default "./mem_probe"
#   LLAMAFACTORY_MEM_PROBE_MAX_ENTRIES  max recorded alloc events per rank, default 150000
#   LLAMAFACTORY_MEM_PROBE_CONTEXT      "all" | "alloc" | "state", default "all"
#   LLAMAFACTORY_MEM_PROBE_STACKS       "all" | "python", default "all"
#   LLAMAFACTORY_MEM_PROBE_INTERVAL     memory_stats sampling interval (s), 0 disables, default 2
#   LLAMAFACTORY_MEM_PROBE_DUMP_ON_EXIT dump snapshot on normal finish too, default 0
#
# Outputs (one set of files per rank):
#   {tag}_{reason}.pickle       -> load at https://docs.pytorch.org/memory_viz
#   {tag}_{reason}_summary.txt  -> torch.cuda.memory_summary()
#   {tag}_{reason}_stats.json   -> torch.cuda.memory_stats()
#   {tag}_timeline.jsonl        -> sampled allocated/reserved/num_ooms timeline
# ==================================================================================================
_MEM_PROBE: dict[str, Any] = {"on": False, "dumped": False, "dir": "", "tag": "", "fp": None, "stop": None}


def _mem_probe_enabled() -> bool:
    return os.getenv("LLAMAFACTORY_MEM_PROBE", "0").lower() in ("1", "true", "yes", "y")


def _mem_probe_sampler(interval: float) -> None:
    stop_event: threading.Event = _MEM_PROBE["stop"]
    while not stop_event.wait(interval):
        try:
            if not torch.cuda.is_available() or not torch.cuda.is_initialized():
                continue

            stats = torch.cuda.memory_stats()
            record = {
                "ts": round(time.time(), 3),
                "device": torch.cuda.current_device(),
                "alloc_MB": round(stats.get("allocated_bytes.all.current", 0) / 2**20, 2),
                "alloc_peak_MB": round(stats.get("allocated_bytes.all.peak", 0) / 2**20, 2),
                "reserved_MB": round(stats.get("reserved_bytes.all.current", 0) / 2**20, 2),
                "reserved_peak_MB": round(stats.get("reserved_bytes.all.peak", 0) / 2**20, 2),
                "active_MB": round(stats.get("active_bytes.all.current", 0) / 2**20, 2),
                "inactive_split_MB": round(stats.get("inactive_split_bytes.all.current", 0) / 2**20, 2),
                "num_alloc_retries": stats.get("num_alloc_retries", 0),
                "num_ooms": stats.get("num_ooms", 0),
            }
            _MEM_PROBE["fp"].write(json.dumps(record) + "\n")
            _MEM_PROBE["fp"].flush()
        except Exception:  # never break training because of the probe
            continue


def _mem_probe_signal_handler(signum: int, frame: Any) -> None:
    # torchrun kills the sibling ranks once one rank OOMs, dump before dying so every card is captured
    _mem_probe_dump(f"signal{signum}")
    _mem_probe_stop()
    os._exit(128 + signum)


def _mem_probe_start() -> None:
    """Start recording CUDA memory history. Must be called before model loading."""
    if _MEM_PROBE["on"] or not _mem_probe_enabled():
        return

    if not torch.cuda.is_available():
        logger.warning("[mem_probe] CUDA is unavailable, memory probe disabled.")
        return

    rank = int(os.getenv("RANK", os.getenv("LOCAL_RANK", "0")))
    output_dir = os.getenv("LLAMAFACTORY_MEM_PROBE_DIR", "mem_probe")
    os.makedirs(output_dir, exist_ok=True)
    _MEM_PROBE["dir"] = output_dir
    _MEM_PROBE["tag"] = f"{socket.gethostname()}_rank{rank:02d}_local{os.getenv('LOCAL_RANK', '0')}_pid{os.getpid()}"

    try:
        torch.cuda.memory._record_memory_history(
            enabled="all",
            context=os.getenv("LLAMAFACTORY_MEM_PROBE_CONTEXT", "all"),
            stacks=os.getenv("LLAMAFACTORY_MEM_PROBE_STACKS", "all"),
            max_entries=int(os.getenv("LLAMAFACTORY_MEM_PROBE_MAX_ENTRIES", "150000")),
        )
    except Exception as e:
        logger.warning(f"[mem_probe] Failed to enable memory history: {e}.")
        return

    _MEM_PROBE["on"] = True
    _MEM_PROBE["stop"] = threading.Event()
    interval = float(os.getenv("LLAMAFACTORY_MEM_PROBE_INTERVAL", "2"))
    if interval > 0:
        _MEM_PROBE["fp"] = open(os.path.join(output_dir, f"{_MEM_PROBE['tag']}_timeline.jsonl"), "a")
        threading.Thread(target=_mem_probe_sampler, args=(interval,), daemon=True).start()

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1, signal.SIGABRT):
        try:
            signal.signal(sig, _mem_probe_signal_handler)
        except Exception:
            pass

    logger.info(f"[mem_probe] Recording CUDA memory history, output dir: {os.path.abspath(output_dir)}.")


def _mem_probe_dump(reason: str) -> None:
    """Dump the snapshot of this rank. Called on OOM / other exceptions / signals."""
    if not _MEM_PROBE["on"] or _MEM_PROBE["dumped"]:
        return

    _MEM_PROBE["dumped"] = True
    prefix = os.path.join(_MEM_PROBE["dir"], f"{_MEM_PROBE['tag']}_{reason}")
    # write the small files first: the snapshot can be tens of MB, and when one rank OOMs torchrun
    # kills the siblings (SIGTERM then SIGKILL), so a slow pickle dump may never finish
    try:
        with open(f"{prefix}_summary.txt", "w") as f:
            f.write(torch.cuda.memory_summary())

        with open(f"{prefix}_stats.json", "w") as f:
            json.dump(torch.cuda.memory_stats(), f, indent=2)
    except Exception as e:
        logger.warning(f"[mem_probe] Failed to dump memory summary: {e}.")

    try:
        torch.cuda.memory._dump_snapshot(f"{prefix}.pickle")
        logger.info(f"[mem_probe] Snapshot saved to {os.path.abspath(prefix)}.pickle.")
    except Exception as e:
        logger.warning(f"[mem_probe] Failed to dump snapshot: {e}.")


def _mem_probe_stop() -> None:
    if not _MEM_PROBE["on"]:
        return

    _MEM_PROBE["on"] = False
    try:
        if _MEM_PROBE["stop"] is not None:
            _MEM_PROBE["stop"].set()

        if _MEM_PROBE["fp"] is not None:
            _MEM_PROBE["fp"].close()
            _MEM_PROBE["fp"] = None

        torch.cuda.memory._record_memory_history(enabled=None)
    except Exception:
        pass


def _is_oom_error(err: BaseException) -> bool:
    oom_cls = getattr(torch, "OutOfMemoryError", getattr(torch.cuda, "OutOfMemoryError", None))
    if oom_cls is not None and isinstance(err, oom_cls):
        return True

    return "out of memory" in str(err).lower()
```

## 恢复步骤 3/3：把 `run_exp` 换成带采集的版本

原始（当前仓库中的形态）：
```python
def run_exp(args: Optional[dict[str, Any]] = None, callbacks: Optional[list["TrainerCallback"]] = None) -> None:
    args = read_args(args)
    if "-h" in args or "--help" in args:
        get_train_args(args)

    ray_args = get_ray_args(args)
    callbacks = callbacks or []
    if ray_args.use_ray:
        _ray_training_function(ray_args, config={"args": args, "callbacks": callbacks})
    else:
        _training_function(config={"args": args, "callbacks": callbacks})
```

带采集的版本：
```python
def run_exp(args: Optional[dict[str, Any]] = None, callbacks: Optional[list["TrainerCallback"]] = None) -> None:
    args = read_args(args)
    if "-h" in args or "--help" in args:
        get_train_args(args)

    _mem_probe_start()  # [TEMPORARY] start recording before tokenizer/model loading
    ray_args = get_ray_args(args)
    callbacks = callbacks or []
    try:
        if ray_args.use_ray:
            _ray_training_function(ray_args, config={"args": args, "callbacks": callbacks})
        else:
            _training_function(config={"args": args, "callbacks": callbacks})
    except BaseException as e:
        _mem_probe_dump("oom" if _is_oom_error(e) else f"exc_{type(e).__name__}")
        _mem_probe_stop()
        raise
    else:
        if os.getenv("LLAMAFACTORY_MEM_PROBE_DUMP_ON_EXIT", "0").lower() in ("1", "true", "yes", "y"):
            _mem_probe_dump("finish")

        _mem_probe_stop()
```

## 备注

- 真实的每卡入口是 `launcher.py.__main__` -> `run_exp()`（torchrun 每个子进程各跑一次），
  所以采集自然是每卡独立、且早于 tokenizer/model 加载。
- ray 模式的 worker 不经过 `run_exp`（走 `Worker._training_function`），未覆盖。
- 已知限制：一个 rank OOM 后 torchrun 会 SIGTERM 再 SIGKILL 其余 rank，
  pickle 较大时可能来不及写完，因此 dump 顺序是先小文件后 pickle。
