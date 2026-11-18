# Pad-Warmup v7 设计：settle-then-carve（先落地常驻分配，再切峰值块）

> **状态：已落地（2026-11-11）。** 本文是 v7 的设计与否决依据存档；落地结果已合并回
> 主文档 `feature4_megatron_pad_warmup.md`（§2.2、§4.4、§9.8、§9.9）。实现阶段增补了
> 一项设计外内容：MoE dispatcher 路由状态释放（上游 Megatron-LM #8297/#8231 的 backport，
> 见主文档 §9.9）。v6 的内容见
> `feature4_megatron_pad_warmup-6.md`（patch 封存为 `feature4_megatron_pad_warmup-6.patch`）。
> **环境验证（§6 探针与 A/B）尚未执行，是下一步。**
>
> **源码依据**：镜像 `qwen-p800-megatron:20261009-p1`（megatron-core 0.18.0 +
> ms-swift 4.3.1），下文所有"已确认"结论均对照该镜像内源码核实。

## 1. 要解决什么问题

v6 在 step0 把第一个 batch pad 到本卡峰值长度，allocator 按峰值轨迹切好大块。但 step0 时
**一批重要的常驻分配还不存在**，它们在第一二步陆续落地，落点由当时的 free block 布局
决定——落进预热切好的大块里，就把大块永久钉碎。表现为：**预热出来的大内存块被小块
占用，后续无法正常整块复用**，碎片 OOM 的机理在 run 中后期重新出现。

常驻惰性分配清单（按量级排序）：

| 分配 | 落地时机 | 量级（本配置） |
|---|---|---|
| Adam `exp_avg/exp_avg_sq` | **首个 `optimizer.step()`**（见 §2.1） | 2×fp32 参数 ÷ dp_size |
| 通信子 context（XCCL/BKCL，裸 malloc） | 各 PG 首次集合通信（§7，v6 已预热但仍在 carve 之前与否取决于顺序） | ~2.3 GiB |
| cudnn/TE workspace、autograd 结构、MoE alltoall 缓冲 | 首个 step 的前向/反向 | shape 相关且**瞬时**，不需 settle |

v6 的隐含假设"step0 时堆是空的"不成立。v7 把因果倒过来：**先让常驻分配各就各位
（settle），清场，再围着它们切峰值块（carve）**。

## 2. 源码确认的五个事实（设计的地基）

### 2.1 optimizer 状态在 fresh run 下是首个 `step()` 才分配的 ✓

megatron-core 的 Adam 带一个 `init_state_fn`（对 `opt.state[p]` 做 `zeros_like` 分配），
但它**只在 ckpt 加载路径被调用**（`MixedPrecisionOptimizer.state_dict/sharded_state_dict`
的 `is_loading=True` 分支）。训练路径没有任何地方调它——fresh run 的状态由内部
`torch.optim.AdamW` / TE `FusedAdam` 在首个 `step()` 里惰性建立。

**推论**：settle 确实有必要；且 `init_state_fn` 是现成接口，可以**不调 step 直接分配状态**
（见 §3.2）。

### 2.2 清场原语现成且有先例 ✓

`swift/utils/torch_utils.py::gc_collect()` = `gc.collect() + empty_cache()`（已做设备抽象），
`trainers/base.py::save_checkpoint()` 每次保存前都在调。清场不是新发明，是本仓库既有模式。

### 2.3 挂点清晰 ✓

`train()` → `setup_training()` → `setup_model_training()`（此时 `self.optimizer` 已建好，
`base.py:595` 已在用）→ `_prepare_dataloader()` → `wrap_pad_warmup()`（`base.py:944`）。
settle 的三个动作全部可以收进 `wrap_pad_warmup`，只需 base.py 多传一个 `optimizer` 参数。

### 2.4 真实 settle step 的状态污染（否决依据，§4.1）✓

跑一个真实/合成 step 会引入一堆状态副作用：RNG 偏移（dropout）、Adam 动量被垃圾梯度污染
（`exp_avg` 是 EMA，lr=0 只挡权重更新、挡不住状态吸收梯度）、`grad_norm_skip_threshold`
可能跳过 step 导致状态根本没分配、first-step 前钩子状态机
（`should_disable_forward_pre_hook` / `_start_iteration`）的交互。

### 2.5 单块预留不等于峰值轨迹（否决依据，§4.3）✓

预热起作用的机制是"峰值步的**完整分配轨迹**（几十种尺寸的张量按序分配/释放）在空堆上
先跑一遍"，不是一个最大块。`torch.empty(peak_bytes)` 只开一段，后续小 step 会把它按
小形状切碎，回到原始碎片 OOM。

## 3. v7 方案

### 3.1 一句话

`wrap_pad_warmup` 里，在通信子预热之后追加两个本地动作：**直接调 `init_state_fn` 把
optimizer 状态分配出来**，然后 **`gc_collect()` 清场**；之后训练循环的第一个 batch 仍由
现有 collator pad 到本卡峰值——峰值轨迹在"常驻分配已落地、堆已清干净"的状态上雕刻。

### 3.2 时序

```
setup_model_training() 完成（模型/DDP buffer/optimizer main params 已就位）
  → _prepare_dataloader → wrap_pad_warmup：
      1. _agree_enable 协商（v6 现有）
      2. 长度扫描 + replay（v6 现有；分钟级，GPU 空闲）
      3. warmup_communicators()（v6 现有 → 通信子 context 落地）
      4. 【新】init optimizer states：遍历 ChainedOptimizer 的每个子 optimizer
         （dense + expert 两个 DistributedOptimizer），调
         opt.init_state_fn(opt.optimizer, opt.config)；函数自带
         `if len(state[p]) == 0` 守卫，ckpt resume 场景幂等
      5. 【新】gc_collect() + allocator 摘要日志（before/after reserved/allocated，可审计）
  → 训练循环：step1 由 collator pad 到本卡峰值 → carve 发生在干净的堆上
```

### 3.3 为什么 settle 不需要跑任何 step

| 常驻分配 | 不跑 step 的落地手段 |
|---|---|
| optimizer 状态（最大头） | `init_state_fn` 直接调，纯 `zeros_like`，不碰梯度数学 |
| 通信子 context | 现有 `warmup_communicators()`（1 元素 all_reduce 即触发建 context） |
| grad buffer | DDP 构造时已分配，非惰性 |
| workspace / autograd / alltoall 缓冲 | 瞬时分配，不属于 settle 对象 |

### 3.4 分布式安全性

`init_state_fn` 与 `gc_collect` 都是**纯本地操作，不新增任何集合通信**。单卡失败
try/except 降级 = 该卡退回 v6 行为（无 settle），不影响其他卡、无挂死面。延续 v6 铁律：
**补丁的任何失败路径都不能打挂训练**。

### 3.5 改动面

| 文件 | 改动 |
|---|---|
| `swift/megatron/trainers/base.py` | 1 行：`wrap_pad_warmup(..., optimizer=self.optimizer)` |
| `swift/megatron/trainers/pad_warmup.py` | 新增 `_settle_allocator_state(optimizer)`（~30 行）+ 调用点 + 摘要日志 |
| `swift/template/base.py` | 不动 |

无新环境变量（默认随 `SWIFT_PAD_WARMUP=1` 开启），无新集合通信，collator/dummy 段/
扫描/重放/闭环校验全部不变。

## 4. 被否决的备选方案（及否决依据）

### 4.1 真实/合成 settle step（"先跑一步再清场"）

机理上最完整（连 shape 相关 workspace 都落地），但状态副作用处理链太长：lr=0 挡权重
（AdamW 的 wd 项乘 lr，成立）但挡不住 **Adam 动量吸收垃圾梯度**（`exp_avg` 是 EMA，
需事后清零 state 和 step 计数）；dropout 消耗 RNG 需快照/恢复（torch + Megatron TP
rng tracker 两处）；合成小 batch 要复制 template 的 batch 构造（`packed_seq_params` /
`prepare_batch` 设备搬运），脆弱；垃圾梯度可能触发 `grad_norm_skip_threshold` 跳过 step
导致状态没分配。收益（workspace 落地）与风险不成比例——**workspace 是瞬时分配，
本来就不需要 settle**。

### 4.2 "真实 batch + 迭代器重建"的 settle

比合成 batch 稳（全走现有代码路径），但仍要带 RNG/动量处理，且 dataloader 建两遍。
作为**探针发现漏网大额常驻分配时的后备方案**保留（见 §6）。

### 4.3 单块 `torch.empty(peak_bytes)` 预留

不复制峰值轨迹，小块会把大段切碎（§2.5）。否。

## 5. 不解决什么（残余风险，与 v6 相同）

| 残余污染源 | 说明 | 后续手段 |
|---|---|---|
| alltoall 缓冲随路由分布增长（-6 文档 §8.2） | 20ep 实测每 rank +95~156 MiB，发生在 carve 之后 | 接受；或通信缓冲独立池 |
| eval 首步 / ckpt 的偶发分配 | 同上 | 观察到问题再处理 |
| 跨卡耦合（-6 文档 §4.3） | 与 carve 时机无关 | 不在本方案范围 |

可选的后续阶段（先不做）：`_check_overshoot` WARN 触发"清场 + 按观测长度重 pad"的
运行时自愈。v7 先把 settle-then-carve 做扎实。

## 6. 验证计划（开发时按序执行）

1. **探针**（一次性，确认 §3.3 的推断）：vanilla v6 下对比 setup 后与 step1 后的
   `memory_reserved`，差值应 ≈ `2×fp32参数/dp + 2.3 GiB`。若有明显超出的大项，
   settle 升级为 §4.2 的真实 step 方案。
2. **单元验证**：settle 后 `len(optimizer.optimizer.state) > 0`（状态确实分配）；
   resume-from-ckpt 场景 `init_state_fn` 幂等（不覆盖已加载状态）；
   `gc_collect` 后 `reserved - allocated ≈ 常驻下界`。
3. **A/B**：v6 vs v7，同 8 卡 P800 配置，3ep 与 20ep 两档；每 N 步记录
   `max_memory_reserved` + 最大空闲块尺寸（snapshot 手法，-6 文档附录 A），
   直接观察 step1 之后预热块是否仍被切碎。
4. **达标线**：v7 的 20ep 峰值水位 ≤ v6；全 run 0 OOM、0 overshoot WARN；
   settle 额外耗时 < 5s（状态分配是 memset 级）。

## 7. 开发清单（启动时按此执行）

1. `pad_warmup.py`：`_settle_allocator_state(optimizer)`（ChainedOptimizer 遍历 +
   守卫 + try/except 降级 + 日志）；`wrap_pad_warmup` 签名加 `optimizer=None`（向后兼容），
   在 `warmup_communicators()` 后调用 settle；
2. `base.py`：调用点传 `self.optimizer`；
3. demo：`demo_settle.py`——模拟 ChainedOptimizer/DistributedOptimizer 结构，
   验证状态分配、幂等、失败降级；
4. 探针（§6.1）先跑，结果回填本文 §6.1；
5. A/B（§6.3-6.4），数据回填主文档；
6. 合并：主文档 `feature4_megatron_pad_warmup.md` 更新为 v7（§2.2 流程、§4.4 版本表
   加 v7 行、新增 settle 章节），patch 重建封存 -6。
