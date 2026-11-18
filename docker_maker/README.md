# make.sh — P800 镜像制作工作流

基于基础镜像 tar，经"开发容器内打补丁 → commit 固化 → docker build"产出最终训练镜像，并推送远端仓库。

## 一轮修改的标准流程

```bash
./make.sh load      # 下载(若缺 tar)+docker load，记录基础镜像名到 .base_image
./make.sh run       # 起开发容器并进入（已存在则直接进）；本目录挂载到容器 /workdir
# 容器内: 打补丁，默认路径下直接 bash patch_apply.sh
./make.sh commit    # 固化为 patched 基础镜像（原始 tag 追加 -pN，自动递增；无改动则跳过）
./make.sh build     # docker build：patched base + dockerfile.p800 → 最终镜像，并 tag+push 远端
./make.sh clean     # 删除开发容器，本轮结束
```

下一轮修改：重新 `./make.sh run`（自动基于上一版 `-pN` 镜像），再打新补丁即可——`-pN` 是逐轮累积的补丁层。

## 命令一览

| 命令 | 作用 |
|---|---|
| `load` | 下载并 docker load 基础镜像；TAR_URL 变更时自动重新下载 |
| `run` | 基于最新 patched 镜像（无则原始镜像）起/进入开发容器 `tmp_update` |
| `commit` | 检测容器内真实改动（滤掉挂载点/缓存/日志等噪音）→ `docker commit` 为 `-pN`；改动清单存 `diff_logs/` |
| `build` | `docker build --build-arg BASE_IMAGE=<最新patched>` → `qwen-p800-megatron:20261009[-pN]`，推送 `mirrors.tencent.com/victor_hub/sft_docker` |
| `clean` | 删除开发容器（不改动的镜像 tag 不动） |
| `auto [命令]` | 一键式：load → run(不进入) → 容器内执行命令（默认打 patch）→ commit → build → clean |

## 配置区（make.sh 头部）

- `TAR_URL`：基础镜像 tar 地址（换底座版本时改这里）
- `FINAL_IMAGE`：最终镜像名（commit 过时自动追加 `-pN`）
- `PUSH_REPO`：远端推送仓库（需先 `docker login`）
- `CONDA_ENV` / `AUTO_CMD_DEFAULT`：auto 模式容器内的 conda 环境名与默认命令
- `CONTAINER`：开发容器名，可用环境变量覆盖以并行起多个（`CONTAINER=tmp_update_2 ./make.sh auto`）

## 状态文件（自动维护，勿手改）

| 文件 | 内容 |
|---|---|
| `.base_image` | docker load 解析出的原始镜像名 |
| `.tar_url` | 上次 load 用的 TAR_URL（检测版本变更） |
| `.container_baseline` | 容器启动基线 diff，commit 时扣除噪音 |
| `diff_logs/<tag>.list` | 每轮 commit 的改动清单留档 |

## 注意事项

- commit 只固化**容器内对镜像的改动**（site-packages 等）；`/workdir`（本目录）和 `/home` 是挂载点，不会被提交。
- 补丁应用脚本是 `docker_patchs/patch_apply.sh`：预检（冲突/已应用）不通过会拒绝并退出，不污染镜像。
- 无有效改动时 commit 自动跳过，不生成空版本号。
