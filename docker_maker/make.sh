#!/usr/bin/env bash
# ==============================================================================
# P800 镜像制作工作流
#
# 一轮修改流程:
#   ./make.sh load      # wget(若缺tar) + docker load, 自动解析并记录基础镜像名
#   ./make.sh run       # 起开发容器并直接进入(已存在则直接进); patch 挂载在 /workdir
#   ./make.sh commit    # 固化为 patched base 镜像(原始tag追加 -pN), 无改动则跳过
#   ./make.sh build     # docker build: patched base + dockerfile.p800 -> 最终在线镜像
#   ./make.sh clean     # 删除开发容器, 本轮结束
#   ./make.sh auto [命令] # 一键式: load->run->容器内执行命令(默认打patch)->commit->build->clean
#
# 下一轮修改: 重新 ./make.sh run (自动基于上一版 -pN 镜像), apply 新patch即可
# ==============================================================================
set -eo pipefail

# export http_proxy="http://star-proxy.oa.com:3128"
# export https_proxy="http://star-proxy.oa.com:3128"
# export ftp_proxy="http://star-proxy.oa.com:3128"

# ---------------- 配置区: 换基础镜像版本时改这里 ----------------
TAR_URL="https://klx-sdk-release-public.su.bcebos.com/xpytorch/docker/xmegatron/20260916/xmegatron_ubuntu2004_py312_torch290_release_3.7.1.0_19.tar"
TAR_URL="https://klx-sdk-release-public.su.bcebos.com/xpytorch/docker/xmegatron/20260922/xmegatron_ubuntu2004_py312_torch290_release_3.7.1.0_22.tar"
TAR_URL="https://klx-sdk-release-public.su.bcebos.com/xpytorch/docker/xmegatron/20260929/xmegatron_ubuntu2004_py312_torch290_release_3.7.1.0_26.tar"
TAR_URL="https://klx-sdk-release-public.su.bcebos.com/xpytorch/docker/xmegatron/20261008/xmegatron_ubuntu2004_py312_torch290_release_3.7.1.0_22_1008.tar"

# 最终在线镜像名(commit 过 patch 时自动追加 -pN 后缀)
FINAL_IMAGE="qwen-p800-megatron:20261009"

# 远端仓库: build 完成后自动 tag + push(需提前 docker login)
PUSH_REPO="mirrors.tencent.com/victor_hub/sft_docker"

# auto 一键式: 容器内 conda 环境名(非交互执行前需 source 激活, 否则会用到系统 python)
CONDA_ENV="python312_torch29_cuda"
# auto 一键式: 默认的容器内命令(可用 ./make.sh auto "<命令>" 覆盖)
AUTO_CMD_DEFAULT="bash /workdir/docker_patchs/patch_apply.sh"

# patch 与指导文档目录(脚本同级 docker_patchs), 容器内挂载到 /workdir
# 可用环境变量覆盖容器名(如 CONTAINER=tmp_update_2 ./make.sh auto) 以并行起多个开发容器
CONTAINER="${CONTAINER:-tmp_update}"
AUTHOR="vector"
# ----------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TAR_FILE="$(basename "${TAR_URL}")"
BASE_FILE=".base_image"      # docker load 后自动解析的原始镜像名
URL_FILE=".tar_url"          # 上次 load 用的 TAR_URL(检测版本变更)
BASELINE_FILE=".container_baseline"  # 容器启动即存在的 diff 条目, commit 时扣除
DIFF_DIR="diff_logs"         # 每轮 commit 的改动清单留档

# 恒定噪音: 挂载点 / 平台 agent 目录 / 缓存 / 日志 / 历史 等, 非人为改动
NOISE='__pycache__|\.pyc$|/tmp/|/var/log/|\.bash_history|/pkgs/cache/|\.cache/|/usr/local/sa|/usr/local/tmanager|^. /workdir($|/)|^. /home($|/)'

msg() { echo ">>> $*"; }
die() { echo "错误: $*" >&2; exit 1; }

# 原始(load 出来的)镜像名
orig_image() {
    [[ -f "${BASE_FILE}" ]] || die "尚未 load 基础镜像, 先执行: ./make.sh load"
    cat "${BASE_FILE}"
}

# 已存在的 patched tag 列表 (原始tag-pN, 按版本排序)
patch_tags() {
    local img repo tag
    img=$(orig_image); repo="${img%%:*}"; tag="${img##*:}"
    docker images "${repo}" --format '{{.Tag}}' | grep -E "^${tag}-p[0-9]+$" | sort -V || true
}

# run/build 的基底: 最新 patched 镜像, 没有则为原始镜像
latest_base() {
    local img latest
    img=$(orig_image)
    latest=$(patch_tags | tail -n 1)
    if [[ -n "${latest}" ]]; then
        echo "${img%:*}:${latest}"
    else
        echo "${img}"
    fi
}

require_container() {
    docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER}" \
        || die "容器 ${CONTAINER} 不存在, 先执行: ./make.sh run"
}

cmd_load() {
    local last_url=""
    [[ -f "${URL_FILE}" ]] && last_url=$(cat "${URL_FILE}")

    # 只有"TAR_URL 未变 且 镜像确实在本地"才跳过; URL 变了必须重新下载+load
    if [[ "${last_url}" == "${TAR_URL}" ]] && [[ -f "${BASE_FILE}" ]] \
        && docker image inspect "$(cat "${BASE_FILE}")" >/dev/null 2>&1; then
        msg "基础镜像已 loaded(TAR_URL 未变): $(cat "${BASE_FILE}")"
        return
    fi

    [[ "${last_url}" != "${TAR_URL}" ]] && msg "检测到 TAR_URL 变更: ${last_url:-<首次>} -> ${TAR_URL}"

    if [[ ! -f "${TAR_FILE}" ]]; then
        msg "下载基础镜像: ${TAR_URL}"
        wget -O "${TAR_FILE}" "${TAR_URL}"
    fi
    msg "docker load ${TAR_FILE}"
    local loaded
    loaded=$(docker load -i "${TAR_FILE}" | sed -n 's/^Loaded image: //p' | head -n 1)
    [[ -n "${loaded}" ]] || die "无法从 docker load 输出解析镜像名"
    echo "${loaded}" > "${BASE_FILE}"
    echo "${TAR_URL}" > "${URL_FILE}"
    msg "基础镜像: ${loaded} load 成功，并记录到 ${BASE_FILE}"
}

cmd_run() {
    # 默认进入交互 shell; 传 --no-enter 则只起容器+抓 baseline, 不进入(供 auto 非交互调用)
    local enter=1
    [[ "${1:-}" == "--no-enter" ]] && enter=0
    if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER}"; then
        msg "进入已有容器 ${CONTAINER}"
        msg "如需丢弃未 commit 的改动重开: ./make.sh clean && ./make.sh run"
    else
        local base
        base=$(latest_base)

        msg "基于 ${base} 启动开发容器 ${CONTAINER}"
        docker run -dti --name "${CONTAINER}" --privileged \
            --ulimit core=-1 \
            --net=host --uts=host --ipc=host \
            --security-opt=seccomp=unconfined \
            --shm-size=256g \
            --restart=always \
            -v ~:/home -v `pwd`:/workdir \
            -w /workdir/docker_patchs \
            "${base}"
        # 记录容器启动即产生的条目(挂载点/平台agent等), 供 commit 时扣除
        docker diff "${CONTAINER}" > "${BASELINE_FILE}"
    fi
    docker start "${CONTAINER}" >/dev/null 2>&1 || true
    if [[ "${enter}" == 1 ]]; then
        docker exec -it "${CONTAINER}" /bin/bash
    fi
}

cmd_commit() {
    require_container
    local img dst ver n=1 cur real
    img=$(orig_image)
    ver=$(patch_tags | tail -n 1)
    [[ -n "${ver}" ]] && n=$(( ${ver##*-p} + 1 ))
    dst="${img%:*}:${img##*:}-p${n}"
    ver="${img##*:}-p${n}"

    # 1) 先剔除仅因子项变化被连带标记 C 的父目录
    #    (必须在过滤噪音之前: 否则 /usr/local/sa 被滤掉后, C /usr/local 失去子项而无法剔除)
    cur=$(docker diff "${CONTAINER}" | grep -E '^[ACD] ' | awk '
        { flag[NR] = $1; p[NR] = $2 }
        END {
            for (i = 1; i <= NR; i++) {
                drop = 0
                for (j = 1; j <= NR; j++)
                    if (i != j && index(p[j], p[i] "/") == 1) { drop = 1; break }
                if (!drop) print flag[i] " " p[i]
            }
        }')

    # 2) 再过滤恒定噪音: 挂载点 / 平台 agent / 缓存 / 日志 / bash_history
    cur=$(printf '%s\n' "${cur}" | grep -vE "${NOISE}" || true)

    # 3) 扣除容器启动基线(容器一启动就存在的条目, 非人为改动)
    if [[ -f "${BASELINE_FILE}" ]]; then
        real=$(comm -23 \
            <(printf '%s\n' "${cur}" | awk 'NF {print $2}' | sort -u) \
            <(awk 'NF {print $2}' "${BASELINE_FILE}" | sort -u) || true)
    else
        msg "提示: 无启动基线 ${BASELINE_FILE}(容器由旧版脚本创建), 仅按噪音规则判断"
        real=$(printf '%s\n' "${cur}" | awk 'NF {print $2}' | sort -u)
    fi

    if [[ -z "${real}" ]]; then
        msg "容器 ${CONTAINER} 未检测到有效改动, 跳过本次 commit(不生成新版本号和改动清单)"
        msg "已排除: 挂载点(/workdir,/home), 平台 agent(/usr/local/sa,/usr/local/tmanager), 缓存/日志/bash_history, 及被连带标记的父目录"
        msg "流程继续: 可直接 ./make.sh build, 将基于当前镜像 $(latest_base) 构建"
        return 0
    fi

    msg "检测到 $(printf '%s\n' "${real}" | wc -l | tr -d ' ') 处改动, commit ${CONTAINER} -> ${dst}"
    docker commit -a "${AUTHOR}" -m "apply patches: $(date +%F)" "${CONTAINER}" "${dst}"

    mkdir -p "${DIFF_DIR}"
    printf '%s\n' "${cur}" | grep -Ff <(printf '%s\n' "${real}" | sed 's|^| |') > "${DIFF_DIR}/${ver}.list" || true
    msg "改动清单已保存: ${DIFF_DIR}/${ver}.list"
    msg "下一步: ./make.sh build -> ./make.sh clean"
}

cmd_build() {
    local base btag suffix final
    base=$(latest_base)
    btag="${base##*:}"
    suffix=""
    [[ "${btag}" == *-p[0-9]* ]] && suffix="-${btag##*-}"
    final="${FINAL_IMAGE}${suffix}"

    msg "docker build: BASE_IMAGE=${base} -> ${final}"
    docker build -f dockerfile.p800 --build-arg BASE_IMAGE="${base}" -t "${final}" .
    msg "Build image Done: ${final}"

    # 以构建产物的 image id 打远端 tag 并推送(远端 tag = 本地镜像名冒号改短横线)
    local push_tag img_id
    push_tag="${final//:/-}"        # qwen-p800-megatron-20260916 或 qwen-p800-megatron-20260916-p1
    img_id=$(docker inspect -f '{{.Id}}' "${final}") \
        || die "取不到镜像 ${final} 的 image id, 推送中止"
    msg "tag: ${img_id:7:19} -> ${PUSH_REPO}:${push_tag}"
    docker tag "${img_id}" "${PUSH_REPO}:${push_tag}"
    docker push "${PUSH_REPO}:${push_tag}"
    msg "Done: 已推送 ${PUSH_REPO}:${push_tag}"
}

cmd_clean() {
    if docker rm -f "${CONTAINER}" 2>/dev/null; then
        rm -f "${BASELINE_FILE}"
        msg "已删除容器 ${CONTAINER}, 本轮结束"
    else
        msg "容器 ${CONTAINER} 不存在, 跳过"
    fi
}

# 一键式: load -> run(非交互) -> 容器内执行命令 -> commit -> build -> clean
# 用法: ./make.sh auto [命令]   (命令缺省用 AUTO_CMD_DEFAULT)
cmd_auto() {
    local cmd="$*"
    [[ -z "${cmd}" ]] && cmd="${AUTO_CMD_DEFAULT}"
    msg "一键式开始: load -> run -> 容器内执行 -> commit -> build -> clean"
    msg "容器内命令: ${cmd}"

    cmd_load
    cmd_run --no-enter

    # 非交互 shell 不会 source ~/.bashrc, 必须显式激活 conda 环境(否则用到系统 python)
    msg "容器内执行(激活 ${CONDA_ENV})..."
    docker exec "${CONTAINER}" bash -c \
        "source /root/miniconda/bin/activate ${CONDA_ENV} && ${cmd}" \
        || die "容器内命令执行失败, 已保留容器 ${CONTAINER} 供排查(未 commit/build/clean)"

    cmd_commit
    cmd_build
    cmd_clean   # 只有前面全部成功才走到这; 失败时 set -e 已中断, 容器保留
    msg "一键式完成"
}

usage() { sed -n '2,17p' "$0" | sed 's/^#\s\{0,1\}//'; }

case "${1:-}" in
    load)   cmd_load ;;
    run)    cmd_run ;;
    commit) cmd_commit ;;
    build)  cmd_build ;;
    clean)  cmd_clean ;;
    auto)   shift; cmd_auto "$@" ;;
    *)      usage; exit 1 ;;
esac
