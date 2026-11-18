#!/usr/bin/env bash
# 在容器内给已安装的 swift 包打补丁(改动落在 site-packages, 可被 docker commit 固化)
#
# 用法(进入容器后 conda 环境已默认激活):
#   bash /workdir/docker_patchs/patch_apply.sh [补丁文件路径]
#
# 行为:
#   - 补丁缺失 / 预检不通过(冲突或已应用过) / 应用失败 / 校验失败 -> 打印错误并 exit 1
#   - 冲突产物(.rej)写入 /tmp, 不污染 site-packages(避免被 commit 进镜像)
set -euo pipefail

# export http_proxy="http://star-proxy.oa.com:3128"
# export https_proxy="http://star-proxy.oa.com:3128"
# export ftp_proxy="http://star-proxy.oa.com:3128"

log() { echo ">>> $*"; }
die() { echo "错误: $*" >&2; exit 1; }

# ---------- 1) 依赖包补全 ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/deps_install.sh"

PATCH_FILE="${1:-${SCRIPT_DIR}/megatron/feature4_megatron_pad_warmup.patch}"
[[ -f "${PATCH_FILE}" ]] || die "找不到补丁文件: ${PATCH_FILE}"
PATCH_FILE="$(cd "$(dirname "${PATCH_FILE}")" && pwd)/$(basename "${PATCH_FILE}")"

# ---------- 2) 用 pip show 定位 site-packages ----------
REJ="/tmp/$(basename "${PATCH_FILE}").rej"
rm -f "${REJ}"

SITE_PKGS=$(pip show ms_swift 2>/dev/null | sed -n 's/^Location: //p' | tail -n 1)
[[ -n "${SITE_PKGS}" && -d "${SITE_PKGS}" ]] \
    || die "未能通过 pip show ms_swift 定位安装目录, 请确认当前 conda 环境已激活且已安装 ms_swift"
log "SITE_PKGS=${SITE_PKGS}"
cd "${SITE_PKGS}"

# ---------- 3) 预检: 冲突或已应用过都会失败, 一律报错退出, 不改动任何文件 ----------
log "预检补丁: $(basename "${PATCH_FILE}")"
if ! patch -p1 --dry-run --forward --no-backup-if-mismatch -r "${REJ}" < "${PATCH_FILE}"; then
    [[ -s "${REJ}" ]] && { echo "---- 冲突详情 ----" >&2; cat "${REJ}" >&2; }
    die "补丁无法干净应用(上下文不匹配 / 已应用过 / 已部分应用), 未改动任何文件: ${PATCH_FILE}"
fi

# ---------- 4) 正式应用(失败由 set -e 中断, patch 自己会打印原因) ----------
log "应用补丁"
patch -p1 --forward --no-backup-if-mismatch -r "${REJ}" < "${PATCH_FILE}"

# ---------- 5) 校验结果 ----------
log "校验补丁结果"
python -m py_compile \
    swift/megatron/trainers/pad_warmup.py \
    swift/megatron/trainers/base.py \
    swift/template/base.py \
    || die "语法校验失败, 补丁可能打歪了, 请检查 ${SITE_PKGS} 下的对应文件"

log "OK: 补丁已生效"
