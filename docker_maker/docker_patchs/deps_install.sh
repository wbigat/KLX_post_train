#!/usr/bin/env bash
# 依赖包补全: 在打补丁前把需要的包装齐(容器内执行)
#
# 由 patch_apply.sh 在头部自动调用, 也可单独执行:
#   bash /workdir/docker_patchs/deps_install.sh
#
# 失败即中断(exit 1), 不做静默跳过
set -euo pipefail

log() { echo ">>> [deps] $*"; }
die() { echo "错误: [deps] $*" >&2; exit 1; }

# star-proxy 会间歇性 403 拦截 ubuntu/github 源, 而本机直连正常。
# apt/wget 失败时自动去代理(env -u)直连重试; 直连也失败才报错。
net_retry() {
    if "$@"; then return 0; fi
    log "代理链路失败, 切换直连重试: $*"
    env -u http_proxy -u https_proxy -u ftp_proxy -u HTTP_PROXY -u HTTPS_PROXY "$@"
}

# ---------------- 配置区: 按需填 ----------------
# pip 包(装进当前 conda 环境), 空格分隔; 留空则跳过
PIP_PACKAGES="swanlab[dashboard]==0.7.20"

# apt 包; 留空则跳过
APT_PACKAGES="apt-utils git ccache unzip openssh-server openssh-client
    cron vim tree curl wget gdb iputils-ping net-tools
    sudo pssh pciutils lrzsz gzip tzdata"

# 时区
TZ_SET="Asia/Shanghai"
# sshd 监听端口(留空则不改 sshd 配置)
SSH_PORT="36000"
# ------------------------------------------------


# ========== 1) apt-get install ==========
if [[ -n "${APT_PACKAGES}" ]]; then
    log "apt-get install: $(echo ${APT_PACKAGES} | wc -w) 个包"
    # 清旧索引再 update, 避免索引过期导致引用已被镜像下架的版本(404)
    rm -rf /var/lib/apt/lists/*
    # --fix-missing + Acquire::Retries: 瞬时断连/部分下载自动重试; 代理 403 则走直连
    net_retry env DEBIAN_FRONTEND=noninteractive apt-get update -o Acquire::Retries=3
    net_retry env DEBIAN_FRONTEND=noninteractive apt-get install -y --fix-missing -o Acquire::Retries=3 ${APT_PACKAGES}
    apt-get clean
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
fi


# ========== 2) pip install ==========
if [[ -n "${PIP_PACKAGES}" ]]; then
    log "pip install: ${PIP_PACKAGES}"
    pip install ${PIP_PACKAGES} --no-cache-dir -i https://pypi.org/simple --trusted-host pypi.org
fi


# ========== 3) wget 下载安装类 ==========
# 在临时目录里 下载 -> 解压 -> pip install, 结束后删除。
# 不污染当前目录(脚本执行时 CWD 可能是挂载目录, 残留会落到宿主机)。
log "wget 安装: pssh 2.3.6"
pssh_dir=$(mktemp -d)
trap 'rm -rf "${pssh_dir}"' EXIT    # 失败也能清理临时目录
(
    cd "${pssh_dir}"
    net_retry wget https://github.com/lilydjwg/pssh/archive/refs/tags/v2.3.6.tar.gz -O pssh-2.3.6.tar.gz
    tar -xf pssh-2.3.6.tar.gz
    cd pssh-2.3.6
    pip install . --no-deps --no-build-isolation
)
rm -rf "${pssh_dir}" && trap - EXIT


# ========== 4) 系统配置(时区 / sshd) ==========
if [[ -n "${TZ_SET}" ]]; then
    log "设置时区: ${TZ_SET}"
    ln -fs "/usr/share/zoneinfo/${TZ_SET}" /etc/localtime
    echo "${TZ_SET}" > /etc/timezone
fi

if [[ -n "${SSH_PORT}" ]]; then
    log "配置 sshd: 端口 ${SSH_PORT}"
    mkdir -p /var/run/sshd
    sed -i "s/#Port 22/Port ${SSH_PORT}/" /etc/ssh/sshd_config
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
    sed -i 's/#AllowTcpForwarding yes/AllowTcpForwarding yes/' /etc/ssh/sshd_config
fi

log "依赖补全完成"
