#!/bin/bash
# MiniMax-M3-NVFP4 vLLM 启动脚本 (auto 版: 含资源清理)
# 模型: /data/models/MiniMax-M3-NVFP4 (EAGLE3 推测解码)
# 日志: /root/JqLogs/coder.log
# 端口: 8000, 模型名: coder

set -e

# === 配置 ===
MODEL_PATH="/data/models/MiniMax-M3-NVFP4"
PROJECT_DIR="/data/deepgemm_sglang_vllm/vllmsource20260630/vllm"
VENV_PATH="${PROJECT_DIR}/.venv"
LOG_PATH="/root/JqLogs/coder.log"
PORT=8000

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step()  { echo -e "${CYAN}[STEP]${NC} $1"; }

cd "${PROJECT_DIR}"

# === 资源检查 ===

# 检查端口占用
check_port() {
    if lsof -i :${PORT} >/dev/null 2>&1; then
        log_warn "端口 ${PORT} 被占用:"
        lsof -i :${PORT} | grep -v "^COMMAND" || true
        return 1
    fi
    return 0
}

# 检查 GPU 上的推理服务进程 (vLLM / SGLang)
check_gpu() {
    local patterns="[p]ython.*[v]llm|[v]llm.*serve|[p]ython.*[s]glang|[s]glang.*server"
    local gpu_procs=$(nvidia-smi 2>/dev/null | grep -iE "[v]llm|[s]glang|[p]ython" || true)
    local ps_procs=$(ps aux | grep -iE "$patterns" | grep -v grep || true)

    if [ -n "$gpu_procs" ] || [ -n "$ps_procs" ]; then
        log_warn "检测到推理服务进程:"
        [ -n "$gpu_procs" ] && echo "  [nvidia-smi]:" && echo "$gpu_procs" | head -3 || true
        [ -n "$ps_procs" ]  && echo "  [ps]:"        && echo "$ps_procs"  | head -5 || true
        return 1
    fi
    return 0
}

# === 资源清理 ===

# 清理端口
cleanup_port() {
    log_step "清理端口 ${PORT}..."
    local pids=$(lsof -ti :${PORT} 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "  释放 PID: $(echo $pids | tr '\n' ' ')"
        echo "$pids" | xargs -r kill -9 2>/dev/null || true
        sleep 1
        log_info "端口 ${PORT} 已释放"
    else
        log_info "端口 ${PORT} 无需清理"
    fi
}

# 找出所有占用 GPU 的推理服务 PID
_find_gpu_infer_pids() {
    local gpu_pids=""
    if nvidia-smi 2>/dev/null | grep -qiE "[v]llm|[s]glang"; then
        gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sort -u || true)
    fi
    local ps_pids=$(ps aux | grep -iE "[p]ython.*[v]llm|[v]llm.*serve|[p]ython.*[s]glang|[s]glang.*server|[v]llm::[a-z]|[s]glang::[a-z]" | awk '{print $2}' || true)
    echo -e "${gpu_pids:-} ${ps_pids:-}" | tr ' ' '\n' | grep -v '^\s*$' | sort -u
}

# 清理 GPU 上的推理服务进程
cleanup_gpu() {
    log_step "清理 GPU 上的推理服务进程..."
    local pids=$(_find_gpu_infer_pids)
    if [ -n "$pids" ]; then
        echo "  清理 PID: $(echo $pids | tr '\n' ' ')"
        echo "$pids" | xargs -r kill -9 2>/dev/null || true
        sleep 2
        log_info "GPU 进程已清理"
    else
        log_info "无推理服务进程占用 GPU"
    fi
}

# === 启动 ===
start_local() {
    log_step "预检查资源占用..."
    local needs_cleanup=0
    if ! check_port; then needs_cleanup=1; fi
    if ! check_gpu;  then needs_cleanup=1; fi

    if [ "$needs_cleanup" -eq 1 ]; then
        echo ""
        log_step "检测到资源占用, 自动清理..."
        cleanup_port
        cleanup_gpu
        sleep 2

        # 再次检查
        local still_occupied=0
        if ! check_port 2>/dev/null; then still_occupied=1; fi
        if ! check_gpu  2>/dev/null; then still_occupied=1; fi
        if [ "$still_occupied" -eq 1 ]; then
            log_error "清理后仍有资源占用, 请手动处理后重试"
            exit 1
        fi
        log_info "✅ 资源清理完成"
    else
        log_info "✅ 资源检查通过, 无需清理"
    fi

    if [ ! -d "${VENV_PATH}" ]; then
        log_error "虚拟环境不存在: ${VENV_PATH}"
        exit 1
    fi
    if ! "${VENV_PATH}/bin/python" -c "import vllm" >/dev/null 2>&1; then
        log_error "vllm 未安装到 ${VENV_PATH}"
        exit 1
    fi

    echo ""
    log_info "=== 启动 MiniMax-M3-NVFP4 vLLM ==="
    log_info "venv:   ${VENV_PATH}"
    log_info "模型:   ${MODEL_PATH}"
    log_info "端口:   ${PORT}"
    log_info "日志:   ${LOG_PATH}"

    export VLLM_MARLIN_USE_ATOMIC_ADD=1

    # 关键: 让 vllm 服务运行在完全独立的 session 中, 脚本退出 / Ctrl+C 都不会影响它
    #   - setsid: 创建新 session, vllm 成为 session leader
    #   - nohup:  屏蔽 SIGHUP
    #   - < /dev/null: stdin 不关联当前 tty
    #   - &:      后台运行
    #   - disown: 从 job table 移除, shell 退出不发送 SIGHUP
    setsid nohup "${VENV_PATH}/bin/python" -m vllm.entrypoints.openai.api_server \
        --model "${MODEL_PATH}" \
        --block-size 128 \
        --tool-call-parser minimax_m3 \
        --enable-auto-tool-choice \
        --reasoning-parser minimax_m3 \
        --kv-cache-dtype auto \
        --tensor-parallel-size 8 \
        --gpu-memory-utilization 0.92 \
        --max-num-batched-tokens 8192 \
        --max-model-len 202000 \
        --max-num-seqs 8 \
        --trust-remote-code \
        --host 0.0.0.0 \
        --port ${PORT} \
        --served-model-name coder \
        --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 8}' \
        --speculative-config '{"method": "eagle3", "model": "/data/models/MiniMax-M3-EAGLE3", "num_speculative_tokens": 3, "attention_backend": "FLASH_ATTN"}' \
        --limit-mm-per-prompt '{"video": 0, "image": {"count": 5, "width": 512, "height": 512}}' \
        --enable-prefix-caching \
        --enable-chunked-prefill \
        --kv-offloading-size 64 \
        >> "${LOG_PATH}" 2>&1 < /dev/null &

    VLLM_PID=$!
    disown ${VLLM_PID} 2>/dev/null || true

    # 给服务几秒开始写日志
    sleep 2

    echo ""
    log_info "✅ vLLM 服务已在后台启动"
    log_info "   PID: ${VLLM_PID} (独立 session, 不受 Ctrl+C / 脚本退出影响)"
    log_info "   日志: ${LOG_PATH}"
    log_info ""
    log_info "📌 Ctrl+C 退出 tail 后, vllm 服务继续运行"
    log_info "🔧 停止服务: $0 stop"
    log_info ""
    log_info "=== 跟踪日志 (Ctrl+C 退出不影响服务) ==="

    tail -f "${LOG_PATH}"
}

# 停止服务
stop() {
    log_step "停止服务..."
    cleanup_port
    cleanup_gpu
    log_info "✅ 所有服务已停止"
}

# 查看状态
status() {
    log_step "检查服务状态..."
    echo ""
    echo "=== 端口 ${PORT} ==="
    if lsof -i :${PORT} >/dev/null 2>&1; then
        lsof -i :${PORT}
    else
        echo "端口空闲"
    fi
    echo ""
    echo "=== GPU 推理服务 (vLLM / SGLang) ==="
    if nvidia-smi 2>/dev/null | grep -qiE "[v]llm|[s]glang|[p]ython"; then
        nvidia-smi 2>/dev/null | grep -iE "[v]llm|[s]glang|[p]ython" | head -3
    else
        echo "(nvidia-smi: 无活跃 GPU 进程)"
    fi
    local PS_PROCS=$(ps aux | grep -iE "[p]ython.*[v]llm|[v]llm.*serve|[p]ython.*[s]glang|[s]glang.*server" | grep -v grep || true)
    if [ -n "$PS_PROCS" ]; then
        echo "$PS_PROCS" | head -3
    else
        echo "(ps: 无推理服务进程)"
    fi
}

# 强制清理
clean() {
    cleanup_port
    cleanup_gpu
}

# 帮助
usage() {
    cat <<EOF
MiniMax-M3-NVFP4 vLLM 启动脚本 (auto 版: 含资源清理)

用法: $0 [命令]

命令:
  start    启动服务 (默认; 占用资源时自动清理)
  stop     停止所有相关服务 (释放端口 + 清理 GPU 进程)
  restart  重启服务
  status   查看服务状态 (端口 + GPU 推理进程)
  clean    强制清理资源 (不启动)
  -h, --help  显示帮助

示例:
  $0              # 启动服务
  $0 stop         # 停止服务
  $0 status       # 查看状态
  $0 restart      # 重启
EOF
}

# === 主逻辑 ===
case "${1:-start}" in
    start|"")
        start_local
        ;;
    stop)
        stop
        ;;
    restart)
        stop
        sleep 2
        start_local
        ;;
    status)
        status
        ;;
    clean)
        clean
        ;;
    -h|--help)
        usage
        ;;
    *)
        log_error "未知命令: $1"
        usage
        exit 1
        ;;
esac
