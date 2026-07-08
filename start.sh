#!/bin/bash
# MiniMax-M3-NVFP4 启动脚本
# 日志: /root/JqLogs/coder.log
# 端口: 8000, 模型名: coder

cd /data/deepgemm_sglang_vllm/vllmsource20260630/vllm

export VLLM_MARLIN_USE_ATOMIC_ADD=1

nohup /data/deepgemm_sglang_vllm/vllmsource20260630/vllm/.venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model /data/models/MiniMax-M3-NVFP4 \
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
    --port 8000 \
    --served-model-name coder \
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 8}' \
    --speculative-config '{"method": "eagle3", "model": "/data/models/MiniMax-M3-EAGLE3", "num_speculative_tokens": 3, "attention_backend": "FLASH_ATTN"}' \
    --limit-mm-per-prompt '{"video": 0, "image": {"count": 5, "width": 512, "height": 512}}' \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --kv-offloading-size 64 \
    >> /root/JqLogs/coder.log 2>&1 &

echo "PID: $!"
echo "日志: tail -f /root/JqLogs/coder.log"
echo "停止: pkill -f 'vllm.*MiniMax-M3-NVFP4'"
