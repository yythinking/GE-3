#!/bin/bash
# 差分隐私 RAG 攻击测试管道启动脚本
# 使用统一的 DP 参数配置

# DP 参数配置
N_SPLIT=30            # voter 数量 / prompt 变体数
DP_EPS=5.0            # 每个 token 的 DP epsilon
DP_DELTA=1e-6         # 每个 token 的 DP delta
TARGET_EPS=200.0      # 总 epsilon 预算上限
TARGET_DELTA=2e-4     # 总 delta 预算上限
MAX_TOKENS=50         # 生成的最大 token 数
FAIL_MODE="rand"      # DP 失败处理模式: ld_pate, rand, stop

python pipeline_DP.py \
    --dataset "./datasets/mini_trec_covid.json" \
    --llm "llama3.1:8b" \
    --llm_attacker "llama3.1:8b" \
    --mode "epoch" \
    --limit 1000 \
    --tp 10 \
    --tk 10 \
    --storage_base "./storage/DP_rag/COVID_bge-m3" \
    --output_base "./output_dp/" \
    --n_split ${N_SPLIT} \
    --dp_eps ${DP_EPS} \
    --dp_delta ${DP_DELTA} \
    --target_eps ${TARGET_EPS} \
    --target_delta ${TARGET_DELTA} \
    --max_tokens ${MAX_TOKENS} \
    --fail_mode ${FAIL_MODE}
