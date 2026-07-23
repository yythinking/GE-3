#!/bin/bash
# run_dp.sh - Differential Privacy RAG Attack Testing Pipeline
# Launches pipeline_DP.py with unified DP parameter configuration

# DP Parameter Configuration
N_SPLIT=30            # Number of voters / prompt variants
DP_EPS=5.0            # DP epsilon per token
DP_DELTA=1e-6         # DP delta per token
TARGET_EPS=200.0      # Total epsilon budget upper bound
TARGET_DELTA=2e-4     # Total delta budget upper bound
MAX_TOKENS=50         # Maximum number of tokens to generate
FAIL_MODE="rand"      # DP failure handling mode: ld_pate, rand, stop

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
