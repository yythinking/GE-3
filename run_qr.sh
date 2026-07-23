#!/bin/bash
# run_qr.sh - Query Rewrite Defense Attack Testing Pipeline
# Launches pipeline_QueryRewrite.py with default configuration

python pipeline_QueryRewrite.py \
    --dataset "./datasets/mini_trec_covid.json" \
    --llm "llama3.1:8b" \
    --mode "epoch" \
    --limit 5 \
    --tp 10 \
    --tk 10 \
    --storage_base "./storage/QR_rag" \
    --output_base "./output" \
    --use_blacklist_detection \
    --use_llm_detection \
    --enable_qr_cache \
    --qr_cache_dir "./storage/query_rewrite_cache"
