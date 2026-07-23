#!/bin/bash
# run_sage.sh - SAGE RAG Attack Testing Pipeline
# Uses SAGE synthetic data protection (no differential privacy)

# =============================================================================
# SAGE Configuration
# =============================================================================
SYNTHETIC_MODE="agent2"                    # Synthetic method: sync (basic) / agent2 (agent-refined)
SYNTHETIC_CACHE_DIR="./storage/synthetic_data"  # Synthetic data cache directory

# =============================================================================
# Base Configuration
# =============================================================================
DATASET="./datasets/mini_trec_covid.json"  # Default dataset
LLM="llama3.1:8b"                          # LLM model
MODE="epoch"                               # Run mode: epoch / chunk
LIMIT=2                                    # Limit value
TP=10                                      # Retrieval Top P
TK=10                                      # Retrieval Top K
STORAGE_BASE="./storage/SAGE_rag"
OUTPUT_BASE="./output"

# =============================================================================
# Run Command
# =============================================================================
echo "================================================"
echo "SAGE RAG Attack Pipeline (No DP)"
echo "================================================"
echo "Dataset         : ${DATASET}"
echo "SAGE Mode       : ${SYNTHETIC_MODE}"
echo "Synthetic Cache : ${SYNTHETIC_CACHE_DIR}"
echo "Defense         : SAGE Only (No Differential Privacy)"
echo "Mode            : ${MODE}, Limit: ${LIMIT}"
echo "LLM             : ${LLM}"
echo "================================================"

python pipeline_SAGE.py \
    --dataset "${DATASET}" \
    --llm "${LLM}" \
    --mode "${MODE}" \
    --limit ${LIMIT} \
    --tp ${TP} \
    --tk ${TK} \
    --storage_base "${STORAGE_BASE}" \
    --output_base "${OUTPUT_BASE}" \
    --synthetic_mode "${SYNTHETIC_MODE}" \
    --synthetic_cache_dir "${SYNTHETIC_CACHE_DIR}"

# =============================================================================
# Usage:
# 
# 1. Basic usage (default configuration):
#    ./run_sage.sh
#
# 2. Change dataset:
#    DATASET="./datasets/mini_HealthCareMagic.json" ./run_sage.sh
#
# 3. Change synthetic method (agent-refined):
#    SYNTHETIC_MODE="agent2" ./run_sage.sh
#
# 4. Force rebuild synthetic data:
#    python pipeline_SAGE.py ... --rebuild_synthetic ...
#
# 5. View all parameters:
#    python pipeline_SAGE.py --help
#
# 6. Use a different LLM model:
#    LLM="gpt-4" ./run_sage.sh
# =============================================================================
