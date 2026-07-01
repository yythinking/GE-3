#!/bin/bash
# run_sage.sh - SAGE RAG 攻击测试管道启动脚本
# 使用SAGE合成数据保护，不包含差分隐私(DP)机制

# =============================================================================
# SAGE 配置
# =============================================================================
SYNTHETIC_MODE="agent2"                    # 合成方法: sync (基础) / agent2 (Agent精炼)
SYNTHETIC_CACHE_DIR="./storage/synthetic_data"  # 合成数据缓存目录

# =============================================================================
# 基础配置
# =============================================================================
DATASET="./datasets/mini_trec_covid.json"  # 默认数据集
LLM="llama3.1:8b"                          # LLM 模型
MODE="epoch"                               # 运行模式: epoch / chunk
LIMIT=2                                    # 限制数值
TP=10                                      # RAG检索 Top P
TK=10                                      # RAG检索 Top K
STORAGE_BASE="./storage/SAGE_rag"
OUTPUT_BASE="./output"

# =============================================================================
# 运行命令
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
# 脚本用法说明:
# 
# 1. 基础使用 (使用默认配置):
#    ./run_sage.sh
#
# 2. 修改数据集:
#    DATASET="./datasets/mini_HealthCareMagic.json" ./run_sage.sh
#
# 3. 修改合成方法 (需要Agent精炼):
#    SYNTHETIC_MODE="agent2" ./run_sage.sh
#
# 4. 强制重建合成数据:
#    python pipeline_SAGE.py ... --rebuild_synthetic ...
#
# 5. 查看所有参数:
#    python pipeline_SAGE.py --help
#
# 6. 使用不同的LLM模型:
#    LLM="gpt-4" ./run_sage.sh
# =============================================================================