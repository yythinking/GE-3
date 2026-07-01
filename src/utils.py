# src/utils.py
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from rouge import Rouge 

# === 基础计算工具 ===

def calculate_cosine_similarity(vec1, vec2):
    """计算余弦相似度"""
    vec1 = np.array(vec1).flatten()
    vec2 = np.array(vec2).flatten()
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))

def calculate_rouge_l_f1(prediction: str, reference: str) -> float:
    """计算 Rouge-L F1 分数"""
    if not prediction or not reference:
        return 0.0
    try:
        rouge = Rouge()
        scores = rouge.get_scores(prediction, reference)
        return scores[0]['rouge-l']['f']
    except Exception:
        return 0.0

# === 绘图工具 (从 Pipeline 移入并优化) ===

def generate_analysis_plots(history_metrics: List[dict], output_dir: str):
    """
    绘制详细分析图表
    优化：包含 SS 的每个坐标点（散点图）以及趋势线
    """
    if not history_metrics: return
    
    # 提取数据
    turns = [m['turn'] for m in history_metrics]
    sc = [m['sc'] for m in history_metrics]
    asr = [m['asr'] for m in history_metrics]
    
    # 语义相似度 (SS) - 回答 vs 问题
    avg_ss = [m['avg_ss'] for m in history_metrics]
    raw_ss_points = [m.get('current_ss', 0) for m in history_metrics] # 获取单点数据
    
    # 上下文相关度 (CRR) - 回答 vs 上下文
    avg_crr = [m['avg_crr'] for m in history_metrics]
    
    pool_size = [m['pool_size'] for m in history_metrics]
    
    # 设置风格
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Attack Efficiency: SC & ASR
    plt.figure(figsize=(12, 6))
    plt.plot(turns, sc, label='Storage Coverage (SC)', color='blue', linewidth=2)
    plt.plot(turns, asr, label='Attack Success Rate (ASR)', color='green', linestyle='--', linewidth=2)
    plt.xlabel('Turns (Interactions)')
    plt.ylabel('Rate (0-1)')
    plt.title('Attack Efficiency: SC & ASR')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_SC_ASR.png"))
    plt.close()
    
    # 2. Quality Metrics: SS (Detailed) & CRR
    plt.figure(figsize=(12, 6))
    
    # 绘制 SS 的所有散点 (半透明)
    plt.scatter(turns, raw_ss_points, color='purple', alpha=0.15, s=15, label='SS (Raw Points)')
    
    # 绘制 SS 和 CRR 的移动平均/趋势线
    plt.plot(turns, avg_ss, label='Avg SS', color='purple', linewidth=2)
    plt.plot(turns, avg_crr, label='Avg CRR', color='orange', linestyle='-.', linewidth=2)
    
    plt.xlabel('Turns (Interactions)')
    plt.ylabel('Cosine Similarity')
    plt.title('Quality Metrics: SS & CRR ')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_Quality_Metrics.png"))
    plt.close()
    
    # 3. Candidate Pool Dynamics
    plt.figure(figsize=(12, 6))
    plt.fill_between(turns, pool_size, color='skyblue', alpha=0.4)
    plt.plot(turns, pool_size, color='steelblue', linewidth=2)
    plt.xlabel('Turns')
    plt.ylabel('Active Candidates Count')
    plt.title('Candidate Pool Dynamics')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_Pool_Size.png"))
    plt.close()