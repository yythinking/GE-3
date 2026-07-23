# src/utils.py
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from rouge import Rouge 

# === Basic Calculation Tools ===

def calculate_cosine_similarity(vec1, vec2):
    """Calculate cosine similarity"""
    vec1 = np.array(vec1).flatten()
    vec2 = np.array(vec2).flatten()
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))

def calculate_rouge_l_f1(prediction: str, reference: str) -> float:
    """Calculate Rouge-L F1 score"""
    if not prediction or not reference:
        return 0.0
    try:
        rouge = Rouge()
        scores = rouge.get_scores(prediction, reference)
        return scores[0]['rouge-l']['f']
    except Exception:
        return 0.0

# === Plotting Tools ===

def generate_analysis_plots(history_metrics: List[dict], output_dir: str):
    """
    Generate detailed analysis plots
    Optimization: includes each coordinate point of SS (scatter plot) and trend lines
    """
    if not history_metrics: return
    
    # Extract data
    turns = [m['turn'] for m in history_metrics]
    sc = [m['sc'] for m in history_metrics]
    asr = [m['asr'] for m in history_metrics]
    
    # Semantic Similarity (SS) - answer vs question
    avg_ss = [m['avg_ss'] for m in history_metrics]
    raw_ss_points = [m.get('current_ss', 0) for m in history_metrics] # Get single point data
    
    # Context Relevance Ratio (CRR) - answer vs context
    avg_crr = [m['avg_crr'] for m in history_metrics]
    
    pool_size = [m['pool_size'] for m in history_metrics]
    
    # Set style
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
    
    # Plot all scatter points of SS (semi-transparent)
    plt.scatter(turns, raw_ss_points, color='purple', alpha=0.15, s=15, label='SS (Raw Points)')
    
    # Plot moving average/trend lines of SS and CRR
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