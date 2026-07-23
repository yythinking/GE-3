<div align="center">

# [Anonymous Submission] GE³: Stealthy Knowledge Extraction from RAG Systems via Benign Queries

[![Status](https://img.shields.io/badge/Status-Under%20Review-orange.svg)]()
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

**[Anonymous Repository for Double-Blind Review]**

</div>

---

## Overview

This repository provides the implementation of **GE³ (Grounded Exploitation–Exploration Extraction)**, a black-box knowledge extraction framework for RAG (Retrieval-Augmented Generation) systems. The framework operates through benign conversational queries to progressively extract the underlying knowledge corpus of a target RAG system.

Key contributions:
- **Echo Knowledge Overflow (EKO):** A newly identified vulnerability where benign inducing suffixes implicitly encourage RAG systems to expose latent retrieval knowledge beyond the directly generated response.
- **Dual-track candidate generation:** Combines LLM-driven contextual reasoning (Exploration) with EKO-driven grounded signal extraction (Exploitation) to generate semantically grounded candidate queries.
- **Maximin-distance query selection:** Formulates query selection as a maximin-distance optimization problem to prioritize diverse and high-value exploration trajectories under a limited query budget.

## ⚠️ Disclaimer

> This repository is intended for **academic research and educational purposes only**. The techniques and tools provided herein must only be used on systems for which you have explicit, authorized permission to test.

## 📖 Threat Model

The adversary interacts with a black-box RAG system through a standard query-response interface. The attacker has:
- **No access** to the system's internal parameters, retrieval index, similarity scores, document identifiers, reranking results, or system prompts.
- **Only observable:** the final filtered responses returned by the system.
- **Local history:** the attacker maintains a local interaction history to guide subsequent query generation.

The target RAG system is protected by standard guardrails: keyword-based intent detection at the input layer, and ROUGE-L-based output filtering at the output layer.

## 📂 Repository Structure

```text
GE³/
├── pipeline_normal.py          # Attack pipeline against Standard RAG (baseline)
├── pipeline_DP.py              # Attack pipeline against DPRAG defense
├── pipeline_QueryRewrite.py    # Attack pipeline against Query Rewriting defense
├── pipeline_SAGE.py            # Attack pipeline against SAGE defense
├── run_dp.sh                   # Shell script: run DPRAG experiments
├── run_qr.sh                   # Shell script: run Query Rewriting experiments
├── run_sage.sh                 # Shell script: run SAGE experiments
│
├── agent/                      # GE³ Adversarial Agent (LangGraph)
│   ├── graph.py                #   Multi-turn attack orchestration (Algorithm 1)
│   ├── prompts.py              #   Prompts for dual-track candidate generation
│   └── utils.py                #   Candidate pool management, deduplication, pruning
│
├── rag/                        # RAG Engine Implementations
│   ├── base_engine.py          #   Abstract base class with safety guardrails
│   ├── standard_rag.py         #   Standard RAG engine (baseline target)
│   ├── DP_RAG.py               #   DPRAG engine (vote-based DP mechanism)
│   ├── qr_rag.py               #   Query Rewriting defense engine
│   ├── sage_rag.py             #   SAGE synthetic data RAG engine
│   ├── sage_engine.py          #   SAGE synthetic data generation
│   ├── dp_mechanisms.py        #   LDGumbel DP mechanism
│   └── prompts.py              #   RAG prompt templates
│
├── models/                     # Model Abstraction Layer
│   ├── interfaces/             #   Abstract interfaces (LLM, Embedding, Reranker)
│   ├── llms/                   #   LLM backends (Ollama, OpenAI-compatible)
│   ├── embeddings/             #   Embedding backends (HuggingFace)
│   └── rerankers/              #   Reranker backends (CrossEncoder, NoReranker)
│
├── src/                        # Utilities
│   ├── data_loader.py          #   Dataset loading and chunking
│   ├── evaluator.py            #   QA pair generation for evaluation
│   └── utils.py                #   Similarity computation and plotting
│
├── datasets/                   # Evaluation Datasets (mini subsets)
│   ├── mini_trec_covid.json    #   TREC-COVID (biomedical)
│   ├── mini_scidocs.json       #   SciDocs (scientific literature)
│   ├── mini_nfcopurs.json      #   NFCorpus (biomedical)
│   └── mini_HealthCareMagic.json # HealthcareMagic (medical dialogue)
│
├── requirements.txt
└── .env.example
```

## ⚙️ Setup

### Prerequisites

- Python 3.11+
- NVIDIA GPU with CUDA (for embedding and reranker models)
- [Ollama](https://ollama.ai/) installed and running (for local LLM inference)

### Installation

```bash
# 1. Clone
git clone <ANONYMOUS_GITHUB_CLONE_URL>
cd GE-3

# 2. Create environment
conda create -n ge3 python=3.11 -y
conda activate ge3

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull the default attacker LLM
ollama pull llama3.1:8b

# 5. (Optional) Configure cloud-based LLM API keys
cp .env.example .env
# Edit .env with your credentials
```

See [`.env.example`](.env.example) for supported API providers.

## 🚀 Running Experiments

Each pipeline corresponds to one defense scenario. All pipelines share the same command-line interface.

```bash
# Baseline (no defense)
python pipeline_normal.py --dataset ./datasets/mini_trec_covid.json

# DPRAG defense
bash run_dp.sh

# Query Rewriting defense
bash run_qr.sh

# SAGE defense
bash run_sage.sh
```

Run `python pipeline_<defense>.py --help` for all available arguments.

### Output

Results are saved to `./output/`:

```text
output/<dataset>_<mode>_<limit>_<defense_info>_<timestamp>/
├── final_metrics.json              # CR, SF, ASR, and other metrics
├── realtime_sc_metrics.jsonl       # Per-turn coverage tracking
├── extracted_dataset_full.json     # Extracted knowledge chunks
├── log.json                        # Rejected queries
└── plot_*.png                      # Analysis plots
```

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **CR** | Coverage Rate — fraction of the target database exposed during interaction |
| **SF** | Semantic Fidelity — semantic consistency between extracted corpus and original database |
| **ASR** | Attack Success Rate — fraction of queries bypassing guardrails with substantive content |

## 📂 Datasets

| Dataset | Domain | Description |
|---------|--------|-------------|
| `mini_trec_covid.json` | Biomedical | COVID-19 research papers (TREC-COVID) |
| `mini_scidocs.json` | Scientific literature | Scientific paper abstracts (SciDocs) |
| `mini_nfcopurs.json` | Consumer health | Consumer-health documents (NFCorpus) |
| `mini_HealthCareMagic.json` | Clinical dialogue | Doctor-patient medical dialogue samples (HealthcareMagic) |

All datasets are JSON files with `input` and `output` fields.

## 📜 License

This project is licensed under the [MIT License](LICENSE).
