# finBERT-Next-Deep-Multi-Modal-Framework-for-30-Day-Return-Forecasting

## 🧠 Project Overview
**finBERT-NEXT** is an end-to-end multi-modal quantitative deep learning framework designed to forecast 30-day stock market returns into discrete signals (**UP**, **DOWN**, **NEUTRAL**). 

Unlike standard language models that only interpret headlines, or traditional quantitative models that only analyze price charts, `finBERT-NEXT` implements a hybrid architectural design. It simultaneously ingests and fuses:
1. **Unstructured Financial Text:** Raw SEC Filings (10-Ks, 10-Qs) paired with macroeconomic reports.
2. **Structured Tabular Market Metrics:** A 5-dimensional continuous vector representing Volatility, Volume Trend, Momentum, Daily Trading Range, and Log Price.

By combining textual sentiment with mathematical price action, the model learns complex alpha-generating patterns that neither modality could capture alone.

---

## 🏗️ Core Architecture & Component Directory

The project structure is broken down into modular components across the model's entire lifecycle:

### 1. `minibert.py` (The Architectural Backbone)
This file defines the structural parameters and deep learning layers of the custom network. 
* `MINI_BERT_CONFIG`: Configures the structural dimensions optimized for custom financial domain tasks.
* `MultiModalMiniBERTForClassification`: The complete PyTorch neural network class. It defines how text tokens pass through the transformer self-attention layers, how the tabular features are projected through a dense linear layer, and how both modalities are concatenated into a unified embedding space before being fed into the final classification head.

### 2. Phase 2: Tokenization & Text Normalization (`stage2A.py` / `stage2B.py`)
* **Purpose:** Custom domain-specific text tokenization. This phase trains a specialized `Byte-Level BPE Tokenizer` from scratch using the training text to understand complex Wall Street jargon.
* **Outputs:** `finbert_tokenizer/` (containing `vocab.json`, `merges.txt`, and tokenizer configs).

### 3. Phase 3: Self-Supervised Pre-Training (`stage3_pretrain.py`)
* **Purpose:** Stage 3 pre-trains the base transformer layers using self-supervised objectives: **Masked Language Modeling (MLM)** and **Cross-Modal Matching**.
* **Output:** `multimodal_minibert_final.pt` (Base weight matrix containing general financial language intelligence).

### 4. Phase 4A: Multi-Modal Dataset Packing (`stage4A_multimodal_pack.py`)
* **Purpose:** The alignment engine. It maps text documents to their historical dates, extracts the quantitative features, computes future 30-day logarithmic returns, applies a `4% DELTA_THRESHOLD` to create target labels, and packs everything into high-performance binary arrays.
* **Output:** `data/arrow_multimodal_classification/` (Apache Arrow matrix optimized for rapid GPU streaming).

### 5. Phase 4B: Accelerated Fine-Tuning (`stage4B_multimodal_finetune.py`)
* **Purpose:** Discriminative supervised training. Freezes/adapts the pre-trained weights from Stage 3, initializes a clean classification head, and optimizes parameters across full training epochs using mixed precision (`fp16`).
* **Output:** `finbert_multimodal_trader/multimodal_quant_final.pt` (The final production trading weights).

### 6. Phase 5: Out-of-Sample Backtesting (`stage5_live_inference.py`)
* **Purpose:** Strict performance grading. Completely isolates textual and tabular records dated in the year **2026** (unseen data). It processes long documents through a rolling-window chunking loop, averages softmax probability states, and outputs complete quantitative grading metrics (Accuracy, Precision, Recall, F1, Confusion Matrix).

---

## 🚀 The End-to-End System Pipeline

```text
 [Raw SEC / Macro Text]            [Raw Stock Price CSVs]
         │                                   │
 ┌───────▼──────────────────────┐            │
 │ Stage 2A & 2B: Tokenization  │            │
 │ Trains custom BPE Vocabulary │            │
 └───────┬──────────────────────┘            │
         │ (finbert_tokenizer/)              │
 ┌───────▼──────────────────────┐            │
 │ Stage 3: Base Pre-Training   │◄───────────┤ (Self-Supervised
 │ Learns financial relations   │            │  Tabular Alignment)
 └───────┬──────────────────────┘            │
         │ (multimodal_minibert_final.pt)    │
 ┌───────▼──────────────────────┐            │
 │ Stage 4A: Multi-Modal Pack   │◄───────────┘ (Aligns dates, computes
 │ Tokenizes text + calculates  │               log returns & 5-quant features)
 │ 4% threshold target labels   │
 └───────┬──────────────────────┘
         │ (data/arrow_multimodal_classification)
 ┌───────▼──────────────────────┐
 │ Stage 4B: GPU Fine-Tuning    │ (Trains on High-End GPU VRAM
 │ Supervised Optimization      │  using Mixed-Precision FP16 backend)
 └───────┬──────────────────────┘
         │ (multimodal_quant_final.pt)
 ┌───────▼──────────────────────┐
 │ Stage 5: 2026 Backtesting    │──► [Outputs: Accuracy, Precision, Recall,
 │ Slices long text into chunks │     F1-Score, and Confusion Matrix]
 └──────────────────────────────┘
