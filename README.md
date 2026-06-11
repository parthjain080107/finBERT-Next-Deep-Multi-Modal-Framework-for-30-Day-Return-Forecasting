# finBERT-Next-Deep-Multi-Modal-Framework-for-30-Day-Return-Forecasting

##  Project Overview
**finBERT-NEXT** is an end-to-end multi-modal quantitative deep learning framework designed to forecast 30-day stock market returns into discrete signals (**UP**, **DOWN**, **NEUTRAL**). 

Unlike standard language models that only interpret headlines, or traditional quantitative models that only analyze price charts, `finBERT-NEXT` implements a hybrid architectural design. It simultaneously ingests and fuses:
1. **Unstructured Financial Text:** Raw SEC Filings (10-Ks, 10-Qs) paired with macroeconomic reports.
2. **Structured Tabular Market Metrics:** A 5-dimensional continuous vector representing Volatility, Volume Trend, Momentum, Daily Trading Range, and Log Price.

By combining textual sentiment with mathematical price action, the model learns complex alpha-generating patterns that neither modality could capture alone.

---
##  Project Objectives
The primary goal of this project is to build a high-performance, production-grade quantitative trading brain that uses multimodal deep learning to predict market direction. 
* **Multimodal Integration:** Seamlessly combine unstructured text (such as 10-K filings, financial news, or earnings transcripts) with structured, continuous tabular parameters (5 custom quant metrics).
* **Directional Classification:** Predict short-to-medium-term market movements into three distinct categories: `DOWN (0)`, `NEUTRAL (1)`, and `UP (2)`.
* **Risk-Averse Alpha Generation:** Optimize the model architecture to maintain high precision for long signals (`UP`) to avoid buying into market traps, while maintaining high recall on short signals (`DOWN`) for robust portfolio hedging.

---
##  Context: The System Explained Simply

To make a prediction, the model acts like a professional Wall Street analyst—it simultaneously reads the **"vibe" (the words)** and tracks the **"math" (the numbers)** across 5 distinct data feeds:

### 1. SEC Filings (10-K & 10-Q)
By law, public companies must file these truth-backed reports with the government.
*   **10-K:** A massive **annual** report breaking down the company's financial health, structural risks, and future strategy.
*   **10-Q:** A lighter, **quarterly** update showing how the company is performing every three months.

### 2. Earnings Calls
Four times a year, corporate executives host a live call to discuss their financial results. The model reads the **written transcript** of these calls, analyzing both the prepared speeches and the unscripted Q&A sessions where nervous or confident executive answers tip off the market.

### 3. FOMC Minutes (Macroeconomic Text)
This looks at the "big picture" economic weather. The FOMC is the group at the Federal Reserve that sets **interest rates**. The model scans their official meeting notes and statements word-by-word to spot clues about inflation, recession risks, and sweeping market shifts.

### 4. Tabular Data (Market Prices)
This is the raw mathematical reality. It skips the words and looks at structured spreadsheet columns tracking **5 continuous market metrics**: live stock prices, trading volume (how many shares are changing hands), and mathematical momentum indicators.

---

###  What are we getting from the Final Model?

The final model serves as an automated, multi-modal **"Trading Brain."** Instead of a human spending hours reading a 150-page document and staring at stock charts, the model processes both instantly. 

Upon receiving a new event, the model fuses the text embeddings and numerical metrics into a single calculation matrix and outputs a **Directional Signal**.

###  The Meaning of the Signals

The model classifies every market event into one of three actionable execution signals:

*   **DOWN (0): A Strong Sell / Hedge Signal**
    *   *What it means:* The model detects severe underlying trouble (e.g., toxic phrasing in a 10-K combined with collapsing price momentum). 
    *   *Action:* In live trading, this tells the system to sell the stock, avoid buying it, or open a short/hedge position to protect capital.
*   **NEUTRAL (1): A "Do Nothing" / Hold Signal**
    *   *What it means:* The text and numbers suggest the market is consolidating, moving sideways, or that there isn't enough high-conviction data to make a directional bet.
    *   *Action:* The system stays flat, holding current cash. 
*   **UP (2): A High-Conviction Buy Signal**
    *   *What it means:* The model detects a powerful alignment of positive news (e.g., strong earnings call Q&A) and healthy price math.
    *   *Action:* The system triggers a long entry (buys the asset).
##  Model Parameters & Configuration
The neural network architecture bridges a custom Mini-BERT backbone with a dense tabular processing pipeline using a Hugging Face Trainer wrapper.

### Architecture & Input Configuration
* **Tabular Features:** 5 continuous numerical parameters
* **Classification Head Output:** 3 classes (`DOWN`, `NEUTRAL`, `UP`)
* **Sequence Length:** Up to 512 tokens
* **Number of Attention Heads:** [8]
* **Hidden Layer Dimension:** [512]

### Fine-Tuning Hyperparameters (Stage 4B)
* **Training Epochs:** 4
* **Learning Rate:** 2e-5
* **Weight Decay:** 0.01
* **Evaluation Strategy:** Epoch-based validation tracking
* **Optimization Trick:** Best model weights automatically loaded at the end of the optimization run

---
##  Core Architecture & Component Directory

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

##  The End-to-End System Pipeline

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
```
##  Evaluation Metrics & Final Results
The model was subjected to a rigorous backtest using completely **2026 Out-of-Sample (OOS) Data** consisting of **401 unseen documents** and corresponding tabular metrics.

### Overall Performance
* **Overall Model Accuracy:** **83.54%**

### Detailed Classification Report
| Class / Label | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **DOWN (0)** | 0.83 | 0.95 | 0.89 | 205 |
| **NEUTRAL (1)** | 0.82 | 0.80 | 0.81 | 128 |
| **UP (2)** | 0.93 | 0.56 | 0.70 | 68 |
| **Macro Average** | 0.86 | 0.77 | 0.80 | 401 |
| **Weighted Average** | 0.84 | 0.84 | 0.83 | 401 |


##  Limitations

- **Class Imbalance:** UP signals (68 samples) are underrepresented vs DOWN (205). Need to consider weighted loss functions for production.
*Class Imbalance Analysis*
- DOWN is 3x more frequent than UP
- Precision for UP (0.93) is strong but Recall is low (0.56)
- Recommendation: Use `class_weights` or threshold tuning in production
- Consider F1-Score weighted strategy for trading system priority
- 
- **Lookback Period:** Model trained on historical data; future market regimes may differ.
- **Data Lag:** SEC filings have inherent publication delays.
- **Not Financial Advice:** Model outputs are signals, not guaranteed trading recommendations.
