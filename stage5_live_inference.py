import json
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from minibert import MINI_BERT_CONFIG, MultiModalMiniBERTForClassification

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION & PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path("D:/BERT_project")
TOKENIZER_DIR = BASE_DIR / "finbert_tokenizer"
MODEL_WEIGHTS = BASE_DIR / "finbert_multimodal_trader" / "multimodal_quant_final.pt"
CORPUS_FILE = BASE_DIR / "data" / "clean" / "master_training_corpus.json"
TABULAR_DIR = BASE_DIR / "data" / "raw_tabular"

BLOCK_SIZE = 512
NUM_TABULAR_FEATURES = 5
DELTA_THRESHOLD = 0.04

LABEL_MAP_REVERSE = {0: "DOWN", 1: "NEUTRAL", 2: "UP"}
LABEL_MAP = {"DOWN": 0, "NEUTRAL": 1, "UP": 2}

_DF_CACHE = {}

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD THE TRAINED TRADER
# ─────────────────────────────────────────────────────────────────────────────
print("1. Waking up Quantitative Trading Algorithm...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = PreTrainedTokenizerFast.from_pretrained(str(TOKENIZER_DIR))
model = MultiModalMiniBERTForClassification(MINI_BERT_CONFIG, NUM_TABULAR_FEATURES, num_labels=3)

if MODEL_WEIGHTS.exists():
    model.load_state_dict(torch.load(str(MODEL_WEIGHTS), map_location=device))
    print(" -> SUCCESS: Core brain weight matrix fully operational.")
else:
    print(" -> [WARNING] Production weights not found. (Waiting for Stage 4B to finish!)")

model.to(device)
model.eval()

# ─────────────────────────────────────────────────────────────────────────────
# 2. FAST DATA RETRIEVAL (RAM CACHED)
# ─────────────────────────────────────────────────────────────────────────────
def _get_clean_dataframe(ticker):
    """Loads and cleans the CSV once, caching it in RAM for instant access."""
    if ticker in _DF_CACHE: return _DF_CACHE[ticker]
    price_file = TABULAR_DIR / f"{ticker}_prices.csv"
    if not price_file.exists(): price_file = TABULAR_DIR / f"{ticker.lower()}_prices.csv"
    if not price_file.exists():
        _DF_CACHE[ticker] = None
        return None
    try:
        df = pd.read_csv(price_file)
        if isinstance(df.columns, pd.MultiIndex) or 'Ticker' in df.iloc[0].values:
            df = pd.read_csv(price_file, header=[0,1])
            df.columns = [c[0] if not c[0].startswith('Unnamed') else c[1] for c in df.columns]
        df.columns = [str(c).strip().title() for c in df.columns]
        if 'Date' not in df.columns and df.columns[0] != 'Date':
            df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        _DF_CACHE[ticker] = df
        return df
    except:
        _DF_CACHE[ticker] = None
        return None

def get_tabular_features(df, date_str):
    """Calculates continuous market metrics instantly."""
    if df is None: return None
    try:
        target_col = next((opt for opt in ['Adj Close', 'Adj_Close', 'Close'] if opt in df.columns), None)
        if not target_col: return None

        past_data = df.loc[:pd.to_datetime(date_str)]
        if len(past_data) < 20: return None
            
        current, prev_20d = past_data.iloc[-1], past_data.iloc[-20]
        get_val = lambda v: float(v.iloc[0]) if isinstance(v, pd.Series) else float(v)
            
        vol = float(past_data[target_col].pct_change().tail(20).std())
        vol = 0.0 if np.isnan(vol) else vol
        vol_trend = float(past_data['Volume'].tail(5).mean() / past_data['Volume'].tail(20).mean())
        vol_trend = 1.0 if np.isnan(vol_trend) or np.isinf(vol_trend) else vol_trend
        mom = float((get_val(current[target_col]) - get_val(prev_20d[target_col])) / get_val(prev_20d[target_col]))
        rng = float((get_val(current['High']) - get_val(current['Low'])) / get_val(current['Close']))
        log_p = float(np.log(get_val(current[target_col]) + 1e-5))
        
        return [vol, vol_trend, mom, rng, log_p]
    except:
        return None

def get_target_label(df, date_str, delta_days=30, threshold=0.04):
    """Calculates ground truth directional target."""
    if df is None: return None
    try:
        target_col = next((opt for opt in ['Adj Close', 'Adj_Close', 'Close'] if opt in df.columns), None)
        if not target_col: return None

        avail_dates = df.index
        start_idx = avail_dates.get_indexer([pd.to_datetime(date_str)], method='bfill')[0]
        if start_idx == -1: return None
        
        future_idx = avail_dates.get_indexer([avail_dates[start_idx] + pd.Timedelta(days=delta_days)], method='bfill')[0]
        if future_idx == -1: return None
        
        get_val = lambda v: float(v.iloc[0]) if isinstance(v, pd.Series) else float(v)
        log_return = float(np.log(get_val(df.iloc[future_idx][target_col]) / get_val(df.iloc[start_idx][target_col])))
        
        if log_return > threshold: return 2
        elif log_return < -threshold: return 0
        return 1
    except:
        return None

# ─────────────────────────────────────────────────────────────────────────────
# 3. CORE INFERENCE LOGIC
# ─────────────────────────────────────────────────────────────────────────────
def generate_trading_signal(live_text, tabular_metrics):
    """Processes long documents via chunking and averages class probabilities."""
    tokens = tokenizer(live_text, add_special_tokens=True)["input_ids"]
    chunk_probabilities = []
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    
    with torch.no_grad():
        for i in range(0, len(tokens), BLOCK_SIZE):
            chunk = tokens[i : i + BLOCK_SIZE]
            if len(chunk) < BLOCK_SIZE:
                chunk = chunk + [pad_id] * (BLOCK_SIZE - len(chunk))
                
            input_ids_tensor = torch.tensor([chunk], dtype=torch.long).to(device)
            attention_mask_tensor = torch.ones((1, BLOCK_SIZE), dtype=torch.long).to(device)
            tabular_tensor = torch.tensor([tabular_metrics], dtype=torch.float32).to(device)
            
            outputs = model(
                input_ids=input_ids_tensor, tabular_features=tabular_tensor,
                attention_mask=attention_mask_tensor, segment_ids=None
            )
            
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs[0]
            chunk_probabilities.append(F.softmax(logits, dim=-1))
            
    if not chunk_probabilities:
        return "ERROR", None

    average_probs = torch.mean(torch.stack(chunk_probabilities), dim=0).squeeze()
    predicted_class_id = torch.argmax(average_probs).item()
    return LABEL_MAP_REVERSE[predicted_class_id], average_probs.cpu().numpy()

# ─────────────────────────────────────────────────────────────────────────────
# 4. 2026 OUT-OF-SAMPLE BACKTEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n2. Isolating 2026 Out-Of-Sample Data...")
    with open(CORPUS_FILE, 'r', encoding='utf-8') as f:
        corpus = json.load(f)
        
    # Strictly filter for documents published in the year 2026
    test_data_2026 = [row for row in corpus if row.get("date", "").startswith("2026")]
    
    if not test_data_2026:
        print("❌ Could not find any 2026 documents in the corpus! Check date formatting.")
        exit()
        
    print(f" -> Found {len(test_data_2026)} documents from 2026. Launching Backtest...\n")

    y_true = []
    y_pred = []

    for row in tqdm(test_data_2026, desc="Evaluating 2026 Data", unit="doc"):
        ticker = row.get("ticker", "").strip().upper()
        date_str = row.get("date")
        
        df = _get_clean_dataframe(ticker)
        
        # 1. Get True Label
        actual_numeric = get_target_label(df, date_str, delta_days=30, threshold=DELTA_THRESHOLD)
        if actual_numeric is None: continue
            
        # 2. Get Features for Model
        tabular_vector = get_tabular_features(df, date_str)
        if not tabular_vector: continue
            
        # 3. Get Model Prediction
        combined_text = f"[CLS] {row.get('sec_text', '')} [SEP] {row.get('macro_text', '')} [SEP]"
        predicted_signal, _ = generate_trading_signal(combined_text, tabular_vector)
        
        if predicted_signal == "ERROR": continue
            
        y_true.append(actual_numeric)
        y_pred.append(LABEL_MAP[predicted_signal])

    # ─────────────────────────────────────────────────────────────────────────────
    # 5. GENERATE PERFORMANCE REPORT
    # ─────────────────────────────────────────────────────────────────────────────
    if len(y_true) > 0:
        print("\n" + "="*60)
        print(" 🚀 2026 OUT-OF-SAMPLE TRADING PERFORMANCE METRICS ")
        print("="*60)
        
        accuracy = accuracy_score(y_true, y_pred)
        print(f"\nOverall Model Accuracy: {accuracy * 100:.2f}%\n")
        
        print("Detailed Classification Report:")
        print(classification_report(y_true, y_pred, target_names=["DOWN (0)", "NEUTRAL (1)", "UP (2)"]))
        
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        print("="*60)
    else:
        print("\n⚠️ No valid testable documents found. (Could be missing price data for 2026).")