import json
import pandas as pd
import numpy as np
from pathlib import Path
from datasets import Dataset
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm  # ✨ Added for a beautiful progress bar!

# ─────────────────────────────────────────────────────────────────────────────
# PATH CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path("D:/BERT_project")
CORPUS_FILE = BASE_DIR / "data" / "clean" / "master_training_corpus.json"
TABULAR_DIR = BASE_DIR / "data" / "raw_tabular"
TOKENIZER_DIR = BASE_DIR / "finbert_tokenizer"
CLASSIFICATION_ARROW_DIR = BASE_DIR / "data" / "arrow_multimodal_classification"

BLOCK_SIZE = 512
DELTA_THRESHOLD = 0.04  

# ✨ RAM CACHE: Prevents the script from reading the same CSV 10,000 times!
_DF_CACHE = {}

def _get_clean_dataframe(ticker):
    """Loads and cleans the CSV once, then caches it in RAM for instant access."""
    if ticker in _DF_CACHE:
        return _DF_CACHE[ticker]
        
    price_file = TABULAR_DIR / f"{ticker}_prices.csv"
    if not price_file.exists():
        price_file = TABULAR_DIR / f"{ticker.lower()}_prices.csv"
        
    if not price_file.exists():
        _DF_CACHE[ticker] = None
        return None
        
    try:
        df = pd.read_csv(price_file)
        
        # Strip/Clean column naming variations securely
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
    """Calculates 5 continuous market metrics instantly from RAM."""
    if df is None: return None
        
    try:
        target_col = None
        for option in ['Adj Close', 'Adj_Close', 'Close', 'Adj close']:
            if option in df.columns:
                target_col = option
                break
        if not target_col: return None

        target_date = pd.to_datetime(date_str)
        past_data = df.loc[:target_date]
        if len(past_data) < 20: 
            return None
            
        current = past_data.iloc[-1]
        prev_20d = past_data.iloc[-20]
        
        def get_series_values(val):
            if isinstance(val, pd.Series): return float(val.iloc[0])
            return float(val)
            
        volatility = float(past_data[target_col].pct_change().tail(20).std())
        if np.isnan(volatility): volatility = 0.0
            
        volume_trend = float(past_data['Volume'].tail(5).mean() / past_data['Volume'].tail(20).mean())
        if np.isnan(volume_trend) or np.isinf(volume_trend): volume_trend = 1.0
            
        momentum = float((get_series_values(current[target_col]) - get_series_values(prev_20d[target_col])) / get_series_values(prev_20d[target_col]))
        daily_range = float((get_series_values(current['High']) - get_series_values(current['Low'])) / get_series_values(current['Close']))
        log_price = float(np.log(get_series_values(current[target_col]) + 1e-5))
        
        return [volatility, volume_trend, momentum, daily_range, log_price]
    except:
        return None

def get_target_label(df, date_str, delta_days=30, threshold=0.04):
    """Calculates directional targets mapping log returns to 0 (DOWN), 1 (NEUTRAL), or 2 (UP)."""
    if df is None: return None
        
    try:
        target_col = None
        for option in ['Adj Close', 'Adj_Close', 'Close']:
            if option in df.columns:
                target_col = option
                break
        if not target_col: return None

        start_date = pd.to_datetime(date_str)
        avail_dates = df.index
        start_idx = avail_dates.get_indexer([start_date], method='bfill')[0]
        if start_idx == -1: return None
        
        future_date = avail_dates[start_idx] + pd.Timedelta(days=delta_days)
        future_idx = avail_dates.get_indexer([future_date], method='bfill')[0]
        if future_idx == -1: return None
        
        def get_series_values(val):
            if isinstance(val, pd.Series): return float(val.iloc[0])
            return float(val)

        p_start = get_series_values(df.iloc[start_idx][target_col])
        p_future = get_series_values(df.iloc[future_idx][target_col])
        
        log_return = float(np.log(p_future / p_start))
        
        if log_return > threshold:
            return 2  # UP
        elif log_return < -threshold:
            return 0  # DOWN
        else:
            return 1  # NEUTRAL
    except:
        return None

if __name__ == "__main__":
    print("1. Preparing Tokenization Engine for Classification Extraction...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(TOKENIZER_DIR))
    
    with open(CORPUS_FILE, 'r', encoding='utf-8') as f:
        corpus = json.load(f)
        
    classification_records = []
    print(f"\n2. Launching Alignment Engine across {len(corpus)} documents...")
    
    # ✨ WRAPPED IN TQDM FOR PROGRESS TRACKING
    for row in tqdm(corpus, desc="Extracting Multi-Modal Features", unit="doc"):
        ticker = row.get("ticker", "").strip().upper()
        date_str = row.get("date")
        
        # Get dataframe instantly from RAM cache
        df = _get_clean_dataframe(ticker)
        
        tabular_vector = get_tabular_features(df, date_str)
        if not tabular_vector:
            continue

        numeric_label = get_target_label(df, date_str, delta_days=30, threshold=DELTA_THRESHOLD)
        if numeric_label is None:
            continue

        combined_text = f"[CLS] {row.get('sec_text', '')} [SEP] {row.get('macro_text', '')} [SEP]"
        tokens = tokenizer.encode(combined_text, add_special_tokens=False)
        
        if len(tokens) == 0:
            continue
            
        # Slicing with Forgiving Auto-Padding Logic
        for i in range(0, len(tokens), BLOCK_SIZE):
            chunk = tokens[i : i + BLOCK_SIZE]
            
            if len(chunk) < BLOCK_SIZE:
                padding_needed = BLOCK_SIZE - len(chunk)
                attention_mask = [1] * len(chunk) + [0] * padding_needed
                chunk = chunk + [0] * padding_needed
            else:
                attention_mask = [1] * BLOCK_SIZE
                
            classification_records.append({
                "input_ids": chunk,
                "attention_mask": attention_mask,
                "tabular_features": tabular_vector,  
                "labels": numeric_label
            })

    print("\n3. Converting to High-Speed Multi-Modal Arrow Format...")
    if len(classification_records) == 0:
        print("❌ Error: Zero classification chunks were generated. Check dataset metrics!")
        exit()
        
    hf_dataset = Dataset.from_list(classification_records)
    hf_dataset = hf_dataset.shuffle(seed=42)
    hf_dataset.save_to_disk(str(CLASSIFICATION_ARROW_DIR))
    
    print(f"\n✅ Multi-Modal Classification Dataset Complete!")
    print(f"Total Chunks Generated: {len(hf_dataset)}")
    print(f"Saved Directory:        {CLASSIFICATION_ARROW_DIR.resolve()}")