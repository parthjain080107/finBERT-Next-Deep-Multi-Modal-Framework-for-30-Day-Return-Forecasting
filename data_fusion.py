import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta

# ─────────────────────────────────────────────────────────────────────────────
# PATH CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path("D:/BERT_project")
DATA_DIR = BASE_DIR / "data"

SEC_SOURCE_DIR = DATA_DIR / "raw_sec_json"
TABULAR_DIR = DATA_DIR / "raw_tabular"
FOMC_DIR = DATA_DIR / "raw_extended" / "fomc"

OUTPUT_FILE = DATA_DIR / "clean" / "master_training_corpus.json"
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# Adjusted to a 4% threshold to filter out standard market noise 
VOLATILITY_THRESHOLD = 0.04 
FORWARD_WINDOW_DAYS = 30    

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD THE MACRO ENVIRONMENT (FOMC)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("          PHASE 2: MULTI-MODAL DATA FUSION & VECTORIZATION             ")
    print("=" * 70)
    print("\n1. Loading and sorting FOMC Macroeconomic Data...")
    
    fomc_records = []
    if FOMC_DIR.exists():
        for file in FOMC_DIR.glob("*.txt"):
            date_str = file.stem.split("_")[-1]
            try:
                date_obj = pd.to_datetime(date_str)
                with open(file, 'r', encoding='utf-8') as f:
                    f_text = f.read()
                fomc_records.append({"date": date_obj, "text": f_text})
            except:
                continue

    df_macro = pd.DataFrame(fomc_records).sort_values("date") if fomc_records else pd.DataFrame(columns=["date", "text"])
    print(f" -> Successfully loaded {len(df_macro)} FOMC macro statements.")

    def get_latest_macro_context(target_date):
        if df_macro.empty:
            return "No macroeconomic conditions found."
        past_statements = df_macro[df_macro['date'] <= pd.to_datetime(target_date)]
        if past_statements.empty:
            return df_macro.iloc[0]['text']
        return past_statements.iloc[-1]['text']

    # ─────────────────────────────────────────────────────────────────────────────
    # 2. RUN EMBEDDED CROSS-MODAL JOIN ENGINE
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n2. Executing Multimodal Data Fusion Joining Loop...")
    master_dataset = []
    stats = {"aligned": 0, "missing_price": 0, "insufficient_history": 0, "too_recent": 0}

    raw_sec_records = []
    if SEC_SOURCE_DIR.exists():
        for f in SEC_SOURCE_DIR.glob("*.json"):
            try:
                with open(f, 'r', encoding='utf-8') as src:
                    raw_sec_records.extend(json.load(src))
            except Exception as e:
                print(f"⚠️ Error reading SEC file {f.name}: {e}")

    print(f" -> Loaded {len(raw_sec_records)} raw SEC entries. Vectorizing market profiles...")

    for row in raw_sec_records:
        ticker = row.get("ticker", "").strip().upper()
        date_str = row.get("date")
        year = row.get("year", int(date_str.split('-')[0]) if date_str else 0)
        doc_text = row.get("sec_text", "")
        
        if not ticker or not date_str or not doc_text:
            continue
            
        price_file = TABULAR_DIR / f"{ticker}_prices.csv"
        if not price_file.exists():
            price_file = TABULAR_DIR / f"{ticker.lower()}_prices.csv"
            
        if not price_file.exists():
            stats["missing_price"] += 1
            continue
            
        try:
            df_prices = pd.read_csv(price_file)
            
            # Clean MultiIndex headers from yfinance
            if 'Ticker' in df_prices.iloc[0].values or 'Price' in df_prices.iloc[0].values:
                df_prices = pd.read_csv(price_file, header=[0, 1])
                df_prices.columns = [col[0] if not col[0].startswith('Unnamed') else col[1] for col in df_prices.columns]
            elif df_prices.columns.str.contains(r'\.').any() or isinstance(df_prices.columns, pd.MultiIndex):
                df_prices.columns = [col[0] if isinstance(col, tuple) else col for col in df_prices.columns]

            df_prices.columns = [str(c).strip().title() for c in df_prices.columns]
            if 'Date' not in df_prices.columns and df_prices.columns[0] != 'Date':
                df_prices.rename(columns={df_prices.columns[0]: 'Date'}, inplace=True)
                
            df_prices['Date'] = pd.to_datetime(df_prices['Date'])
            df_prices.set_index('Date', inplace=True)
            df_prices.sort_index(inplace=True)
            
            target_col = next((c for c in ['Adj Close', 'Adj_Close', 'Close'] if c in df_prices.columns), None)
            if target_col is None:
                stats["missing_price"] += 1
                continue

            doc_date = pd.to_datetime(date_str)
            
            # Look backwards to calculate technical feature vectors (requires at least 20 days of past history)
            past_data = df_prices.loc[:doc_date]
            if len(past_data) < 20:
                stats["insufficient_history"] += 1
                continue

            current_row = past_data.iloc[-1]
            prev_20d_row = past_data.iloc[-20]
            def val(v): return float(v.iloc[0]) if isinstance(v, pd.Series) else float(v)

            # Compute 5-Dimensional Tabular Vector
            volatility = float(past_data[target_col].pct_change().tail(20).std())
            volume_trend = float(past_data['Volume'].tail(5).mean() / past_data['Volume'].tail(20).mean())
            momentum = float((val(current_row[target_col]) - val(prev_20d_row[target_col])) / val(prev_20d_row[target_col]))
            daily_range = float((val(current_row['High']) - val(current_row['Low'])) / val(current_row['Close']))
            log_price = float(np.log(val(current_row[target_col]) + 1e-5))

            tabular_vector = [
                0.0 if np.isnan(volatility) else volatility,
                1.0 if (np.isnan(volume_trend) or np.isinf(volume_trend)) else volume_trend,
                momentum, 
                daily_range, 
                log_price
            ]

            # Look forward 30 days to compute the ground-truth prediction label
            start_idx = df_prices.index.get_indexer([doc_date], method='bfill')[0]
            future_date = df_prices.index[start_idx] + pd.Timedelta(days=FORWARD_WINDOW_DAYS)
            future_idx = df_prices.index.get_indexer([future_date], method='bfill')[0]
            
            if start_idx == -1 or future_idx == -1:
                stats["too_recent"] += 1
                continue

            log_return = float(np.log(val(df_prices.iloc[future_idx][target_col]) / val(df_prices.iloc[start_idx][target_col])))
            
            # Map labels to Integers for PyTorch Classification
            if log_return > VOLATILITY_THRESHOLD:
                label = 2  # UP
            elif log_return < -VOLATILITY_THRESHOLD:
                label = 0  # DOWN
            else:
                label = 1  # NEUTRAL
                
            macro_context = get_latest_macro_context(doc_date)
            
            master_dataset.append({
                "ticker": ticker,
                "date": doc_date.strftime("%Y-%m-%d"),
                "year": year, # Tagged explicitly for train/test splitting!
                "sec_text": doc_text,
                "macro_text": macro_context,
                "tabular_vector": tabular_vector,
                "label": label
            })
            stats["aligned"] += 1
            
            if stats["aligned"] % 500 == 0:
                print(f" -> Processed {stats['aligned']} multimodal profiles...")
            
        except Exception as e:
            stats["missing_price"] += 1

    # ─────────────────────────────────────────────────────────────────────────────
    # 3. SAVING THE NEURAL TRAINING WAREHOUSE
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n3. Saving Master Training Corpus...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(master_dataset, f, indent=4)

    print("\n" + "-" * 70)
    print(" ✅ STAGE 2 FUSION & VECTORIZATION REPORT ")
    print("-" * 70)
    print(f" Total Multi-Modal Records Fused : {stats['aligned']}")
    print(f" Dropped (Insufficient History)  : {stats['insufficient_history']}")
    print(f" Dropped (Missing Pricing Data)  : {stats['missing_price']}")
    print(f" Dropped (No Future Reference)   : {stats['too_recent']}")
    print(f" Target Master Vector Array File : {OUTPUT_FILE.resolve()}")
    print("-" * 70)