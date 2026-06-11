import os
import pandas as pd
import yfinance as yf
from pathlib import Path
import requests

# ─────────────────────────────────────────────────────────────────────────────
# PATH CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path("D:/BERT_project")
TABULAR_DIR = BASE_DIR / "data" / "raw_tabular"
TABULAR_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. FETCH THE S&P 500 TICKERS (THE CLEAN WAY)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("      PHASE 1: HISTORICAL MARKET DATA ACQUISITION (2019-2026)        ")
    print("=" * 70)
    print("\n1. Fetching S&P 500 Ticker List from GitHub Data Repository...")
    
    try:
        # Read a perfectly formatted CSV directly from a reliable datasets repo
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        df_sp500 = pd.read_csv(url)
        
        # Extract the 'Symbol' column to a list
        tickers = df_sp500['Symbol'].tolist()
        
        # Yahoo Finance uses hyphens instead of dots (e.g., BRK.B -> BRK-B)
        tickers = [t.replace('.', '-') for t in tickers]
        
        print(f" -> Successfully loaded {len(tickers)} companies without any HTML mess.")
        
    except Exception as e:
        print(f"[CRITICAL ERROR] Could not fetch tickers: {e}")
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]

    # ─────────────────────────────────────────────────────────────────────────────
    # 2. MASSIVE MARKET DATA DOWNLOAD
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n2. Initiating Historical Price Downloads (2019 - 2026)...")

    success_count = 0
    fail_count = 0

    for idx, ticker in enumerate(tickers, 1):
        output_file = TABULAR_DIR / f"{ticker}_prices.csv"
        
        # Skip if we already downloaded it (useful if the script crashes halfway)
        if output_file.exists():
            success_count += 1
            continue
            
        try:
            # CRITICAL CHANGE: Extended 'end' to 2027 to ensure ALL 2026 data is captured
            df = yf.download(ticker, start="2019-01-01", end="2027-01-01", progress=False)
            
            if df.empty:
                fail_count += 1
                continue
                
            # Clean up the dataframe before saving
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            # Strip the timezone from the date index so it's clean (e.g., "2023-10-31")
            df.index = pd.to_datetime(df.index).tz_localize(None)
            
            # Save to a CSV file
            df.to_csv(output_file)
            success_count += 1
            
        except Exception as e:
            fail_count += 1
            
        # Progress tracking
        if idx % 50 == 0:
            print(f" -> Progress: Downloaded market profiles for {idx}/{len(tickers)} companies...")

    # ─────────────────────────────────────────────────────────────────────────────
    # 3. COMPLETION REPORT
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n" + "-" * 70)
    print(" ✅ MARKET DATA ACQUISITION REPORT ")
    print("-" * 70)
    print(f" Total Companies Attempted : {len(tickers)}")
    print(f" Successfully Saved CSVs   : {success_count}")
    print(f" Failed Downloads          : {fail_count}")
    print(f" Data secured in           : {TABULAR_DIR.resolve()}")
    print("-" * 70)