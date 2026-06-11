import os
import json
from pathlib import Path
from datasets import load_dataset

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path("D:/BERT_project")
TRANSCRIPT_DIR = BASE_DIR / "data" / "raw_extended" / "earnings_calls"
TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. DOWNLOAD THE MASSIVE DATASET
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("      PHASE 1: EARNINGS CALL TRANSCRIPT ACQUISITION (2019-2026)      ")
print("=" * 70)
print("\n1. Connecting to Hugging Face Datasets...")
print(" -> Downloading S&P 500 Earnings Call Transcripts (~20,000 documents)...")

try:
    # This pulls the dataset directly into memory. It might take a minute!
    dataset = load_dataset("glopardo/sp500-earnings-transcripts", split="train")
    print(f" -> Successfully loaded {len(dataset)} transcripts.")
except Exception as e:
    print(f"[CRITICAL ERROR] Failed to download dataset: {e}")
    exit()

# ─────────────────────────────────────────────────────────────────────────────
# 2. FILTER & SAVE TO LOCAL DIRECTORY
# ─────────────────────────────────────────────────────────────────────────────
print("\n2. Processing and saving transcripts to local folders...")

success_count = 0
skipped_count = 0

# CRITICAL CHANGE: Expanded to 2026 to ensure we have isolated evaluation data!
TARGET_YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]

for row in dataset:
    # Extract metadata from the Hugging Face row
    ticker = row.get("ticker")
    year = row.get("year")
    quarter = row.get("quarter") # Usually formatted as "Q1", "Q2", etc.
    transcript_text = row.get("transcript")
    
    # Filter by our target years (2019-2026)
    if year not in TARGET_YEARS or not ticker or not transcript_text:
        skipped_count += 1
        continue
        
    # Create the ticker folder (e.g., D:/BERT_project/data/raw_extended/earnings_calls/AAPL/)
    ticker_folder = TRANSCRIPT_DIR / ticker
    ticker_folder.mkdir(exist_ok=True)
    
    # Define the output file path
    output_file = ticker_folder / f"{ticker}_{year}_{quarter}.json"
    
    # Save the text in a simple JSON structure
    transcript_data = {
        "symbol": ticker,
        "year": year,
        "quarter": quarter,
        "content": transcript_text
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(transcript_data, f, indent=2)
        
    success_count += 1
    
    # Simple progress tracker for the console
    if success_count % 2500 == 0:
        print(f" -> Progress: Saved {success_count} valid S&P 500 transcripts...")

# ─────────────────────────────────────────────────────────────────────────────
# 3. COMPLETION REPORT
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "-" * 70)
print(" ✅ TRANSCRIPT ACQUISITION REPORT ")
print("-" * 70)
print(f" Successfully Saved to Disk : {success_count} transcripts")
print(f" Skipped (Outside 2019-2026) : {skipped_count}")
print(f" Data secured in : {TRANSCRIPT_DIR.resolve()}")
print("-" * 70)