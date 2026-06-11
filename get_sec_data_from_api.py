import time
import json
import requests
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION & SEC COMPLIANCE
# ─────────────────────────────────────────────────────────────────────────────
TEST_DIR = Path(__file__).resolve().parent
ROOT_DIR = TEST_DIR.parent

# Re-routed to look inside your main data architecture as requested
RAW_TEXT_DIR = ROOT_DIR / "data" / "raw_sec_json"
RAW_TEXT_DIR.mkdir(parents=True, exist_ok=True)

# Live S&P 500 constituents endpoint
SP500_CSV_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"

# SEC compliance headers
SEC_HEADERS = {
    "User-Agent": "IIT Roorkee om_cc@me.iitr.ac.in",
    "Accept-Encoding": "gzip, deflate"
}

# 0.15s delay enforces safe request traffic patterns (~7 requests/sec)
RATE_LIMIT_DELAY = 0.15 

def get_sp500_tickers():
    """Streams the live S&P 500 catalog and returns a list of cleaned ticker symbols."""
    print(f"🌐 Fetching live S&P 500 constituents from GitHub registry...")
    try:
        response = requests.get(SP500_CSV_URL)
        if response.status_code != 200:
            print("❌ Failed to fetch S&P 500 index details from source repository.")
            return []
        
        lines = response.text.strip().split('\n')
        if not lines:
            return []
            
        header = [col.strip().strip('"') for col in lines[0].split(',')]
        try:
            symbol_idx = header.index('Symbol')
        except ValueError:
            symbol_idx = 0 
            
        tickers = []
        for line in lines[1:]:
            cols = [c.strip().strip('"') for c in line.split(',')]
            if len(cols) > symbol_idx:
                ticker = cols[symbol_idx].upper().replace('.', '-')
                if ticker:
                    tickers.append(ticker)
                    
        print(f"📊 Successfully loaded {len(tickers)} S&P 500 companies into memory pipeline.")
        return tickers
    except Exception as e:
        print(f"❌ Error downloading or processing S&P 500 list: {e}")
        return []

def get_company_cik(ticker):
    """Maps ticker string to official SEC Central Index Key (CIK)."""
    url = "https://www.sec.gov/files/company_tickers.json"
    try:
        response = requests.get(url, headers=SEC_HEADERS)
        time.sleep(RATE_LIMIT_DELAY)
        
        if response.status_code != 200:
            url = "https://data.sec.gov/files/company_tickers.json"
            response = requests.get(url, headers=SEC_HEADERS)
            time.sleep(RATE_LIMIT_DELAY)

        if response.status_code == 200:
            data = response.json()
            for item in data.values():
                if str(item['ticker']).upper() == ticker.upper():
                    return str(item['cik_str']).zfill(10)
    except Exception as e:
        pass
    return None

def fetch_and_cache_historical_data(ticker, form_type="10-K", start_year=2019, end_year=2025):
    """Queries SEC archives for historical forms and caches text chunks inside data/raw_sec_json."""
    cik = get_company_cik(ticker)
    if not cik:
        return

    submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    try:
        resp = requests.get(submissions_url, headers=SEC_HEADERS)
        time.sleep(RATE_LIMIT_DELAY)
        if resp.status_code != 200:
            return
            
        history = resp.json()["filings"]["recent"]
        extracted_filings = []
        
        for i in range(len(history["accessionNumber"])):
            f_type = history["form"][i]
            doc_date = history["filingDate"][i]
            year = int(doc_date.split("-")[0])
            
            # Enforce strict historical parameters range (2019 - 2025)
            if f_type == form_type and start_year <= year <= end_year:
                accession = history["accessionNumber"][i].replace("-", "")
                doc_name = history["primaryDocument"][i]
                archive_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{doc_name}"
                
                extracted_filings.append({
                    "ticker": ticker,
                    "form": form_type,
                    "date": doc_date,
                    "year": year,
                    "url": archive_url
                })

        if not extracted_filings:
            return

        print(f" ✅ [{ticker}] Found {len(extracted_filings)} filings matching {start_year}-{end_year}. Downloading...")

        for filing in extracted_filings:
            doc_resp = requests.get(filing["url"], headers=SEC_HEADERS)
            time.sleep(RATE_LIMIT_DELAY)
            
            if doc_resp.status_code == 200:
                # Capture the first chunk of structural document text for tokenization passing
                filing["sec_text"] = doc_resp.text[:500000]  
            else:
                filing["sec_text"] = ""

        output_file = RAW_TEXT_DIR / f"{ticker.lower()}_{form_type.lower()}_{start_year}_{end_year}_meta.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(extracted_filings, f, indent=4)

    except Exception as e:
        print(f"⚠️ Failed tracking records for {ticker}: {e}")

if __name__ == "__main__":
    print("=" * 70)
    print("   S&P 500 HISTORICAL INDEX LOOP (2019 - 2025 DATA ACQUISITION)   ")
    print("=" * 70)
    
    # 1. Download full constituent roster dynamically
    sp500_tickers = get_sp500_tickers()
    
    if not sp500_tickers:
        print("❌ Pipeline execution aborted. Ticker index could not be retrieved.")
        exit(1)
        
    print("\n2. Processing Batch Historical Requests Across SEC API Nodes...")
    start_time = time.time()
    
    for idx, ticker in enumerate(sp500_tickers, 1):
        if idx % 25 == 0:
            print(f" -> Progress Checkpoint: Processed {idx}/{len(sp500_tickers)} companies...")
        
        # Pulls historical 10-K files from 2019 through 2025
        fetch_and_cache_historical_data(ticker, form_type="10-K", start_year=2019, end_year=2025)
        
    duration = time.time() - start_time
    print("-" * 70)
    print(" ✅ HISTORICAL DATA HARVEST COMPLETE!")
    print(f" Total Execution Runtime   : {duration/60:.2f} minutes")
    print(f" Saved Data File Warehouse : {RAW_TEXT_DIR.resolve()}")
    print("-" * 70)