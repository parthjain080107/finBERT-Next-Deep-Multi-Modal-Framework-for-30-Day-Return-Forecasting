import time
import requests
import re
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path("D:/BERT_project")
FOMC_DIR = BASE_DIR / "data" / "raw_extended" / "fomc"
FOMC_DIR.mkdir(parents=True, exist_ok=True)

FED_BASE_URL = "https://www.federalreserve.gov"

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml',
    'Accept-Language': 'en-US,en;q=0.9',
}

TARGET_YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]

try:
    import lxml
    PARSER = "lxml"
except ImportError:
    PARSER = "html.parser"

if __name__ == "__main__":
    print("=" * 70)
    print("      PHASE 1: FOMC MACROECONOMIC CONTEXT ACQUISITION (2019-2026)    ")
    print("=" * 70)
    print(f"\n1. Scanning Federal Reserve Calendars (Using parser: '{PARSER}')...")
    target_links = []

    for year in TARGET_YEARS:
        if year >= 2021:
            url = f"{FED_BASE_URL}/monetarypolicy/fomccalendars.htm"
        else:
            url = f"{FED_BASE_URL}/monetarypolicy/fomchistorical{year}.htm"
            
        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            
            if r.status_code != 200:
                print(f" ⚠️ Warning: Fed server blocked access for {year} (Status Code: {r.status_code})")
                continue
                
            soup = BeautifulSoup(r.text, PARSER)
            
            # FIXED REGEX: Simply looks for 'monetary' followed by an 8 digit date. 
            # This bypasses any folder changes the Fed makes!
            links = soup.find_all('a', href=re.compile(r'monetary\d{8}[a-z]?\.htm'))
            
            for link in links:
                href = link.get('href')
                full_url = urljoin(FED_BASE_URL, href)
                
                date_match = re.search(r'\d{8}', href)
                if date_match:
                    date_str = date_match.group(0)
                    file_year = int(date_str[:4])
                    if file_year == year and full_url not in [l[0] for l in target_links]:
                        target_links.append((full_url, date_str))
        except Exception as e:
            print(f" -> Error parsing calendar page for {year}: {e}")

    print(f" -> Extracted {len(target_links)} unique statements maps across target window.")

    # ─────────────────────────────────────────────────────────────────────────────
    # 2. DOWNLOAD & CLEAN STATEMENTS
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n2. Scrubbing raw text bodies from Federal Reserve servers...")
    success_count = 0
    fail_count = 0

    for idx, (url, date_str) in enumerate(target_links, 1):
        output_file = FOMC_DIR / f"FOMC_Statement_{date_str}.txt"
        
        if output_file.exists():
            success_count += 1
            continue
            
        try:
            doc_resp = requests.get(url, headers=HEADERS, timeout=10)
            if doc_resp.status_code != 200:
                fail_count += 1
                continue
                
            doc_soup = BeautifulSoup(doc_resp.text, PARSER)
            
            article_div = doc_soup.find('div', id='article') or doc_soup.find('div', id='content')
            
            if not article_div:
                article_div = doc_soup.body
                
            if article_div:
                raw_text = article_div.get_text(separator=' ', strip=True)
                clean_text = re.sub(r'\s+', ' ', raw_text)
                
                if len(clean_text) > 500:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(clean_text)
                    success_count += 1
                else:
                    fail_count += 1
            else:
                fail_count += 1
            
            time.sleep(0.5)
            
            if success_count % 10 == 0:
                print(f" -> Progress: Harvested {success_count} federal records...")
                
        except Exception as e:
            fail_count += 1

    # ─────────────────────────────────────────────────────────────────────────────
    # 3. COMPLETION REPORT
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n" + "-" * 70)
    print(" ✅ FOMC MACROECONOMIC DATA REPORT ")
    print("-" * 70)
    print(f" Successfully Saved : {success_count} documents")
    print(f" Failed/Skipped     : {fail_count} files")
    print(f" Data secured in    : {FOMC_DIR.resolve()}")
    print("-" * 70)