import json
from pathlib import Path
from transformers import PreTrainedTokenizerFast
from datasets import Dataset

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path("D:/BERT_project")
DATA_DIR = BASE_DIR / "data"

INPUT_FILE = DATA_DIR / "clean" / "master_training_corpus.json"
OUTPUT_DIR = DATA_DIR / "arrow_dataset"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOKENIZER_DIR = BASE_DIR / "finbert_tokenizer"
MAX_LEN = 512  # Must exactly match minibert.py max_position_embeddings

if __name__ == "__main__":
    print("=" * 70)
    print("   PHASE 2: MULTI-MODAL TENSOR PACKING (STRICTLY 2019-2025)   ")
    print("=" * 70)

    # ─────────────────────────────────────────────────────────────────────────────
    # 1. LOAD TOKENIZER
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n1. Loading Custom Financial Tokenizer...")
    try:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(str(TOKENIZER_DIR))
        PAD_TOKEN_ID = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    except Exception as e:
        print(f"❌ Failed to load tokenizer. Did you run stage2A? Error: {e}")
        exit()

    # ─────────────────────────────────────────────────────────────────────────────
    # 2. LOAD AND STRICTLY FILTER MASTER CORPUS
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n2. Loading Master Training Corpus & Applying 2025 Firewall...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        master_data = json.load(f)

    training_records = []
    skipped_records = 0

    for row in master_data:
        # CRITICAL: Hard-coded firewall. 2026 data is completely isolated.
        year = int(row.get("year", 0))
        if year <= 2025:
            training_records.append(row)
        else:
            skipped_records += 1

    print(f" -> Retained {len(training_records)} valid historical records for pre-training.")
    print(f" -> Blocked {skipped_records} future records (2026+) to prevent data leakage.")

    if not training_records:
        print("❌ Error: No valid training records found <= 2025!")
        exit()

    # ─────────────────────────────────────────────────────────────────────────────
    # 3. TOKENIZATION & TENSOR PACKING
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n3. Tokenizing text and packing 5-D Tabular Multi-Modal chunks...")
    packed_samples = []
    
    for idx, row in enumerate(training_records):
        sec_text = row.get("sec_text", "")
        macro_text = row.get("macro_text", "")
        
        # Pulling the exact 5-dimensional math array calculated in data_fusion.py
        tabular_vector = row.get("tabular_vector", [0.0]*5) 
        
        # We append SEC and MACRO text together so the model learns cross-context
        combined_text = sec_text + " " + macro_text
        
        # Tokenize sequence 
        tokens = tokenizer.encode(combined_text, add_special_tokens=True)
        
        # Slice into chunks of 512
        for i in range(0, len(tokens), MAX_LEN):
            chunk = tokens[i : i + MAX_LEN]
            
            # If the chunk is shorter than 512, pad it perfectly
            if len(chunk) < MAX_LEN:
                padding_needed = MAX_LEN - len(chunk)
                attention_mask = [1] * len(chunk) + [0] * padding_needed
                chunk = chunk + [PAD_TOKEN_ID] * padding_needed
            else:
                attention_mask = [1] * MAX_LEN
                
            packed_samples.append({
                "input_ids": chunk,
                "attention_mask": attention_mask,
                "tabular_features": tabular_vector,
                "ctmm_labels": 1  # 1 indicates a true aligned matching pair (Text <-> Market)
            })
            
        if (idx + 1) % 500 == 0:
            print(f" -> Processed text constraints for {idx + 1}/{len(training_records)} records...")

    print(f" -> Successfully sliced texts into {len(packed_samples)} total {MAX_LEN}-token chunks.")

    # ─────────────────────────────────────────────────────────────────────────────
    # 4. SAVE TO HIGH-SPEED ARROW DATASET
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n4. Converting to High-Speed Arrow Format for GPU Loader...")
    arrow_dataset = Dataset.from_list(packed_samples)
    arrow_dataset.save_to_disk(str(OUTPUT_DIR))

    print("\n" + "-" * 70)
    print(" ✅ STAGE 2B TENSOR PACKING REPORT ")
    print("-" * 70)
    print(f" Total Pre-Training Chunks : {len(packed_samples)}")
    print(f" Tabular Vector Dimension  : {len(packed_samples[0]['tabular_features'])} features")
    print(f" Evaluation Leakage        : 0 (Strict 2025 cutoff enforced)")
    print(f" Saved Tensor Directory    : {OUTPUT_DIR.resolve()}")
    print("-" * 70)