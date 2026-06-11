import os
import json
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

# ─────────────────────────────────────────────────────────────────────────────
# PATH CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path("D:/BERT_project")
CORPUS_FILE = BASE_DIR / "data" / "clean" / "master_training_corpus.json"
TOKENIZER_DIR = BASE_DIR / "finbert_tokenizer"
TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. GENERATE AN EXTRACTED TEXT STREAM FROM MASTER CORPUS (2019-2025 ONLY)
# ─────────────────────────────────────────────────────────────────────────────
def corpus_iterator():
    """Streams data into the tokenizer memory. Strictly filters out 2026 data!"""
    if not CORPUS_FILE.exists():
        raise FileNotFoundError(f"❌ Missing master training corpus file: {CORPUS_FILE}")
        
    with open(CORPUS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    valid_records = 0
    skipped_records = 0
    
    for record in data:
        # CRITICAL CHANGE: Zero Data Leakage. Only train vocab on <= 2025.
        year = int(record.get("year", 0))
        if year <= 2025:
            valid_records += 1
            yield record.get("sec_text", "")
            yield record.get("macro_text", "")
        else:
            skipped_records += 1
            
    print(f" -> Streamed {valid_records} valid historical records.")
    print(f" -> Safely ignored {skipped_records} evaluation records (2026+).")

if __name__ == "__main__":
    print("=" * 70)
    print("     PHASE 2: SEC & MACRO VOCABULARY TOKENIZATION (2019-2025)        ")
    print("=" * 70)
    
    print("\n1. Preparing internal memory streaming iterator...")

    # ─────────────────────────────────────────────────────────────────────────────
    # 2. INTRODUCE RAW UNTRAINED INITIALIZER
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n2. Initializing empty WordPiece model architecture...")
    raw_tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    raw_tokenizer.pre_tokenizer = Whitespace()

    trainer = WordPieceTrainer(
        vocab_size=30522,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    )

    # ─────────────────────────────────────────────────────────────────────────────
    # 3. TRAIN AND WRAP FOR HUGGING FACE COMPATIBILITY
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n3. Launching WordPiece optimization calculations across data streams...")
    print(" -> This may take a moment while it analyzes millions of words...")
    
    raw_tokenizer.train_from_iterator(corpus_iterator(), trainer=trainer)

    print(" -> Saving underlying tokenizer mapping...")
    raw_tokenizer.save(str(TOKENIZER_DIR / "tokenizer.json"))

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(TOKENIZER_DIR / "tokenizer.json"),
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]"
    )
    
    print(" -> Wrapping and saving Hugging Face PreTrainedTokenizerFast object...")
    fast_tokenizer.save_pretrained(str(TOKENIZER_DIR))

    print("\n" + "-" * 70)
    print(" ✅ STAGE 2A COMPLETION REPORT ")
    print("-" * 70)
    print(f" Target Vocabulary Size : {fast_tokenizer.vocab_size} tokens")
    print(f" Data Leakage Status    : Zero (2026 strictly excluded)")
    print(f" Saved Tokenizer Path   : {TOKENIZER_DIR.resolve()}")
    print("-" * 70)