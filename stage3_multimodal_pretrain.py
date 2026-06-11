import torch
from pathlib import Path
from datasets import load_from_disk
from transformers import (
    PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)

from minibert import MINI_BERT_CONFIG, MultiModalMiniBERTForPretraining

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION & TOGGLES
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path("D:/BERT_project")
TOKENIZER_DIR = BASE_DIR / "finbert_tokenizer"
ARROW_DIR = BASE_DIR / "data" / "arrow_dataset"
MODEL_OUTPUT_DIR = BASE_DIR / "multimodal_finbert_model"
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MICRO_RUN_MODE = False
NUM_TABULAR_FEATURES = 5  

# ─────────────────────────────────────────────────────────────────────────────
# 1. THE MULTI-MODAL HUGGING FACE BRIDGE
# ─────────────────────────────────────────────────────────────────────────────
class HFCompatibleMultiModalMiniBERT(MultiModalMiniBERTForPretraining):
    def forward(self, input_ids, tabular_features, attention_mask=None, labels=None, ctmm_labels=None, **kwargs):
        # Hugging Face passes text targets as 'labels' 
        # We route it directly into your architecture's 'mlm_labels'
        return super().forward(
            input_ids=input_ids,
            tabular_features=tabular_features,
            attention_mask=attention_mask,
            segment_ids=None,
            mlm_labels=labels,  
            ctmm_labels=ctmm_labels
        )

# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA COLLATOR FOR SMLM & CROSS-MODAL MATCHING (CTMM)
# ─────────────────────────────────────────────────────────────────────────────
class MultiModalDataCollator(DataCollatorForLanguageModeling):
    def __call__(self, examples):
        tabular_vectors = [ex.pop("tabular_features") for ex in examples]
        batch = super().__call__(examples)
        
        batch_size = len(tabular_vectors)
        tabular_tensor = torch.tensor(tabular_vectors, dtype=torch.float32)
        
        ctmm_labels = torch.ones(batch_size, dtype=torch.long)
        
        num_negatives = batch_size // 2
        if num_negatives > 0:
            indices = torch.randperm(batch_size)[:num_negatives]
            shuffle_indices = torch.randperm(num_negatives)
            tabular_tensor[indices] = tabular_tensor[indices][shuffle_indices]
            ctmm_labels[indices] = 0
            
        batch["tabular_features"] = tabular_tensor
        batch["ctmm_labels"] = ctmm_labels
        
        return batch

# ─────────────────────────────────────────────────────────────────────────────
# 3. INITIALIZE MODEL & DATASETS
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("      PHASE 3: MULTI-MODAL SELF-SUPERVISED PRE-TRAINING LOOP         ")
print("=" * 70)

device_name = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
print(f"\n[SYSTEM CHECK] Hardware Accelerator Target: {device_name}")

print("\n1. Streaming binary tensors via cross-modal memory mapping...")
dataset = load_from_disk(str(ARROW_DIR))
tokenizer = PreTrainedTokenizerFast.from_pretrained(str(TOKENIZER_DIR))

print("2. Instantiating custom architecture and model configuration mapping...")

# 🏎️ CRITICAL FIX: We are now using the HFCompatible wrapper here!
model = HFCompatibleMultiModalMiniBERT(MINI_BERT_CONFIG, NUM_TABULAR_FEATURES)

model.tie_weights()
print(" -> Embeddings and MLM weights successfully tied.")

data_collator = MultiModalDataCollator(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# ─────────────────────────────────────────────────────────────────────────────
# 4. CHOOSE TRAIN SETTING BASED ON MODE
# ─────────────────────────────────────────────────────────────────────────────
if MICRO_RUN_MODE:
    print("\n⚠️ MICRO_RUN_MODE IS ACTIVATED.")
    dataset = dataset.select(range(min(64, len(dataset))))
    
    train_args = TrainingArguments(
        output_dir=str(MODEL_OUTPUT_DIR / "debug_runs"),
        max_steps=5,                      
        per_device_train_batch_size=4,    
        logging_steps=1,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        save_safetensors=False
    )
else:
    print(f"\n🚀 PROCEEDING WITH PRODUCTION MODE RUN OVER ALL {len(dataset)} MASTER TOKENS.")
    train_args = TrainingArguments(
        output_dir=str(MODEL_OUTPUT_DIR / "checkpoints"),
        num_train_epochs=2,               
        
        # ⚡ OPTIMIZED STEP BALANCING (EFFECTIVE BATCH SIZE = 256)
        per_device_train_batch_size=8,     
        gradient_accumulation_steps=32,    
        
        learning_rate=1e-4,               
        weight_decay=0.01,                
        warmup_steps=10000,               
        save_steps=5000,                  
        save_total_limit=2,               
        logging_steps=100,                
        fp16=torch.cuda.is_available(),    
        remove_unused_columns=False,

        # 🏎️ Pin memory to system RAM for streamlined GPU throughput
        dataloader_pin_memory=True,       
        
        # 🛡️ Weight-tying Save Crash Fix
        save_safetensors=False             
    )

# ─────────────────────────────────────────────────────────────────────────────
# 5. EXECUTION BLOCK
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    print("\n🚀 LAUNCHING MULTI-MODAL PRETRAINING...")
    trainer.train()

    if not MICRO_RUN_MODE:
        print("\n" + "-" * 70)
        print(" ✅ PRETRAINING COMPLETE! SAVING FINAL MULTI-MODAL WEIGHTS ")
        print("-" * 70)
        torch.save(model.state_dict(), str(MODEL_OUTPUT_DIR / "multimodal_minibert_final.pt"))
        print(f" Model securely saved to: {MODEL_OUTPUT_DIR.resolve()}")