import torch
import evaluate
import numpy as np
from pathlib import Path
from datasets import load_from_disk
from transformers import TrainingArguments, Trainer

from minibert import MINI_BERT_CONFIG, MultiModalMiniBERTForClassification

# ─────────────────────────────────────────────────────────────────────────────
# PATH CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path("D:/BERT_project")
CLASSIFICATION_ARROW_DIR = BASE_DIR / "data" / "arrow_multimodal_classification"
PRETRAINED_WEIGHTS = BASE_DIR / "multimodal_finbert_model" / "multimodal_minibert_final.pt"
FINAL_QUANT_MODEL_DIR = BASE_DIR / "finbert_multimodal_trader"
FINAL_QUANT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

NUM_TABULAR_FEATURES = 5

# ─────────────────────────────────────────────────────────────────────────────
# 1. THE HUGGING FACE BRIDGE
# ─────────────────────────────────────────────────────────────────────────────
class HFCompatibleMultiModalClassifier(MultiModalMiniBERTForClassification):
    def forward(self, input_ids, tabular_features, attention_mask=None, labels=None, **kwargs):
        # Explicitly routing parameters safely to your core classification architecture
        return super().forward(
            input_ids=input_ids,
            tabular_features=tabular_features,
            attention_mask=attention_mask,
            segment_ids=None,
            labels=labels
        )

# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD DATASET & METRICS
# ─────────────────────────────────────────────────────────────────────────────
print("1. Loading custom sharded Arrow classification data matrix...")
full_dataset = load_from_disk(str(CLASSIFICATION_ARROW_DIR))

split_dataset = full_dataset.train_test_split(test_size=0.15, seed=42)
train_data = split_dataset["train"]
val_data = split_dataset["test"]

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def classification_collator(examples):
    batch = {}
    batch["input_ids"] = torch.tensor([ex["input_ids"] for ex in examples], dtype=torch.long)
    batch["attention_mask"] = torch.tensor([ex["attention_mask"] for ex in examples], dtype=torch.long)
    batch["tabular_features"] = torch.tensor([ex["tabular_features"] for ex in examples], dtype=torch.float32)
    batch["labels"] = torch.tensor([ex["labels"] for ex in examples], dtype=torch.long)
    return batch

# ─────────────────────────────────────────────────────────────────────────────
# 3. CONSTRUCT MODEL & LINK WEIGHT ARRAYS
# ─────────────────────────────────────────────────────────────────────────────
print("\n2. Building MultiModal Classification Neural Network Architecture...")
model = HFCompatibleMultiModalClassifier(MINI_BERT_CONFIG, NUM_TABULAR_FEATURES, num_labels=3)

if PRETRAINED_WEIGHTS.exists():
    print(" -> Found Pretrained Weights. Injecting Stage 3 base state parameters...")
    state_dict = torch.load(str(PRETRAINED_WEIGHTS))
    model.load_state_dict(state_dict, strict=False)
    print(" -> SUCCESS: Stage 3 Pretrained Weights injected.")
else:
    print(" -> [WARNING] Stage 3 weights not found. Training completely from scratch.")

# ─────────────────────────────────────────────────────────────────────────────
# 4. CONFIGURE & LAUNCH FINE-TUNING LOOP
# ─────────────────────────────────────────────────────────────────────────────
print("\n3. Configuring Fine-Tuning Optimizer...")
training_args = TrainingArguments(
    output_dir=str(FINAL_QUANT_MODEL_DIR / "checkpoints"),
    num_train_epochs=4,               
    
    # ⚡ Set to 16 for optimized velocity (independent of phase 3)
    per_device_train_batch_size=16,     
    per_device_eval_batch_size=16,     
    
    learning_rate=2e-5,  
    weight_decay=0.01,
    evaluation_strategy="epoch",  # ✨ FIXED: Changed from eval_strategy to support older package versions
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=torch.cuda.is_available(),
    remove_unused_columns=False,

    # ⚙️ Standard non-fused AdamW optimizer
    optim="adamw_torch",        
    dataloader_pin_memory=True,       

    # 🛡️ THE CRASH SHIELD: Prevents 0-byte folder saving crashes
    save_safetensors=False             
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    data_collator=classification_collator,
    compute_metrics=compute_metrics,
)

print("\n🚀 LAUNCHING MULTI-MODAL ALGORITHMIC FINE-TUNING LOOP...")
trainer.train()

print("\n✅ Training complete. Saving optimized trading parameters...")
torch.save(model.state_dict(), str(FINAL_QUANT_MODEL_DIR / "multimodal_quant_final.pt"))
print(f"Optimized model binary successfully exported to: {FINAL_QUANT_MODEL_DIR.resolve()}")