"""
train_sft.py  —  Supervised Fine-Tuning baseline
Run: python src/train_sft.py
Trains GPT-2 on chosen responses from HH-RLHF.
This is the shared starting point for both DPO and RLHF.
"""
import os, time, torch
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          TrainingArguments, Trainer, DataCollatorForLanguageModeling)
from torch.utils.data import Dataset

os.makedirs("results/figures", exist_ok=True)
os.makedirs("models/sft_checkpoint", exist_ok=True)

print("="*60)
print("  STEP 1: Supervised Fine-Tuning (SFT)")
print("="*60)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "gpt2"
MAX_LENGTH = 256
BATCH_SIZE = 8
EPOCHS = 3
LR = 2e-5

print(f"\n  Device : {DEVICE}")
print(f"  Model  : {MODEL_NAME}")
print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory//1024**2} MB")

# ── Load tokenizer
print("\n[1/5] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

#Load dataset 
print("[2/5] Loading dataset...")
dataset = load_dataset("Anthropic/hh-rlhf")
train_data = dataset["train"].select(range(5000))
val_data   = dataset["test"].select(range(500))
print(f"  Train: {len(train_data)} | Val: {len(val_data)}")

# Tokenise 
class SFTDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.samples = []
        for item in data:
            enc = tokenizer(item["chosen"], truncation=True,
                            max_length=max_len, padding="max_length",
                            return_tensors="pt")
            self.samples.append({
                "input_ids":      enc["input_ids"].squeeze(),
                "attention_mask": enc["attention_mask"].squeeze(),
                "labels":         enc["input_ids"].squeeze().clone(),
            })
    def __len__(self):  return len(self.samples)
    def __getitem__(self, i): return self.samples[i]

print("[3/5] Tokenising...")
train_ds = SFTDataset(train_data, tokenizer, MAX_LENGTH)
val_ds   = SFTDataset(val_data,   tokenizer, MAX_LENGTH)
print(f"  Done. Train tokens: {len(train_ds)*MAX_LENGTH:,}")

#  Model 
print("[4/5] Loading model...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
params = sum(p.numel() for p in model.parameters())/1e6
print(f"  Parameters: {params:.1f}M")

#  Training 
print("[5/5] Training SFT model...")
args = TrainingArguments(
    output_dir="models/sft_checkpoint",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LR,
    warmup_steps=100,
    weight_decay=0.01,
    logging_steps=50,
    eval_steps=200,
    save_steps=500,
    load_best_model_at_end=True,
    report_to="none",
    fp16=torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
)

start = time.time()
result = trainer.train()
elapsed = (time.time()-start)/60

print(f"\n  Training done in {elapsed:.1f} minutes")
print(f"  Final train loss: {result.training_loss:.4f}")

#  Save model 
model.save_pretrained("models/sft_checkpoint")
tokenizer.save_pretrained("models/sft_checkpoint")
print("  Model saved → models/sft_checkpoint/")

#  Plot loss curve
train_loss = [(x["step"], x["loss"]) for x in logs if "loss" in x]
eval_loss  = [(x["step"], x["eval_loss"]) for x in logs if "eval_loss" in x]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("SFT Training — GPT-2 on HH-RLHF", fontweight="bold")

if train_loss:
    steps, losses = zip(*train_loss)
    axes[0].plot(steps, losses, color="#2196F3", linewidth=2)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Step"); axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)

if eval_loss:
    steps, losses = zip(*eval_loss)
    axes[1].plot(steps, losses, color="#4CAF50", linewidth=2, marker="o")
    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Step"); axes[1].set_ylabel("Loss")
    axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("results/figures/sft_training_curves.png", dpi=300, bbox_inches="tight")
print("  Graph saved → results/figures/sft_training_curves.png")

#  Save stats
pd.DataFrame([{
    "model": MODEL_NAME, "method": "SFT",
    "train_samples": len(train_ds), "epochs": EPOCHS,
    "final_train_loss": round(result.training_loss, 4),
    "training_time_min": round(elapsed, 1),
}]).to_csv("results/tables/sft_results.csv", index=False)
print("  Stats saved → results/tables/sft_results.csv")

#  Quick generation test 
print("\n  Generation test:")
prompt = "\n\nHuman: What is machine learning?\n\nAssistant:"
inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=60,
                         do_sample=True, temperature=0.7, pad_token_id=tokenizer.eos_token_id)
print("  " + tokenizer.decode(out[0], skip_special_tokens=True)[len(prompt):].strip()[:200])

print("\n" + "="*60)
print("  SFT COMPLETE — checkpoint saved")
print("  Next: python src/train_dpo.py")
print("="*60)