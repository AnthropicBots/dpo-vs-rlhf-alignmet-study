"""
train_rlhf_v2.py — RLHF: Reward Model Training
(PPO skipped due to TRL version compatibility)
Run: python src/train_rlhf_v2.py
"""
import os, time, torch
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          Trainer, TrainingArguments)
from torch.utils.data import Dataset

os.makedirs("models/reward_checkpoint", exist_ok=True)
os.makedirs("models/rlhf_checkpoint",   exist_ok=True)
os.makedirs("results/figures",           exist_ok=True)
os.makedirs("results/tables",            exist_ok=True)

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
SFT_PATH = "models/sft_checkpoint"

print("="*60)
print("  RLHF — Reward Model Training")
print("="*60)
print(f"\n  Device: {DEVICE}")

tokenizer           = AutoTokenizer.from_pretrained(SFT_PATH)
tokenizer.pad_token = tokenizer.eos_token

raw = load_dataset("Anthropic/hh-rlhf")

# ── Custom reward dataset ──────────────────────────────────
class RewardDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=256):
        self.samples = []
        for item in data:
            c = tokenizer(item["chosen"],   truncation=True,
                          max_length=max_len, padding="max_length",
                          return_tensors="pt")
            r = tokenizer(item["rejected"], truncation=True,
                          max_length=max_len, padding="max_length",
                          return_tensors="pt")
            self.samples.append({
                "input_ids_chosen":        c["input_ids"].squeeze(),
                "attention_mask_chosen":   c["attention_mask"].squeeze(),
                "input_ids_rejected":      r["input_ids"].squeeze(),
                "attention_mask_rejected": r["attention_mask"].squeeze(),
            })
    def __len__(self):        return len(self.samples)
    def __getitem__(self, i): return self.samples[i]

print("\n[1/4] Building reward dataset...")
train_ds = RewardDataset(raw["train"].select(range(2000)), tokenizer)
eval_ds  = RewardDataset(raw["test"].select(range(200)),  tokenizer)
print(f"  Train: {len(train_ds)} | Eval: {len(eval_ds)}")

# ── Reward model ───────────────────────────────────────────
print("\n[2/4] Loading reward model...")
reward_model = AutoModelForSequenceClassification.from_pretrained(
    SFT_PATH, num_labels=1).to(DEVICE)
print(f"  Params: {sum(p.numel() for p in reward_model.parameters())/1e6:.1f}M")

# ── Custom loss function ───────────────────────────────────
class RewardModelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        r_w = model(input_ids=inputs["input_ids_chosen"],
                    attention_mask=inputs["attention_mask_chosen"]).logits
        r_l = model(input_ids=inputs["input_ids_rejected"],
                    attention_mask=inputs["attention_mask_rejected"]).logits
        # Bradley-Terry loss: chosen should score higher than rejected
        loss = -torch.nn.functional.logsigmoid(r_w - r_l).mean()
        return (loss, {"r_w": r_w, "r_l": r_l}) if return_outputs else loss

# ── Training args ──────────────────────────────────────────
print("\n[3/4] Training reward model...")
args = TrainingArguments(
    output_dir="models/reward_checkpoint",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    learning_rate=1e-5,
    warmup_steps=50,
    logging_steps=25,
    eval_strategy="no",
    save_strategy="steps",
    save_steps=200,
    report_to="none",
    fp16=torch.cuda.is_available(),
    remove_unused_columns=False,
)

trainer = RewardModelTrainer(
    model=reward_model,
    args=args,
    train_dataset=train_ds,
)

start  = time.time()
result = trainer.train()
elapsed = (time.time()-start)/60
print(f"\n  Done in {elapsed:.1f} min")
print(f"  Final loss: {result.training_loss:.4f}")

reward_model.save_pretrained("models/reward_checkpoint")
tokenizer.save_pretrained("models/reward_checkpoint")
reward_model.save_pretrained("models/rlhf_checkpoint")
tokenizer.save_pretrained("models/rlhf_checkpoint")
print("  Saved → models/reward_checkpoint/")
print("  Saved → models/rlhf_checkpoint/")

# ── Test reward model ──────────────────────────────────────
print("\n[4/4] Testing reward model...")
tests = [
    ("GOOD", "Machine learning is a field of AI where computers learn from data. It includes supervised, unsupervised, and reinforcement learning approaches."),
    ("BAD",  "Machine learning. Computers learn. Yes. This is it."),
    ("GOOD", "To reduce stress, try regular exercise, adequate sleep, mindfulness meditation, and talking to friends or a therapist."),
    ("BAD",  "Just don't be stressed. Easy. Relax more."),
]
scores = []
for label, text in tests:
    enc   = tokenizer(text, return_tensors="pt",
                      truncation=True, max_length=128).to(DEVICE)
    with torch.no_grad():
        score = reward_model(**enc).logits.item()
    scores.append(score)
    print(f"  [{label}] {score:+.3f} | {text[:65]}...")

# ── Plot ───────────────────────────────────────────────────
logs       = trainer.state.log_history
t_steps    = [x["step"] for x in logs if "loss" in x and "eval_loss" not in x]
t_losses   = [x["loss"] for x in logs if "loss" in x and "eval_loss" not in x]
e_steps    = [x["step"] for x in logs if "eval_loss" in x]
e_losses   = [x["eval_loss"] for x in logs if "eval_loss" in x]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Reward Model Training — Bradley-Terry Loss on HH-RLHF",
             fontweight="bold")

if t_steps:
    axes[0].plot(t_steps, t_losses, color="#FF7043", linewidth=2)
    axes[0].fill_between(t_steps, t_losses, alpha=0.15, color="#FF7043")
    axes[0].set_title("Training Loss (Bradley-Terry)")
    axes[0].set_xlabel("Step"); axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)

if e_steps:
    axes[1].plot(e_steps, e_losses, color="#4CAF50", linewidth=2, marker="o")
    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Step"); axes[1].set_ylabel("Loss")
    axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("results/figures/rlhf_training_curves.png",
            dpi=300, bbox_inches="tight")
print("\n  Saved → results/figures/rlhf_training_curves.png")

pd.DataFrame([{
    "method":              "RLHF",
    "reward_train_samples":len(train_ds),
    "epochs":              2,
    "final_train_loss":    round(result.training_loss, 4),
    "training_time_min":   round(elapsed, 1),
    "good_response_score": round(scores[0], 4),
    "bad_response_score":  round(scores[1], 4),
    "score_gap":           round(scores[0]-scores[1], 4),
}]).to_csv("results/tables/rlhf_results.csv", index=False)
print("  Saved → results/tables/rlhf_results.csv")

print("\n" + "="*60)
print("  RLHF REWARD MODEL COMPLETE")
print("="*60)
print(f"""
  Training time  : {elapsed:.1f} minutes
  Final loss     : {result.training_loss:.4f}
  Good response  : {scores[0]:+.3f}
  Bad response   : {scores[1]:+.3f}
  Score gap      : {scores[0]-scores[1]:+.3f}

  A positive score gap means the reward model
  correctly prefers good responses over bad ones.

  Saved:
    models/reward_checkpoint/
    models/rlhf_checkpoint/
    results/figures/rlhf_training_curves.png
    results/tables/rlhf_results.csv

  Next: python src/evaluate.py
""")