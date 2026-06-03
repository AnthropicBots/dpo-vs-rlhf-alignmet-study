"""
train_dpo_v2.py — Direct Preference Optimisation (improved)
Run: python src/train_dpo_v2.py
Requires: models/sft_checkpoint must exist (run train_sft.py first)
"""
import os, time, torch
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig

os.makedirs("models/dpo_checkpoint", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)

print("="*60)
print("  STEP 2: Direct Preference Optimisation (DPO)")
print("="*60)

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
SFT_PATH   = "models/sft_checkpoint"
MAX_LENGTH = 256
BETA       = 0.1
EPOCHS     = 1
BATCH_SIZE = 4
LR         = 1e-5

print(f"\n  Device : {DEVICE}")
print(f"  Beta   : {BETA}")
print(f"  LR     : {LR}")

# Load tokenizer + models
print("\n[1/5] Loading SFT checkpoint...")
tokenizer = AutoTokenizer.from_pretrained(SFT_PATH)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "left"

model     = AutoModelForCausalLM.from_pretrained(SFT_PATH).to(DEVICE)
ref_model = AutoModelForCausalLM.from_pretrained(SFT_PATH).to(DEVICE)
print("  Policy model    : loaded")
print("  Reference model : loaded (frozen SFT copy)")

#Dataset
print("\n[2/5] Preparing preference dataset...")
raw = load_dataset("Anthropic/hh-rlhf")

def format_pair(sample):
    parts         = sample["chosen"].split("\n\nAssistant:")
    prompt        = parts[0] + "\n\nAssistant:" if len(parts) > 1 else ""
    chosen_resp   = parts[-1].strip() if len(parts) > 1 else sample["chosen"]
    rejected_resp = sample["rejected"].split("\n\nAssistant:")[-1].strip()
    return {"prompt": prompt, "chosen": chosen_resp, "rejected": rejected_resp}

train_ds = raw["train"].select(range(3000)).map(
    format_pair, remove_columns=raw["train"].column_names)
eval_ds  = raw["test"].select(range(300)).map(
    format_pair, remove_columns=raw["test"].column_names)

print(f"  Train pairs : {len(train_ds)}")
print(f"  Eval pairs  : {len(eval_ds)}")

#DPO Config
print("\n[3/5] Configuring DPO trainer...")
config = DPOConfig(
    output_dir="models/dpo_checkpoint",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LR,
    beta=BETA,
    max_length=MAX_LENGTH,
    logging_steps=25,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=200,
    load_best_model_at_end=True,
    report_to="none",
    fp16=torch.cuda.is_available(),
    remove_unused_columns=False,
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=config,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tokenizer,
)

#Train
print("\n[4/5] Training DPO model...")
print("  Watch for:")
print("  loss decreasing from ~0.693")
print("  rewards/chosen  going UP")
print("  rewards/rejected going DOWN")
print("  rewards/margins  widening\n")

start  = time.time()
result = trainer.train()
elapsed = (time.time()-start)/60

print(f"\n  Done in {elapsed:.1f} minutes")
print(f"  Final loss: {result.training_loss:.4f}")

model.save_pretrained("models/dpo_checkpoint")
tokenizer.save_pretrained("models/dpo_checkpoint")
print("  Saved → models/dpo_checkpoint/")

#Save graphs
print("\n[5/5] Saving graphs and stats...")
logs         = trainer.state.log_history
steps        = [x["step"] for x in logs if "loss" in x and "eval_loss" not in x]
losses       = [x["loss"] for x in logs if "loss" in x and "eval_loss" not in x]
r_chosen     = [x["rewards/chosen"]   for x in logs if "rewards/chosen"   in x]
r_rejected   = [x["rewards/rejected"] for x in logs if "rewards/rejected" in x]
r_margin     = [x["rewards/margins"]  for x in logs if "rewards/margins"  in x]
reward_steps = [x["step"]             for x in logs if "rewards/chosen"   in x]

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle("DPO Training — GPT-2 on HH-RLHF (3000 pairs, beta=0.1)",
             fontweight="bold")

if steps:
    axes[0].plot(steps, losses, color="#7E57C2", linewidth=2)
    axes[0].fill_between(steps, losses, alpha=0.15, color="#7E57C2")
    axes[0].set_title("DPO Loss"); axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss");    axes[0].grid(alpha=0.3)

if r_chosen:
    axes[1].plot(reward_steps, r_chosen,   color="#4CAF50", label="Chosen",   linewidth=2)
    axes[1].plot(reward_steps, r_rejected, color="#F44336", label="Rejected", linewidth=2)
    axes[1].axhline(0, color="gray", linestyle="--", alpha=0.4)
    axes[1].set_title("Reward Scores"); axes[1].set_xlabel("Step")
    axes[1].legend(); axes[1].grid(alpha=0.3)

if r_margin:
    axes[2].plot(reward_steps, r_margin, color="#FF9800", linewidth=2)
    axes[2].fill_between(reward_steps, r_margin, alpha=0.15, color="#FF9800")
    axes[2].axhline(0, color="red", linestyle="--", alpha=0.4)
    axes[2].set_title("Reward Margin (chosen − rejected)")
    axes[2].set_xlabel("Step"); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("results/figures/dpo_training_curves.png", dpi=300, bbox_inches="tight")
print("  Saved → results/figures/dpo_training_curves.png")

pd.DataFrame([{
    "method":            "DPO",
    "beta":              BETA,
    "train_pairs":       len(train_ds),
    "eval_pairs":        len(eval_ds),
    "epochs":            EPOCHS,
    "final_loss":        round(result.training_loss, 4),
    "training_time_min": round(elapsed, 1),
    "avg_reward_margin": round(sum(r_margin[-5:])/5, 4) if r_margin else 0,
}]).to_csv("results/tables/dpo_results.csv", index=False)
print("  Saved → results/tables/dpo_results.csv")

# Generation test
print("\n  Generation test (DPO aligned model):")
for prompt in ["\n\nHuman: What is machine learning?\n\nAssistant:",
               "\n\nHuman: How do I reduce stress?\n\nAssistant:"]:
    enc = tokenizer(prompt, return_tensors="pt",
                    truncation=True, max_length=150).to(DEVICE)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=60, do_sample=True,
                             temperature=0.7,
                             pad_token_id=tokenizer.eos_token_id)
    resp = tokenizer.decode(out[0][enc["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()
    q = prompt.split("Human:")[1].split("Assistant:")[0].strip()
    print(f"\n  Q: {q}")
    print(f"  A: {resp[:150]}")

print("\n" + "="*60)
print("  DPO COMPLETE — next: python src/train_rlhf_v2.py")
print("="*60)
