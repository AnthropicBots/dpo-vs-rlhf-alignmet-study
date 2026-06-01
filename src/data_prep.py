"""
data_prep.py
============
Loads and explores the Anthropic HH-RLHF dataset.
This is the dataset we use for both RLHF and DPO training.
Run: python src/data_prep.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from datasets import load_dataset

#Create output folders
os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)
os.makedirs("data", exist_ok=True)

print("=" * 60)
print("DPO vs RLHF Study — Data Exploration")
print("=" * 60)

#Step 1: Load dataset 
print("\n[1/6] Loading Anthropic HH-RLHF dataset...")
dataset = load_dataset("Anthropic/hh-rlhf")
print(f"  Train size : {len(dataset['train']):,}")
print(f"  Test size  : {len(dataset['test']):,}")

#  Step 2: Peek at the data 
print("\n[2/6] Sample entry:")
sample = dataset['train'][0]
print(f"\n  CHOSEN (preferred response):\n  {sample['chosen'][:300]}...")
print(f"\n  REJECTED (dispreferred response):\n  {sample['rejected'][:300]}...")

#  Step 3: Compute statistics 
print("\n[3/6] Computing statistics (this takes ~30 seconds)...")

N = 10000   # analyse first 10k samples for speed
chosen_lens   = [len(x['chosen'].split())   for x in dataset['train'].select(range(N))]
rejected_lens = [len(x['rejected'].split()) for x in dataset['train'].select(range(N))]

stats = {
    "Metric": [
        "Total Train Samples", "Total Test Samples",
        "Avg Chosen Length (words)", "Avg Rejected Length (words)",
        "Max Chosen Length (words)", "Max Rejected Length (words)",
        "Min Chosen Length (words)", "Min Rejected Length (words)",
    ],
    "Value": [
        f"{len(dataset['train']):,}",
        f"{len(dataset['test']):,}",
        f"{sum(chosen_lens)/len(chosen_lens):.1f}",
        f"{sum(rejected_lens)/len(rejected_lens):.1f}",
        f"{max(chosen_lens)}",
        f"{max(rejected_lens)}",
        f"{min(chosen_lens)}",
        f"{min(rejected_lens)}",
    ]
}

df_stats = pd.DataFrame(stats)
print("\n" + df_stats.to_string(index=False))
df_stats.to_csv("results/tables/dataset_statistics.csv", index=False)
print("\n  Saved → results/tables/dataset_statistics.csv")

#  Step 4: Plot length distributions 
print("\n[4/6] Plotting length distributions...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Anthropic HH-RLHF Dataset Analysis\n(DPO vs RLHF Study)",
             fontsize=14, fontweight='bold', y=1.02)

# Plot 1 — Chosen vs Rejected length histogram
axes[0].hist(chosen_lens,   bins=60, alpha=0.7, color='#2196F3', label='Chosen')
axes[0].hist(rejected_lens, bins=60, alpha=0.7, color='#F44336', label='Rejected')
axes[0].set_title('Response Length Distribution', fontweight='bold')
axes[0].set_xlabel('Word Count')
axes[0].set_ylabel('Frequency')
axes[0].legend()
axes[0].axvline(sum(chosen_lens)/len(chosen_lens),
                color='#1565C0', linestyle='--', linewidth=2,
                label=f'Chosen avg: {sum(chosen_lens)/len(chosen_lens):.0f}')
axes[0].axvline(sum(rejected_lens)/len(rejected_lens),
                color='#B71C1C', linestyle='--', linewidth=2,
                label=f'Rejected avg: {sum(rejected_lens)/len(rejected_lens):.0f}')
axes[0].legend(fontsize=8)

# Plot 2 — Box plot comparison
data_box = [chosen_lens, rejected_lens]
bp = axes[1].boxplot(data_box, patch_artist=True, notch=True,
                      labels=['Chosen', 'Rejected'])
bp['boxes'][0].set_facecolor('#2196F3')
bp['boxes'][1].set_facecolor('#F44336')
axes[1].set_title('Length Box Plot Comparison', fontweight='bold')
axes[1].set_ylabel('Word Count')
axes[1].set_xlabel('Response Type')

# Plot 3 — Difference: chosen_len - rejected_len
diffs = [c - r for c, r in zip(chosen_lens, rejected_lens)]
axes[2].hist(diffs, bins=60, color='#4CAF50', alpha=0.8, edgecolor='white')
axes[2].axvline(0, color='black', linestyle='-', linewidth=1.5)
axes[2].axvline(sum(diffs)/len(diffs), color='#1B5E20',
                linestyle='--', linewidth=2,
                label=f'Mean diff: {sum(diffs)/len(diffs):.1f}')
axes[2].set_title('Chosen − Rejected Length Difference', fontweight='bold')
axes[2].set_xlabel('Word Count Difference')
axes[2].set_ylabel('Frequency')
axes[2].legend()

plt.tight_layout()
plt.savefig("results/figures/dataset_length_analysis.png",
            dpi=300, bbox_inches='tight')
plt.show()
print("  Saved → results/figures/dataset_length_analysis.png")

#  Step 5: Conversation depth analysis 
print("\n[5/6] Analysing conversation structure...")

human_turns   = []
assistant_turns = []

for item in dataset['train'].select(range(N)):
    text = item['chosen']
    human_turns.append(text.count('\n\nHuman:'))
    assistant_turns.append(text.count('\n\nAssistant:'))

fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
fig2.suptitle("Conversation Structure Analysis", fontweight='bold')

axes2[0].hist(human_turns, bins=range(1, 12), color='#9C27B0',
              alpha=0.8, edgecolor='white', align='left')
axes2[0].set_title('Human Turns per Conversation')
axes2[0].set_xlabel('Number of Turns')
axes2[0].set_ylabel('Count')
axes2[0].set_xticks(range(1, 11))

axes2[1].hist(assistant_turns, bins=range(1, 12), color='#FF9800',
              alpha=0.8, edgecolor='white', align='left')
axes2[1].set_title('Assistant Turns per Conversation')
axes2[1].set_xlabel('Number of Turns')
axes2[1].set_ylabel('Count')
axes2[1].set_xticks(range(1, 11))

plt.tight_layout()
plt.savefig("results/figures/conversation_structure.png",
            dpi=300, bbox_inches='tight')
plt.show()
print("  Saved → results/figures/conversation_structure.png")

#  Step 6: Save sample data
print("\n[6/6] Saving sample data to CSV...")

samples = []
for i in range(100):
    item = dataset['train'][i]
    samples.append({
        "id": i,
        "chosen":   item['chosen'][:500],
        "rejected": item['rejected'][:500],
        "chosen_len":   len(item['chosen'].split()),
        "rejected_len": len(item['rejected'].split()),
    })

pd.DataFrame(samples).to_csv("results/tables/sample_data.csv", index=False)
print("  Saved → results/tables/sample_data.csv")

#  Summary 
print("\n" + "=" * 60)
print("  DATA EXPLORATION COMPLETE")
print("=" * 60)
print("""
  Files saved:
    results/tables/dataset_statistics.csv
    results/tables/sample_data.csv
    results/figures/dataset_length_analysis.png
    results/figures/conversation_structure.png

  Key findings:
    - Dataset has chosen (preferred) and rejected (dispreferred) pairs
    - This is exactly the format DPO needs directly
    - RLHF needs this to train a reward model first
    - Chosen responses tend to be LONGER than rejected ones
      (humans prefer more complete answers)

  Next step: Run src/train_sft.py to train the SFT baseline
""")