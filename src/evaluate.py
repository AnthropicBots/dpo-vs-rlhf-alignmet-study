
"""
evaluate.py — Full comparison of SFT vs DPO vs RLHF
Run: python src/evaluate.py
Requires: all 3 model checkpoints to exist
"""
import os, torch, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from transformers import AutoTokenizer, AutoModelForCausalLM

os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("="*60)
print("  EVALUATION: SFT vs DPO vs RLHF")
print("="*60)
print(f"\n  Device: {DEVICE}\n")

#  Test prompts 
TEST_PROMPTS = [
    "\n\nHuman: What is machine learning?\n\nAssistant:",
    "\n\nHuman: How do I reduce stress at work?\n\nAssistant:",
    "\n\nHuman: Explain artificial intelligence to a beginner.\n\nAssistant:",
    "\n\nHuman: What are some good habits for productivity?\n\nAssistant:",
    "\n\nHuman: How does the internet work?\n\nAssistant:",
    "\n\nHuman: What makes a good leader?\n\nAssistant:",
    "\n\nHuman: How can I improve my sleep quality?\n\nAssistant:",
    "\n\nHuman: Tell me about climate change.\n\nAssistant:",
    "\n\nHuman: What is the best way to learn programming?\n\nAssistant:",
    "\n\nHuman: How do I deal with anxiety?\n\nAssistant:",
]

MODELS = {
    "SFT Baseline":    "models/sft_checkpoint",
    "DPO Fine-tuned":  "models/dpo_checkpoint",
    "RLHF Fine-tuned": "models/rlhf_checkpoint",
}

COLORS = {
    "SFT Baseline":    "#78909C",
    "DPO Fine-tuned":  "#7E57C2",
    "RLHF Fine-tuned": "#FF7043",
}

#  Helper functions 
def load_model(path):
    tok = AutoTokenizer.from_pretrained(path)
    tok.pad_token    = tok.eos_token
    tok.padding_side = "left"
    mdl = AutoModelForCausalLM.from_pretrained(path).to(DEVICE)
    mdl.eval()
    return tok, mdl

def generate(tok, mdl, prompt, max_new=80):
    enc = tok(prompt, return_tensors="pt",
              truncation=True, max_length=200).to(DEVICE)
    with torch.no_grad():
        out = mdl.generate(
            **enc,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(
        out[0][enc["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

def score_response(response):
    """
    Composite quality score (0-1):
    - Lexical diversity  (unique words / total words)
    - Length adequacy    (normalised to 60 words ideal)
    - Repetition penalty (unique bigrams / total bigrams)
    """
    words = response.split()
    if len(words) < 3:
        return 0.0
    diversity    = len(set(words)) / max(len(words), 1)
    length_score = min(len(words) / 60.0, 1.0)
    if len(words) > 1:
        bigrams  = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        no_rep   = 1 - (len(bigrams)-len(set(bigrams))) / max(len(bigrams), 1)
    else:
        no_rep = 1.0
    return round(diversity*0.4 + length_score*0.3 + no_rep*0.3, 4)

#  Run evaluation 
all_results       = []
scores_by_method  = {}
latency_by_method = {}
responses_by_method = {}

for method, path in MODELS.items():
    if not os.path.exists(path):
        print(f"  Skipping {method} — checkpoint not found")
        continue

    print(f"\n  Evaluating: {method}")
    print(f"  Loading from: {path}")
    tok, mdl = load_model(path)

    scores, latencies, responses = [], [], []

    for i, prompt in enumerate(TEST_PROMPTS):
        t0   = time.time()
        resp = generate(tok, mdl, prompt)
        lat  = time.time() - t0
        sc   = score_response(resp)

        scores.append(sc)
        latencies.append(lat)
        responses.append(resp)

        all_results.append({
            "method":     method,
            "prompt_id":  i,
            "prompt":     prompt.split("Human:")[1].split("Assistant:")[0].strip(),
            "response":   resp[:300],
            "score":      sc,
            "latency_s":  round(lat, 3),
            "word_count": len(resp.split()),
        })

        if i < 3:
            q = prompt.split("Human:")[1].split("Assistant:")[0].strip()
            print(f"  [{i+1}] Q: {q[:50]}...")
            print(f"       A: {resp[:80]}...")
            print(f"       Score: {sc:.3f} | Latency: {lat:.2f}s")

    scores_by_method[method]    = scores
    latency_by_method[method]   = latencies
    responses_by_method[method] = responses

    del mdl
    torch.cuda.empty_cache()

#  Save raw results 
df_raw = pd.DataFrame(all_results)
df_raw.to_csv("results/tables/all_responses.csv", index=False)
print("\n  Saved → results/tables/all_responses.csv")

#  Summary table 
summary = []
for method in scores_by_method:
    sc  = scores_by_method[method]
    lat = latency_by_method[method]
    summary.append({
        "Method":          method,
        "Avg Score":       round(np.mean(sc),  4),
        "Std Dev":         round(np.std(sc),   4),
        "Min Score":       round(np.min(sc),   4),
        "Max Score":       round(np.max(sc),   4),
        "Avg Latency(s)":  round(np.mean(lat), 3),
        "Avg Word Count":  round(np.mean([
            len(r.split()) for r in responses_by_method[method]
        ]), 1),
    })

df_summary = pd.DataFrame(summary)
df_summary.to_csv("results/tables/evaluation_summary.csv", index=False)

print("\n" + "="*60)
print("  EVALUATION SUMMARY")
print("="*60)
print(df_summary.to_string(index=False))

# 
# FIGURES
methods = list(scores_by_method.keys())
colors  = [COLORS[m] for m in methods]

#  Figure 1: Main bar chart 
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("DPO vs RLHF vs SFT — Full Evaluation Results",
             fontsize=14, fontweight="bold")

avg_scores = [np.mean(scores_by_method[m]) for m in methods]
std_scores = [np.std(scores_by_method[m])  for m in methods]

bars = axes[0,0].bar(methods, avg_scores, color=colors, alpha=0.85,
                     yerr=std_scores, capsize=6,
                     error_kw={"linewidth":2, "ecolor":"gray"})
axes[0,0].set_title("Average Quality Score (higher = better)",
                    fontweight="bold")
axes[0,0].set_ylabel("Score (0–1)")
axes[0,0].set_ylim(0, 1.1)
axes[0,0].grid(axis="y", alpha=0.3)
for bar, sc in zip(bars, avg_scores):
    axes[0,0].text(bar.get_x()+bar.get_width()/2,
                   bar.get_height()+0.03,
                   f"{sc:.3f}", ha="center", fontweight="bold", fontsize=11)

#  Figure 2: Per-prompt heatmap 
x    = np.arange(len(TEST_PROMPTS))
w    = 0.25
for i, m in enumerate(methods):
    axes[0,1].bar(x+i*w, scores_by_method[m], w,
                  label=m, color=colors[i], alpha=0.8)
axes[0,1].set_title("Per-Prompt Score Comparison", fontweight="bold")
axes[0,1].set_xlabel("Prompt ID")
axes[0,1].set_ylabel("Score")
axes[0,1].set_xticks(x+w)
axes[0,1].set_xticklabels([str(i) for i in range(len(TEST_PROMPTS))])
axes[0,1].legend(fontsize=9)
axes[0,1].grid(axis="y", alpha=0.3)

#  Figure 3: Latency 
avg_lat = [np.mean(latency_by_method[m]) for m in methods]
lat_bars = axes[1,0].bar(methods, avg_lat, color=colors, alpha=0.85)
axes[1,0].set_title("Average Generation Latency (lower = better)",
                    fontweight="bold")
axes[1,0].set_ylabel("Seconds per response")
axes[1,0].grid(axis="y", alpha=0.3)
for bar, lat in zip(lat_bars, avg_lat):
    axes[1,0].text(bar.get_x()+bar.get_width()/2,
                   bar.get_height()+0.01,
                   f"{lat:.2f}s", ha="center", fontweight="bold")

#  Figure 4: Score distribution violin 
parts = axes[1,1].violinplot(
    [scores_by_method[m] for m in methods],
    positions=range(len(methods)),
    showmeans=True, showmedians=True
)
for i, pc in enumerate(parts["bodies"]):
    pc.set_facecolor(colors[i])
    pc.set_alpha(0.7)
axes[1,1].set_xticks(range(len(methods)))
axes[1,1].set_xticklabels(methods, fontsize=9)
axes[1,1].set_title("Score Distribution (violin plot)", fontweight="bold")
axes[1,1].set_ylabel("Score")
axes[1,1].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("results/figures/main_comparison.png",
            dpi=300, bbox_inches="tight")
print("\n  Saved → results/figures/main_comparison.png")

#  Figure 2: Training efficiency 
try:
    sft_r  = pd.read_csv("results/tables/sft_results.csv")
    dpo_r  = pd.read_csv("results/tables/dpo_results.csv")
    rlhf_r = pd.read_csv("results/tables/rlhf_results.csv")

    train_times = {
        "SFT Baseline":    float(sft_r["training_time_min"].iloc[0]),
        "DPO Fine-tuned":  float(dpo_r["training_time_min"].iloc[0]),
        "RLHF Fine-tuned": float(rlhf_r["training_time_min"].iloc[0]),
    }

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    fig2.suptitle("Training Efficiency Comparison", fontweight="bold")

    t_methods = list(train_times.keys())
    t_colors  = [COLORS[m] for m in t_methods]
    t_vals    = list(train_times.values())

    bars2 = axes2[0].bar(t_methods, t_vals, color=t_colors, alpha=0.85)
    axes2[0].set_title("Total Training Time (minutes)", fontweight="bold")
    axes2[0].set_ylabel("Minutes")
    axes2[0].grid(axis="y", alpha=0.3)
    for bar, v in zip(bars2, t_vals):
        axes2[0].text(bar.get_x()+bar.get_width()/2,
                      v+0.2, f"{v:.1f}m",
                      ha="center", fontweight="bold")

    final_losses = {
        "SFT Baseline":    float(sft_r["final_train_loss"].iloc[0]),
        "DPO Fine-tuned":  float(dpo_r["final_loss"].iloc[0]),
        "RLHF Fine-tuned": float(rlhf_r["final_train_loss"].iloc[0]),
    }
    l_vals = list(final_losses.values())
    bars3  = axes2[1].bar(t_methods, l_vals, color=t_colors, alpha=0.85)
    axes2[1].set_title("Final Training Loss", fontweight="bold")
    axes2[1].set_ylabel("Loss")
    axes2[1].grid(axis="y", alpha=0.3)
    for bar, v in zip(bars3, l_vals):
        axes2[1].text(bar.get_x()+bar.get_width()/2,
                      v+0.01, f"{v:.4f}",
                      ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig("results/figures/training_efficiency.png",
                dpi=300, bbox_inches="tight")
    print("  Saved → results/figures/training_efficiency.png")

except Exception as e:
    print(f"  Note: {e}")

# Figure 3: Word count comparison
fig3, ax3 = plt.subplots(figsize=(10, 5))
fig3.suptitle("Response Word Count Distribution by Method",
              fontweight="bold")

wc_data = []
wc_labels = []
for m in methods:
    wc = [len(r.split()) for r in responses_by_method[m]]
    wc_data.append(wc)
    wc_labels.append(m)

parts3 = ax3.violinplot(wc_data, positions=range(len(methods)),
                        showmeans=True)
for i, pc in enumerate(parts3["bodies"]):
    pc.set_facecolor(colors[i])
    pc.set_alpha(0.7)
ax3.set_xticks(range(len(methods)))
ax3.set_xticklabels(wc_labels)
ax3.set_ylabel("Word Count")
ax3.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("results/figures/response_length_comparison.png",
            dpi=300, bbox_inches="tight")
print("  Saved → results/figures/response_length_comparison.png")

# Figure 4: Sample responses side by side 
fig4, axes4 = plt.subplots(len(methods), 1,
                            figsize=(14, 4*len(methods)))
fig4.suptitle("Sample Responses: Prompt 1 — What is machine learning?",
              fontweight="bold", fontsize=13)

for i, m in enumerate(methods):
    resp = responses_by_method[m][0]
    sc   = scores_by_method[m][0]
    axes4[i].text(0.01, 0.95, f"{m}  |  Score: {sc:.3f}",
                  transform=axes4[i].transAxes,
                  fontsize=12, fontweight="bold", color=colors[i],
                  va="top")
    wrapped = resp[:400]
    axes4[i].text(0.01, 0.75, wrapped,
                  transform=axes4[i].transAxes,
                  fontsize=10, va="top", wrap=True,
                  color="#333333")
    axes4[i].axis("off")
    axes4[i].set_facecolor("#F8F8F8")
    for spine in axes4[i].spines.values():
        spine.set_visible(False)

plt.tight_layout()
plt.savefig("results/figures/sample_responses.png",
            dpi=300, bbox_inches="tight")
print("  Saved → results/figures/sample_responses.png")

#  Final summary 
print("\n" + "="*60)
print("  EVALUATION COMPLETE")
print("="*60)
print("""
  Files saved:
    results/tables/all_responses.csv
    results/tables/evaluation_summary.csv
    results/figures/main_comparison.png
    results/figures/training_efficiency.png
    results/figures/response_length_comparison.png
    results/figures/sample_responses.png
""")
print(df_summary.to_string(index=False))
print("\n  Next: python src/write_paper.py")