import matplotlib.pyplot as plt
import pandas as pd
import os

os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)

train_steps  = [50,100,150,200,250,300,350,400,450,500,550,600,
                650,700,750,800,850,900,950,1000,1050,1100,1150,
                1200,1250,1300,1350,1400,1450,1500,1550,1600,
                1650,1700,1750,1800,1850]
train_losses = [3.856,1.506,1.324,1.271,1.236,1.265,1.229,1.246,
                1.195,1.319,1.212,1.285,1.212,1.195,1.233,1.191,
                1.215,1.18,1.186,1.227,1.169,1.205,1.221,1.134,
                1.235,1.167,1.129,1.184,1.18,1.201,1.172,1.125,
                1.182,1.206,1.162,1.226,1.13]

eval_epochs = [0.32,0.64,0.96,1.28,1.60,1.92,2.24,2.56,2.88,3.00]
eval_losses = [1.233,1.209,1.198,1.190,1.187,1.183,1.182,1.181,1.180,1.180]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("SFT Training — GPT-2 on HH-RLHF (5000 samples, 3 epochs)",
             fontsize=13, fontweight="bold")

axes[0].plot(train_steps, train_losses, color="#2196F3", linewidth=2)
axes[0].fill_between(train_steps, train_losses, alpha=0.1, color="#2196F3")
axes[0].set_title("Training Loss", fontweight="bold")
axes[0].set_xlabel("Step")
axes[0].set_ylabel("Loss")
axes[0].axhline(y=1.18, color="red", linestyle="--", alpha=0.5, label="Final: 1.18")
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[0].annotate("Start: 3.86", xy=(50,3.856), xytext=(300,3.4),
                 arrowprops=dict(arrowstyle="->",color="gray"), fontsize=9, color="gray")
axes[0].annotate("End: 1.13", xy=(1850,1.13), xytext=(1400,1.5),
                 arrowprops=dict(arrowstyle="->",color="green"), fontsize=9, color="green")

axes[1].plot(eval_epochs, eval_losses, color="#4CAF50", linewidth=2,
             marker="o", markersize=7)
axes[1].fill_between(eval_epochs, eval_losses, alpha=0.1, color="#4CAF50")
axes[1].set_title("Validation Loss", fontweight="bold")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Eval Loss")
axes[1].set_ylim(1.17, 1.25)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("results/figures/sft_training_curves.png", dpi=300, bbox_inches="tight")
print("✅ Saved → results/figures/sft_training_curves.png")

pd.DataFrame([{
    "model":  "gpt2",
    "method":"SFT",
    "train_samples":     5000,
    "epochs":3,
    "initial_loss":3.856,
    "final_train_loss":1.286,
    "final_eval_loss": 1.180,
    "training_time_min": 11.9,
}]).to_csv("results/tables/sft_results.csv", index=False)
print("✅ Saved → results/tables/sft_results.csv")

print("\n  SFT Results Summary:")
print("Initial loss : 3.856")
print("Final train loss : 1.286")
print("Final eval loss  : 1.180")
print("Loss reduction : 69.4%")
print("Training time : 11.9 minutes")
print("Checkpoint : models/sft_checkpoint/")
print("Status:COMPLETE ✅")