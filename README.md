# DPO vs RLHF: A Comparative Alignment Study

**Paper Status:** In Progress — Day 2/5 Complete
**Author:** Mohit Yadav — Chandigarh University
**Target Venue:** EMNLP 2025 / COLING 2026
**GitHub:** https://github.com/AnthropicBots/dpo-vs-rlhf-alignmet-study

---

## Abstract
This study presents a systematic empirical comparison of two dominant
LLM alignment paradigms: Reinforcement Learning from Human Feedback (RLHF)
and Direct Preference Optimisation (DPO). We fine-tune GPT-2 (124M) on the
Anthropic HH-RLHF dataset (160,800 preference pairs) under both training
regimes and evaluate across response quality, training efficiency, and
alignment robustness metrics.

---

## Research Questions
1. Does DPO achieve comparable alignment quality to RLHF at lower compute cost?
2. How do both methods perform on out-of-distribution prompts?
3. Which method is more robust to reward hacking?

---

## Progress

| Day | Task | Status |
|-----|------|--------|
| Day 1 | Environment setup, dataset exploration | ✅ Done |
| Day 2 | SFT baseline training (loss 3.86 → 1.18) | ✅ Done |
| Day 3 | DPO + RLHF training | 🔄 In Progress |
| Day 4 | Full evaluation + graphs | ⏳ Pending |
| Day 5 | Paper writing + arXiv upload | ⏳ Pending |

---

## Key Results
*(Updated after Day 4)*

| Method | Avg Quality Score | Training Time | Reward Hacking |
|--------|------------------|---------------|----------------|
| SFT Baseline | - | 11.9 min | N/A |
| DPO | - | - | None (by design) |
| RLHF | - | - | Possible |

---

## Reproduce This Work
```bash
git clone https://github.com/AnthropicBots/dpo-vs-rlhf-alignmet-study
cd dpo-vs-rlhf-alignmet-study
pip install -r requirements.txt
python src/data_prep.py    # Day 1
python src/train_sft.py    # Day 2
python src/train_dpo.py    # Day 3
python src/train_rlhf.py   # Day 3
python src/evaluate.py     # Day 4
python src/write_paper.py  # Day 5
```

---

## Citation
```bibtex
@article{yadav2026dpo,
  title={DPO vs RLHF: An Empirical Comparison of LLM Alignment Techniques},
  author={Yadav, Mohit},
  institution={Chandigarh University},
  year={2026}
}
```