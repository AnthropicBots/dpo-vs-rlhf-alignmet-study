# DPO vs RLHF: A Comparative Alignment Study

**Paper Status:** In Progress  
**Authors:** [Mohit Yadav] — [Chandigarh University]  
**Target Venue:** EMNLP 2025 / COLING 2026

---

## Abstract
This study presents a systematic empirical comparison of two dominant
LLM alignment paradigms: Reinforcement Learning from Human Feedback (RLHF)
and Direct Preference Optimisation (DPO). We fine-tune LLaMA-3-8B on the
Anthropic HH-RLHF dataset under both training regimes and evaluate across
helpfulness, harmlessness, honesty (3H), and computational efficiency metrics.

---

## Research Questions
1. Does DPO achieve comparable alignment quality to RLHF at lower compute cost?
2. How do both methods perform on out-of-distribution prompts?
3. Which method is more robust to reward hacking?

---

## Key Results
*(To be filled after experiments)*

| Method | Helpfulness | Harmlessness | Training Time |
|--------|-------------|--------------|---------------|
| RLHF   | -           | -            | -             |
| DPO    | -           | -            | -             |

---

## Reproduce This Work
```bash
git clone https://github.com/YOURUSERNAME/dpo-vs-rlhf-alignment-study
cd dpo-vs-rlhf-alignment-study
pip install -r requirements.txt
python src/train_dpo.py
```

---

## Citation
```bibtex
@article{yourname2025dpo,
  title={DPO vs RLHF: A Comparative Alignment Study},
  author={Mohit Yadav},
  year={2025}
}
```
