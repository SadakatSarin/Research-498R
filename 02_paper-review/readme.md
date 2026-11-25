
# Research Paper Reviews (`02_paper-review`)

This directory contains structured reviews, architectural breakdowns, PyTorch/CLIP reproductions, and comparative analyses of foundational and state-of-the-art **Few-Shot Learning (FSL)** papers studied for **CSE498R**.

---

## Reviewed Papers

| # | Paper Title | Authors & Venue | Core Paradigm | Folder | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **Prototypical Networks for Few-Shot Learning** | Snell et al., *NeurIPS 2017* | Metric Learning (Euclidean Class Prototypes) | [`Prototypical Networks for Few-shot Learning`](./Prototypical%20Networks%20for%20Few-shot%20Learning) | **Completed** |
| **2** | **MPA: Multimodal Prototype Augmentation for Few-Shot Learning** | Wu et al., *AAAI 2026* | Frozen CLIP + LLM Text + Multi-View + AUCA | [`MPA: Multimodal Prototype Augmentation for Few-Shot Learning`](./MPA:%20Multimodal%20Prototype%20Augmentation%20for%20Few-Shot%20Learning) | **Completed** |

---

## Key Evolution & Comparative Takeaways

Studying these two papers side-by-side highlights how few-shot learning evolved from pure visual metric learning to multimodal foundation-model adaptation:

| Dimension | 1. Prototypical Networks (2017) | 2. MPA (AAAI 2026) |
| :--- | :--- | :--- |
| **Backbone** | 4-Block CNN trained from scratch | Frozen pre-trained CLIP (`ViT-L/14` / `ViT-B/32`) |
| **Modality** | Single-modal (raw support images only) | Multimodal (multi-view images + LLM text descriptions) |
| **Prototype Construction** | Mean vector of support image embeddings | Enriched support set + synthetic "Uncertain Class" (AUCA) |
| **Classifier** | Non-parametric squared Euclidean distance | Trainable Logistic Regression per episode |
| **Main Limitation** | Struggles with noisy/scarce support shots & domain shift | Relies on external LLM prompts & strong CLIP domain coverage |
| **My Reproduced Result** | **`98.60%`** (Omniglot, 5-way 5-shot) | **`87.28%`** (CIFAR-FS, 5-way 1-shot, `ViT-B/32`) |

---

## Folder Contents

Each paper folder includes:
- Original research paper (`.pdf`)
- Detailed technical summary & reproduction notes (`README.md`)
- Runnable Jupyter/Kaggle/Colab implementation notebook (`.ipynb`)

---

## Next Planned Reviews

- **Model-Agnostic Meta-Learning (MAML)** — Finn et al., *ICML 2017* (Optimization-based meta-learning)
- **Matching Networks for One Shot Learning** — Vinyals et al., *NeurIPS 2016* (Attention-based metric learning)
```[cite: 12]

*(Note: The links in the table are URL-encoded to match your exact folder names in `02_paper-review/` so clicking them on GitHub opens each paper's folder directly[cite: 12].)*
