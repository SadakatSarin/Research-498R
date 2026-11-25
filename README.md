# Few-Shot Learning Research (CSE498R)

## Overview

This repository documents my research paper reviews, PyTorch/CLIP implementations from scratch, ablation studies, and experimental benchmarks in **Few-Shot Learning (FSL)**—tracing the evolution from classic metric-learning baselines to modern multimodal vision-language approaches.

---

## Research Objectives

- Understand the core paradigms of Few-Shot Learning (metric-based, optimization-based, and multimodal).
- Study and summarize foundational and state-of-the-art research papers.
- Reproduce existing models from scratch in PyTorch on Google Colab and Kaggle.
- Evaluate models across standard single-domain and cross-domain few-shot benchmarks.
- Compare experimental results, identify limitations, and formulate a novel research direction.

---

## Learning Progress

| Topic | Paradigm | Status |
| :--- | :--- | :--- |
| Few-Shot Learning Fundamentals | Core Concepts & Episodic Training | Completed |
| Metric Learning (`Prototypical Networks`) | Distance-Based Embeddings | Completed |
| Multimodal FSL (`MPA: AAAI 2026`) | Vision-Language + LLM Augmentation | Completed |
| Optimization-Based Meta-Learning (`MAML`) | Gradient-Based Adaptation | Planned |
| Attention-Based FSL (`Matching Networks`) | Memory & Cosine Attention | Planned |

---

## Research Papers Reviewed & Implemented

| Paper | Venue | Method | Status |
| :--- | :--- | :--- | :--- |
| **Prototypical Networks for Few-Shot Learning** (Snell et al.) | NeurIPS 2017 | Metric Learning (Euclidean Prototypes) | **Completed** (`01_ProtoNet_2017/`) |
| **MPA: Multimodal Prototype Augmentation for Few-Shot Learning** (Wu et al.) | AAAI 2026 | Frozen CLIP + LLM Text + Multi-View + AUCA | **Completed** (`02_MPA_AAAI2026/`) |
| **Model-Agnostic Meta-Learning (MAML)** (Finn et al.) | ICML 2017 | Optimization-Based Meta-Learning | Planned |
| **Matching Networks for One Shot Learning** (Vinyals et al.) | NeurIPS 2016 | Attention-Based Metric Learning | Planned |

---

## Models Implemented & Reproduced Results

### 1. Prototypical Networks (`01_ProtoNet_2017/`)
- **Implementation:** 4-block Convolutional Encoder trained episodically from scratch in PyTorch (`Google Colab`).
- **Additional Experiments:** t-SNE cluster visualization, Euclidean vs. Cosine distance metric ablation, and cross-dataset evaluation on color images (`CIFAR-FS`).
- **Reproduced Benchmark:** **`98.60%`** 5-way 5-shot test accuracy on **Omniglot**.

### 2. MPA: Multimodal Prototype Augmentation (`02_MPA_AAAI2026/`)
- **Implementation:** Frozen Hugging Face `CLIP (ViT-B/32)` backbone with **LMSE** (multi-variant semantic text prompts), **HMA** (hierarchical naturalistic & geometric image views), **AUCA** (adaptive uncertain class interpolation + Gaussian noise), and `scikit-learn` Logistic Regression (`Kaggle T4 GPU`).
- **Reproduced Benchmark:** **`87.28%`** 5-way 1-shot test accuracy across 100 unseen episodes on **CIFAR-FS**.

| Model | Year | Backbone | Dataset | Setting | Reproduced Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Prototypical Networks** | 2017 | 4-Layer CNN (From Scratch) | Omniglot | 5-Way 5-Shot | **`98.60%`** |
| **MPA (LMSE + HMA + AUCA)** | 2026 | Frozen CLIP `ViT-B/32` | CIFAR-FS | 5-Way 1-Shot | **`87.28%`** |

---

## Datasets Used

- **Omniglot:** Handwritten characters across 50 alphabets (standard grayscale meta-learning benchmark).
- **CIFAR-FS:** 100-class color object recognition split into 64 train / 16 val / 20 test classes (64–99 unseen evaluation).
- **Mini-ImageNet:** Planned for cross-model comparison.

---

## Technologies

- **Deep Learning & Vision-Language:** Python, PyTorch, Torchvision, Hugging Face `transformers` (CLIP)
- **Machine Learning & Analysis:** Scikit-learn, NumPy, Matplotlib
- **Compute & Environment:** Google Colab, Kaggle Notebooks (NVIDIA T4 GPU)

---

## Repository Structure

```text
├── 01_ProtoNet_2017/        # Prototypical Networks (NeurIPS 2017): Paper PDF, Colab notebook, ablation & README
├── 02_MPA_AAAI2026/         # MPA (AAAI 2026): Paper PDF, Kaggle notebook, reproduction results & README
└── README.md                # Main research overview, progress tracker, and benchmark comparison
