# MPA: Multimodal Prototype Augmentation for Few-Shot Learning (AAAI 2026)

**Authors:** Liwen Wu, Wei Wang, Lei Zhao, Zhan Gao, Qika Lin, Shaowen Yao, Zuozhu Liu, Bin Pu  
**Conference:** AAAI 2026  
**Official Code:** [https://github.com/ww36user/MPA](https://github.com/ww36user/MPA)  
**Core Idea:** Instead of building class prototypes from 1 to 5 noisy support images alone, enrich the support set using **LLM-generated text descriptions**, **multi-view image augmentations**, and **synthetic "uncertain" boundary classes** extracted via CLIP before training a lightweight Logistic Regression classifier.

---

## 1. The Problem
Traditional prototype-based few-shot models (like 2017 Prototypical Networks) compute each class prototype using only raw support images. When you only have **1 to 5 images per class**, or when classes look very similar (fine-grained birds/cars) or come from a new domain (satellite/medical images):
* Raw images lack enough visual context and suffer from background noise.
* Single-view image prototypes fail to capture how an object looks from different angles, lighting, or distances.
* Ambiguous query samples near class borders easily get misclassified.

---

## 2. How MPA Works (Three Key Modules)

MPA uses a frozen pre-trained **CLIP (ViT-L/14)** image encoder ($f$) and text encoder ($h$) to extract features, enriching the support set through three modules before fitting a trainable **Logistic Regression** classifier ($g$):

### Module 1: LLM-based Multi-Variant Semantic Enhancement (LMSE)
* Prompts an LLM (e.g., GPT-4.0) with: *"Please generate an appearance description for {class name}, with four paraphrased variants."*
* Passes the 5 text descriptions through the frozen **CLIP text encoder** to get clean semantic features ($F_t$).
* Gives the model visual attribute clues (e.g., *"small, with a sharp, curved beak"*) that are free from image background noise.

### Module 2: Hierarchical Multi-View Augmentation (HMA)
* Simulates natural changes in viewing distance, camera angles, and lighting from each raw support image:
  * **Naturalistic Views:** Center crops ($120, 170, 200$ pixels), rotations ($45^\circ, 90^\circ, 180^\circ, 270^\circ, 315^\circ$), and color jitter.
  * **Geometric Views:** Horizontal reflections (flips).
- Encodes all augmented views through the **CLIP image encoder** to create a richer, multi-perspective visual cluster.

### Module 3: Adaptive Uncertain Class Absorber (AUCA)
* Creates an extra synthetic **"Uncertain Class"** ($D_u$) during training to absorb noisy boundary samples and reduce overlap between real classes.
* Combines **prototype interpolation** ($D_i = \alpha F_j + (1-\alpha) F_k$ between two different classes) and **Gaussian noise sampling** ($D_n \sim \mathcal{N}(0, 1)$).
* Automatically adjusts the mixing weight $\lambda$ based on how similar the classes in the current batch are (using smaller $\lambda$ when classes are tightly clustered in cross-domain tasks).

---

## 3. Paper Benchmark Results

MPA was evaluated across **4 single-domain** and **6 cross-domain** benchmarks, achieving massive improvements especially in the hardest **5-way 1-shot** setting:

| Benchmark Group | Datasets Tested | 5-Way 1-Shot Avg | 5-Way 5-Shot Avg | Gain Over 2nd Best (1-Shot) |
| :--- | :--- | :--- | :--- | :--- |
| **Single-Domain** | miniImageNet, tieredImageNet, CIFAR-FS, FC100 | **95.60%** *(98.87% miniImageNet)* | **96.52%** *(99.07% miniImageNet)* | **+12.29%** avg |
| **Cross-Domain** | CUB, Cars, Places, Plantae, EuroSAT, CropDisease | **94.18%** *(98.95% CUB)* | **96.28%** *(99.63% Cars)* | **+24.56%** avg |

### Component Ablation (EuroSAT 5-Way 1-Shot)
* **Baseline (CLIP only):** `76.41%`
* **+ LMSE (LLM Text):** `83.03%` *(+6.62% jump from text descriptions alone)*
* **+ LMSE + HMA (Multi-View):** `85.69%`
* **+ LMSE + HMA + AUCA (Full MPA):** `87.05%`

### Computational Efficiency
* Because CLIP stays frozen and only Logistic Regression is trained per episode, adding LMSE and HMA uses **<3% extra GPU memory** and adds **<0.02 seconds** per image at test time.

---

## 4. My PyTorch Reproduction (Kaggle)

Implemented a lightweight reproduction of MPA (`MPA_CIFAR_FS.ipynb`) on **Kaggle (NVIDIA T4 GPU)** using Hugging Face `transformers` (`openai/clip-vit-base-patch32`) and `scikit-learn` Logistic Regression across 100 unseen test episodes on **CIFAR-FS** (classes 64–99):

| Setting | Dataset | Backbone | Episodes | Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **5-Way 1-Shot** | CIFAR-FS (Unseen Classes) | CLIP `ViT-B/32` | 100 | **`87.28%`** |

### Key Implementation Takeaways
* **Zero Backbone Training:** Freezing CLIP and only fitting a multi-class Logistic Regression classifier per episode makes training fast and memory-light.
* **Multimodal Synergy:** Combining 5 paraphrased text prompts (LMSE) with 5 visual views per support image (HMA) turns a single 1-shot image into 10 complementary feature vectors, while AUCA prevents boundary overfitting.
* **Transformers v5 Compatibility:** Unwrapped `BaseModelOutputWithPooling` via `.pooler_output` before L2-normalizing embeddings.

---

## 5. Evolution: 2017 ProtoNet vs. 2026 MPA

| Feature | Prototypical Networks (2017) | MPA (AAAI 2026) |
| :--- | :--- | :--- |
| **Feature Extractor** | 4-block CNN trained from scratch | Frozen pre-trained CLIP (`ViT-L/14`) |
| **Input Modalities** | Images only | Images + Multi-view augmentations + LLM text descriptions |
| **Boundary Handling** | None (nearest prototype only) | Synthetic "Uncertain Class" via interpolation + Gaussian noise |
| **Classifier** | Squared Euclidean distance | Trainable Logistic Regression |
| **miniImageNet (5-way 1-shot)** | `49.42%` | **`98.87%`** |
