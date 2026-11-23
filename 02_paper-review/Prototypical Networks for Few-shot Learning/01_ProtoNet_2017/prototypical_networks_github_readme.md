# Prototypical Networks for Few-shot Learning (NIPS 2017)

**Authors:** Jake Snell, Kevin Swersky, Richard Zemel  
**Topic:** Few-Shot & Zero-Shot Classification  
**Implementation:** PyTorch on Omniglot (`ProtoNet_Omniglot.ipynb`)

---

## 1. Paper Summary

### The Problem
Standard deep learning models require thousands of labeled images per class. When given only 1 to 5 images of an unseen class (**Few-Shot Learning**), traditional classifiers severely overfit. Earlier few-shot approaches relied on complex attention modules or LSTM meta-learners, which are computationally heavy and difficult to train.

### Core Idea
Prototypical Networks learn a neural network encoder $f_\phi$ that maps images into a feature space where points of the same class cluster around a single center point called a **prototype**:

1. **Compute Class Prototypes ($c_k$):**  
   Pass the labeled examples of each class (the *support set* $S_k$) through the encoder and take their mean vector:
   $$c_k = \frac{1}{|S_k|} \sum_{(x_i, y_i) \in S_k} f_\phi(x_i)$$

2. **Classify Query Images ($x$):**  
   Pass an unlabeled test image (the *query point*) through the same encoder and measure its distance to every prototype. The closest prototype determines the predicted class:
   $$p_\phi(y=k|x) = \frac{\exp(-d(f_\phi(x), c_k))}{\sum_{k'} \exp(-d(f_\phi(x), c_{k'}))}$$

3. **Episodic Training:**  
   Instead of standard mini-batches, the model trains on simulated few-shot tasks (**episodes**), each sampling $N_C$ classes, $N_S$ support images per class, and $N_Q$ query images per class.

4. **Zero-Shot Extension:**  
   When no example images exist for a class, a separate network $g_\vartheta(v_k)$ maps class metadata/attributes ($v_k$) into the shared embedding space to serve as the prototype.

### Key Takeaways from the Paper
* **Squared Euclidean Distance > Cosine Similarity:** Taking the class mean is mathematically optimal only under Bregman divergences such as squared Euclidean distance ($||z - z'||^2$).
* **Train with a Higher "Way":** Training on harder episodes with more classes (e.g., 20-way or 60-way) improves generalization when testing on 5-way tasks.
* **Match the "Shot":** Training and testing with the same number of support examples per class gives the best performance.

---

## 2. PyTorch Implementation Breakdown (`ProtoNet_Omniglot.ipynb`)

The notebook reproduces Prototypical Networks from scratch on the **Omniglot** handwritten character dataset in four steps:

1. **Dataset Preparation:**  
   Downloads Omniglot, resizes images to $28 \times 28$ grayscale tensors, and groups images by character label into separate training (`background=True`) and unseen test (`background=False`) dictionaries.
2. **Episodic Sampler (`sample_episode`):**  
   Dynamically generates $N$-way $K$-shot tasks by randomly sampling $N$ classes and splitting each class's images into a **Support Set** (to build prototypes) and a **Query Set** (to compute classification loss).
3. **4-Block CNN Encoder & Prototypical Loss:**  
   * **Encoder:** Four convolutional blocks (`Conv2d(3x3)` $\rightarrow$ `BatchNorm2d` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2)`), mapping each $28 \times 28$ image into a **64-dimensional** embedding vector.
   * **Loss Function:** Averages support embeddings per class to form prototypes, computes pairwise **squared Euclidean distances** (`torch.cdist(z_query, prototypes) ** 2`), and optimizes negative log-likelihood over the negative distances.
4. **Training & Evaluation:**  
   * **Training:** 10 epochs $\times$ 100 episodes (1,000 episodes total) using **20-way 5-shot** tasks with Adam (`lr=1e-3`).
   * **Testing:** Evaluated across 200 random **5-way 5-shot** episodes on completely unseen characters with zero fine-tuning.

---

## 3. Results

| Setup | Dataset | Accuracy | Notes |
| :--- | :--- | :--- | :--- |
| **Our Implementation** | Omniglot (5-way 5-shot) | **98.60%** | Trained for 1,000 episodes (20-way) without rotation augmentation |
| **Original Paper (Table 1)** | Omniglot (5-way 5-shot) | **99.70%** | Trained with 60-way episodes + $90^\circ$ rotation augmentation |