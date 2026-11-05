# Few-Shot Learning – Learning Roadmap

A structured roadmap to understand Few-Shot Learning, its major approaches, and commonly used datasets.

---

# 1. Understand Few-Shot Learning

Learn the following concepts in order:

1. **What is Few-Shot Learning?**
2. **Why is Few-Shot Learning needed?**
3. **Difference between Traditional Machine Learning and Few-Shot Learning.**
4. **What is N-way K-shot classification?**
5. **What are Support Set and Query Set?**
6. **What is an Episode in Few-Shot Learning?**
7. **What are Meta-learning and Transfer Learning?**
8. **What are Base Classes and Novel Classes?**

---

# 2. Learn the Major Few-Shot Learning Approaches

Study these approaches in the following order:

## 2.1 Siamese Networks

Learn how two inputs are compared using similarity scores.

**Focus:** Similarity Learning

## 2.2 Prototypical Networks

Learn how class prototypes are created using feature embeddings.

**Focus:** Distance-based Classification

## 2.3 Matching Networks

Learn how query samples are matched against support samples.

**Focus:** Attention and Similarity

## 2.4 MAML (Model-Agnostic Meta-Learning)

Understand how a model learns to adapt quickly using a few examples.

**Focus:** Meta-learning and Optimization

## 2.5 Relation Networks

Learn how neural networks measure relationships between samples.

**Focus:** Learned Similarity Functions

---

# 3. Learn Popular Few-Shot Learning Datasets

Start with commonly used Few-Shot Learning datasets.

| Dataset | Purpose |
|---|---|
| Omniglot | Handwritten character classification |
| MiniImageNet | Image classification |
| CIFAR-FS | Few-shot image classification |
| CUB-200 | Fine-grained bird classification |
| tieredImageNet | Large-scale image classification |

## Recommended Learning Order

1. **Omniglot** – Start here to understand few-shot learning with relatively simple handwritten character images.
2. **MiniImageNet** – Move here to practice few-shot image classification on more complex natural images.

---

# 4. Overall Learning Strategy

Follow this sequence:

1. Understand the fundamental concepts.
2. Learn the five major Few-Shot Learning approaches.
3. Understand how each approach works mathematically and practically.
4. Experiment with Omniglot.
5. Move to MiniImageNet.
6. Implement and compare different approaches.

**Goal:** Build a strong theoretical and practical understanding of Few-Shot Learning.
