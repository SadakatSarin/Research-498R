
# Prototypical Networks for Few-shot Learning (NIPS 2017)

**Authors:** Jake Snell, Kevin Swersky, Richard Zemel  
**Topic:** Few-Shot & Zero-Shot Classification  
**Core Idea:** Instead of training a complex model to adapt to new classes, map images into a feature space where each class is represented by a single "center point" (prototype), and classify new images by finding the closest prototype.

---

## 1. The Problem
Standard deep learning models need thousands of images per class to learn well. If you only give them 1 to 5 images of a brand-new class (Few-Shot Learning), they severely overfit. Previous methods tried to solve this using complicated attention mechanisms or LSTM meta-learners, which are heavy and hard to train.

---

## 2. How Prototypical Networks Work

The model learns a neural network encoder $f_\phi$ that turns input images into points (vectors) in an embedding space:

1. **Create Prototypes (Support Set):**  
   Take a few labeled examples for each new class (the *support set*). Pass them through the neural network and calculate the **average (mean) vector** for each class. This average point is the class **prototype** ($c_k$):
   $$c_k = \frac{1}{|S_k|} \sum_{(x_i, y_i) \in S_k} f_\phi(x_i)$$

2. **Classify New Images (Query Set):**  
   Pass an unlabeled test image (the *query point*) through the same neural network. Measure the distance between this image and every class prototype. The closest prototype wins!
   $$p_\phi(y=k|x) = \frac{\exp(-d(f_\phi(x), c_k))}{\sum_{k'} \exp(-d(f_\phi(x), c_{k'}))}$$

3. **Episodic Training:**  
   During training, the model practices on fake mini-tasks called **episodes**. In each episode, we randomly pick $N_C$ classes, $N_S$ support images per class, and $N_Q$ query images per class. This forces the network to learn *how to group similar things* rather than memorizing fixed classes.

4. **Zero-Shot Extension:**  
   If we have **zero** images for a class and only a text description or attribute vector ($v_k$), a separate network $g_\vartheta(v_k)$ maps that description into the same feature space to act as the prototype.

---

## 3. Key Findings & Design Tips

* **Use Squared Euclidean Distance (Not Cosine):**  
  Most earlier papers used cosine similarity, but the authors showed **squared Euclidean distance** ($||z - z'||^2$) works much better. Mathematically, taking the average (mean) of points only gives the optimal cluster center when using a Bregman divergence like squared Euclidean distance.
* **Train with More Classes ("Higher Way"):**  
  If you want to test on **5-way** classification (choosing between 5 classes), train the model on harder **20-way** or **60-way** episodes. Harder training episodes force the model to learn sharper features.
* **Match the "Shot" Number:**  
  If you test on 5-shot tasks, train on 5-shot episodes.
* **Simple CNN Backbone:**  
  No pre-trained ResNets or complex tricks were used—just **4 convolutional blocks** (each with a $3\times3$ Conv layer, BatchNorm, ReLU, and $2\times2$ MaxPool).

---

## 4. Benchmark Results

| Dataset | Setting | Accuracy | Takeaway |
| :--- | :--- | :--- | :--- |
| **Omniglot** | 5-way 1-shot / 5-shot | **98.8%** / **99.7%** | Beats Matching Networks & MAML |
| **Omniglot** | 20-way 1-shot / 5-shot | **96.0%** / **98.9%** | State-of-the-art with much simpler code |
| **miniImageNet** | 5-way 1-shot / 5-shot | **49.42%** / **68.20%** | Beats MAML (63.15%) & Meta-LSTM (60.60%) |
| **CUB-200** *(Zero-Shot)* | 50-way 0-shot | **54.8%** *(attributes)* / **58.3%** *(text)* | Works even without example images |

---

## 5. Summary for Implementation
* **1-Shot case:** When there is only 1 example per class, the prototype is just that single image's embedding (making it identical to nearest-neighbor / Matching Networks).
* **Why it matters:** It is fast, easy to code in PyTorch, requires no extra fine-tuning at test time, and serves as the foundation for modern prototype-based few-shot methods.
```[cite: 2, 3, 4, 5, 6, 7, 8, 9]
