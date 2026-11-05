# Matching Networks for One-Shot Learning: A Beginner-Friendly Study Guide

*Based on the paper by Vinyals, Blundell, Lillicrap, Kavukcuoglu and Wierstra (Google DeepMind, 2016)*

## 1. The Big Picture in 60 Seconds

A child can see **one** picture of a giraffe in a book and then recognize giraffes for the rest of their life. A normal deep learning model needs hundreds or thousands of giraffe photos to do the same.

This paper asks: **can we build a neural network that learns a brand-new class from just one example ("one-shot learning") without being retrained?**

Their answer is the **Matching Network**. The idea, in plain words:

1.       Give the network a tiny "cheat sheet" of labelled examples (the **support set**).

2.       Give it a new, unlabelled item (the **query**).

3.       The network **compares** the query to every example on the cheat sheet, finds which ones it looks most like, and **copies their labels**, weighted by how similar they are.

That's it. No retraining, no fine-tuning. If you hand it a different cheat sheet with different classes, it just works with those.

**Analogy:** Imagine a bouncer with a small stack of 5 photo IDs. A person walks up. The bouncer looks at all 5 photos, decides "this person looks 80% like photo #3 and 5% like each of the others," and goes with #3. Swap in a new stack of IDs tomorrow and the bouncer still does the same job, with no new training.

## 2. Key Vocabulary (Learn These First)

| Term | Simple meaning |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **One-shot learning** | Learning a new class from exactly **1** labelled example. |
| **Few-shot / k-shot learning** | Same idea but with k examples per class (e.g. 5-shot = 5 examples per class). |
| **N-way k-shot task** | A test where there are **N** possible classes and you get **k** examples of each. Random guessing gets 1/N accuracy. A 5-way task = 20% by chance. |
| **Support set (S)** | The small set of labelled examples you are given at test time. The "cheat sheet." |
| **Query / test example (x̂)** | The new, unlabelled item you must classify. |
| **Batch (B)** | The group of query examples used to compute the loss during training. |
| **Parametric model** | Knowledge is stored in the model's weights. Learning = slowly adjusting weights with gradient descent. (Normal neural networks.) |
| **Non-parametric model** | Knowledge is stored in the data itself. Adding a new example is instant. (Example: nearest neighbours.) |
| **Embedding** | Turning an input (image, word) into a vector of numbers so that similar things end up close together. |
| **Attention** | A way of deciding *how much to focus on each item*, producing weights that add up to 1. |
| **Meta-learning** | "Learning to learn." Training a model on many small tasks so it becomes good at picking up new tasks quickly. |
| **Episode** | One small training task (a mini support set plus a mini batch). |
| **Fine-tuning** | Continuing to train a model on new data. Matching Nets **don't need it**. |

## 3. The Problem: Why Normal Deep Learning Struggles

The paper points out two worlds:

**Parametric models (e.g. standard neural networks)**

•         Strong generalisation once trained.

•         But they need **lots of data** and **many slow gradient updates**.

•         Each training example must be "baked" into the weights. Learning a new class means retraining, which risks *catastrophic forgetting* (forgetting old classes).

**Non-parametric models (e.g. k-nearest neighbours)**

•         Can absorb new examples **instantly** (just add them to memory).

•         No catastrophic forgetting.

•         But performance depends heavily on the **distance metric** you choose, and raw distances (like pixel differences) are often poor.

**The paper's goal:** combine the best of both. Use a neural network to learn a *great* similarity measure (the parametric part), then use a nearest-neighbour-style lookup (the non-parametric part) so new classes can be added instantly.

## 4. The Two Main Contributions

1.       **Modeling:** A network architecture (Matching Networks) that uses **attention and memory** to map a support set plus a query to a label.

2.       **Training procedure:** Train the network the same way you will test it, by showing it **few examples per class and switching tasks every minibatch**. The principle: *"test and train conditions must match."*

There's also a side contribution: they define **new benchmark tasks**, including **miniImageNet** (a small version of ImageNet that fits in memory) and a **one-shot language task** on the Penn Treebank.

## 5. The Core Model (Section 2.1)

### 5.1 The Setup

You are given a support set of k labelled pairs:

S = {(x₁, y₁), (x₂, y₂), ..., (x_k, y_k)}

•         xᵢ = an example (e.g. an image)

•         yᵢ = its label (as a **one-hot vector**, e.g. cat = [1, 0, 0], dog = [0, 1, 0], bird = [0, 0, 1])

The model defines a classifier c_S(x̂) that, for a new query x̂, outputs a probability distribution over labels:

P(ŷ | x̂, S)

In words: *"Given this query and this support set, what's the probability of each label?"* The predicted class is simply the one with the highest probability:

ŷ = argmax_y P(y | x̂, S)

Key point: S → c_&#x53;**&#x20;is a mapping from a support set to a classifier.** Give it a new support set S′ and you instantly get a new classifier, with no retraining.

### 5.2 Equation 1: The Heart of the Paper

ŷ = Σ (i = 1 to k)  a(x̂, xᵢ) · yᵢ

**Read it as:** *"The predicted label is a weighted sum of the support-set labels, where each weight says how much the query 'matches' that support example."*

•         Σ means "add up over all support examples i = 1 ... k".

•         a(x̂, xᵢ) is the **attention weight**: how similar the query is to support example i. These weights are positive and sum to 1.

•         yᵢ is the one-hot label of support example i.

#### Worked Example

Support set: a cat photo, a dog photo, a bird photo. Labels:

•         y₁ = [1, 0, 0] (cat)

•         y₂ = [0, 1, 0] (dog)

•         y₃ = [0, 0, 1] (bird)

A new photo arrives. Suppose the network decides the attention weights are 0.50, 0.275, 0.225. Then:

ŷ = 0.50·[1,0,0] + 0.275·[0,1,0] + 0.225·[0,0,1] = [0.50, 0.275, 0.225]

The output is a probability distribution: 50% cat, 27.5% dog, 22.5% bird. The prediction is **cat**.

Notice what happened: because labels are one-hot, the weighted sum just collects the attention weight that belongs to each class. If two support examples were both cats, their weights would be added together for the "cat" slot.

#### Why this equation is clever

The paper shows Equation 1 is a **general form** that includes older methods:

•         If a is a **kernel** function, Equation 1 is a **kernel density estimator (KDE)**.

•         If a is zero for all but the nearest few examples, Equation 1 is **k-nearest neighbours (kNN)**.

•         It can also be seen as an **associative memory**: the xᵢ are "keys," the yᵢ are "values," and attention "points" at the right memory slot to retrieve its label.

And because it is non-parametric, **memory grows with the support set**. Adding more examples simply adds more entries, with no weight changes needed.

### 5.3 The Attention Kernel (Section 2.1.1)

Equation 1 needs a definition of a(x̂, xᵢ). The simplest choice the paper uses is **softmax over cosine similarity**:

a(x̂, xᵢ) = exp( c( f(x̂), g(xᵢ) ) ) / Σ (j = 1 to k) exp( c( f(x̂), g(xⱼ) ) )

Let's unpack each piece.

**1.&#x20;**&#x66;**&#x20;and&#x20;**&#x67;**&#x20;(embedding functions).** These are neural networks (e.g. a CNN for images). f(x̂) turns the query into a vector; g(xᵢ) turns each support example into a vector. They can be the same network (f = g) or different ones.

**2.&#x20;**&#x63;(·,·)**&#x20;(cosine similarity).** Measures how much two vectors point in the same direction:

c(u, v) = (u · v) / (‖u‖ · ‖v‖)

•         Value of **+1** = pointing the same way (very similar)

•         **0** = unrelated

•         **−1** = opposite

It ignores vector length and only cares about direction, which makes it a robust similarity score.

**3. Softmax.** Takes any list of scores and turns them into positive numbers that sum to 1. Steps: raise e to each score (making everything positive and amplifying bigger scores), then divide by the total.

#### Worked Example with Numbers

Suppose the cosine similarities between the query and the three support examples are 0.9 (cat), 0.3 (dog), 0.1 (bird).

1.       Exponentiate: e^0.9 ≈ 2.46, e^0.3 ≈ 1.35, e^0.1 ≈ 1.11

2.       Sum: 2.46 + 1.35 + 1.11 ≈ 4.91

3.       Divide: 2.46/4.91 ≈ **0.50**, 1.35/4.91 ≈ **0.275**, 1.11/4.91 ≈ **0.225**

Those are exactly the attention weights used in the earlier example. The most similar support example gets the biggest vote.

#### How this differs from plain metric learning

The paper notes the classifier is **discriminative**: it isn't enough for the query to be close to its own class. It must also be **far from the other classes in the support set**. Because the softmax compares against *all* support examples at once, the training objective lines up exactly with the real test task (multi-way one-shot classification). Related older methods are Neighbourhood Component Analysis (NCA), triplet loss and large-margin nearest neighbours, but those use pairwise comparisons rather than the whole set.

Also, the loss is **differentiable**, so the whole thing can be trained end-to-end with normal backpropagation.

### 5.4 Full Context Embeddings, FCE (Section 2.1.2)

The simple version above has a weakness: each g(xᵢ) is computed **independently**. The network looks at one support example at a time and has no idea what else is in the support set. Likewise f(x̂) ignores the support set.

The paper's fix: make the embeddings **context-aware**.

•         Replace g(xᵢ) with g(xᵢ, S). Now the embedding of each support example can change depending on the other examples. Why helpful? If two support examples are very similar to each other, the network can adjust their embeddings to make them easier to tell apart. This is done with a **bidirectional LSTM** that reads the support set as a sequence.

•         Replace f(x̂) with f(x̂, S). The query's embedding gets to "look at" the whole support set through an LSTM with **read-attention** over g(S), repeated for **K processing steps**. More steps = more "thinking depth." This also lets the model ignore irrelevant support items.

**Does it help?** On Omniglot it made no difference. On the harder miniImageNet it improved accuracy by about **2 percentage points**.

**Plain-English summary:** Basic Matching Nets compare the query to each example on its own. FCE lets the model first "read the whole cheat sheet," and also lets the query "re-read it several times," before deciding.

## 6. The Training Strategy (Section 2.2)

### 6.1 The Golden Rule

**Train the way you will be tested.**

At test time, the model sees a few examples of **new** classes and must classify. So during training, we **fake that situation over and over**, with different classes each time. This is meta-learning, "learning to learn from a support set."

### 6.2 What is an Episode?

To make one training step:

1.       **Sample a label set L** from a task distribution T. For instance, pick 5 random classes, such as {cats, dogs, ...}.

2.       **Sample a support set S** from L (e.g. 1 or 5 examples per chosen class).

3.       **Sample a batch B** from L (other examples of the same classes, not in S).

4.       Train the network to **predict the labels in B, given S**.

5.       Compute the gradient, update the weights, then go back to step 1 with a **new random label set**.

Typically there are up to \~5 classes with up to \~5 examples each, so a support set has roughly **5 to 25 examples**.

### 6.3 Equation 2: The Training Objective

θ = argmax_θ  E\_{L∼T} [ E\_{S∼L, B∼L} [ Σ\_{(x,y)∈B} log P_θ(y | x, S) ] ]

Read it from the inside out:

| Piece | Meaning |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| P_θ(y \| x, S) | Probability the model (with parameters θ) assigns to the **correct** label y for query x, given support set S. |
| log P_θ(...) | Log-probability. Higher = model is more confident in the right answer. (Maximizing log-probability is the same as minimizing cross-entropy loss.) |
| Σ over (x,y) ∈ B | Add this up over every example in the batch. |
| E\_{S∼L, B∼L} | Average over random choices of support set and batch for a given label set. |
| E\_{L∼T} | Average over random choices of label set (random tasks). |
| argmax_θ | Find the parameters θ that make all of the above as large as possible. |

**In one sentence:** *Find the network weights that, on average across many randomly generated mini-tasks, make the model assign the highest probability to the correct labels of the query examples.*

### 6.4 Why No Fine-Tuning Is Needed

After training with Equation 2, the weights θ encode a **general skill** ("compare and match"), not knowledge about specific classes. At test time you feed in a new support set S′ of unseen classes and the same network produces predictions. This is possible because of the non-parametric lookup in Equation 1.

**Caveat:** this only works if the new task distribution T′ is **similar in kind** to the training distribution T. If T′ is very different, the model fails. (The ImageNet "dogs" experiment below shows this.)

## 7. Experiments and Results

### 7.1 Setup

All experiments are **N-way k-shot** tasks on classes **never seen in training**. Chance performance = 1/N.

### 7.2 Omniglot (handwritten characters)

•         1,623 characters from 50 alphabets, 20 hand-drawn examples each. Often called the "transpose of MNIST" (many classes, few examples per class).

•         Embedding network: 4 blocks of *3×3 convolution (64 filters) → batch norm → ReLU → 2×2 max-pool*, on 28×28 images, giving a 64-dimensional embedding.

•         1,200 characters used for training; the rest for testing. Data augmented with 90° rotations.

| Model | 5-way 1-shot | 5-way 5-shot | 20-way 1-shot | 20-way 5-shot |
| ---------------------------- | --------------- | --------------- | ---------------- | ---------------- |
| Pixels (cosine) | 41.7% | 63.2% | 26.7% | 42.6% |
| Baseline classifier (cosine) | 80.0% | 95.0% | 69.5% | 89.1% |
| Convolutional Siamese Net | 96.7% | 98.4% | 88.0% | 96.5% |
| **Matching Nets (cosine)** | **98.1%** | **98.9%** | **93.8%** | **98.5%** |

Takeaways: Matching Nets win everywhere. Siamese nets are competitive with 5 examples but degrade rapidly in one-shot. The paper also tested on MNIST (10-way 1-shot) with a model trained only on Omniglot: Baseline ≈ 63%, Siamese ≈ 70%, **Matching Nets 72%**.

### 7.3 miniImageNet

The authors created **miniImageNet**: 100 ImageNet classes, 600 colour images each (84×84), 60,000 images total. 80 classes for training, 20 for testing. Harder than Omniglot but small enough for fast experiments.

| Model | 5-way 1-shot | 5-way 5-shot |
| ------------------------------- | --------------- | --------------- |
| Pixels | 23.0% | 26.6% |
| Baseline classifier (cosine) | 36.6% | 46.0% |
| Matching Nets (cosine) | 41.2% | 56.2% |
| **Matching Nets (cosine, FCE)** | **44.2%** | **57.0%** |
| Matching Nets (FCE + fine-tune) | 46.6% | 60.0% |

Here **FCE clearly helps**, adding about 2 points.

### 7.4 Full ImageNet (rand and dogs)

Two test setups:

•         **rand:** remove 118 random classes from training and test on them.

•         **dogs:** remove all 118 dog classes and test on them. This is harder because the classes are very similar (fine-grained).

They started the Matching Net's feature extractors from a pretrained **Inception** model, then trained on 5-way 1-shot tasks.

| Model (5-way 1-shot) | L_rand | L_dogs |
| ----------------------------------------- | --------- | ------ |
| Inception classifier | 87.6% | 59.8% |
| **Matching Nets (FCE)** | **93.2%** | 58.8% |
| Inception oracle (trained on all classes) | ≈ 99% | ≈ 99% |

•         On **rand**, Matching Nets improve by almost 6 points, roughly **halving the errors**.

•         On **dogs**, performance is slightly *worse* (−1%). The authors' explanation: during training, support sets were drawn from *random, diverse* classes, but dog classes are all similar to each other (fine-grained). Training and test distributions don't match. This is the "T′ must resemble T" caveat in action.

### 7.5 One-Shot Language Modeling (Penn Treebank)

A new task: given a sentence with a blanked-out word, and a support set of 5 sentences each with a *different* blanked-out word (labelled by which word is missing), choose which label fits the query sentence best. Example: the query about currency ("...the \_\_\_ was quoted at N marks down...") should match the support sentence whose missing word is "dollar."

•         5-way choice, so chance = **20%**.

•         Matching Nets (simple encoding) with k = 1, 2, 3 examples per class: **32.4%, 36.1%, 38.2%**.

•         An LSTM language model that has seen *all* the data (an unfair "oracle" upper bound): **72.8%**.

So it beats chance clearly but is far from the oracle; the authors suggest combining parametric LSTM language models with non-parametric components in future work.

## 8. Limitations the Authors Admit

1.       **Computation grows with support set size.** Every query is compared to every support example, so larger support sets make each gradient update more expensive.

2.       **Struggles when labels are biased or fine-grained** (the dogs experiment).

3.       **Only works if test tasks resemble training tasks.** The model learned "how to match" within a certain style of task.

4.       **Language results lag far behind** the oracle model.

## 9. Appendix Equations Explained (FCE in Detail)

These are the more technical equations for Full Context Embeddings. They can look scary but follow the same ideas you already know.

### 9.1 The Query Embedding f(x̂, S)

The query goes through an LSTM for K "processing steps." At each step k:

**Eq. 3:** ĥₖ, cₖ = LSTM( f′(x̂), [hₖ₋₁, rₖ₋₁], cₖ₋₁ )

*The LSTM takes the raw query features f′(x̂) as input, plus its previous hidden state hₖ₋₁ concatenated with a "read-out" rₖ₋₁ from the support set, plus its previous cell memory cₖ₋₁. It outputs a new hidden state ĥₖ and cell state cₖ.*

**Eq. 4:** hₖ = ĥₖ + f′(x̂)

*A **skip connection**: add the original features back in. This helps keep the original information from being lost and makes training easier.*

**Eq. 5:** rₖ₋₁ = Σ (i = 1 to |S|) a( hₖ₋₁, g(xᵢ) ) · g(xᵢ)

*The read-out is an **attention-weighted average of the support-set embeddings**. Same structure as Equation 1: weights times items, summed.*

**Eq. 6:** a( hₖ₋₁, g(xᵢ) ) = softmax( hₖ₋₁ᵀ · g(xᵢ) )

*The attention weights come from a softmax over **dot products** between the current hidden state and each support embedding. A bigger dot product means "more relevant," so more attention.*

After K steps, the final embedding is f(x̂, S) = h_K.

**Intuition:** At every step the query "asks" the support set a question (the attention), reads back an answer (r), updates its understanding (the LSTM), and repeats. After K rounds of this, the query embedding has been refined using the whole support set.

### 9.2 The Support Set Embedding g(xᵢ, S)

g(xᵢ, S) = h→ᵢ + h←ᵢ + g′(xᵢ)

where

•         h→ᵢ, c→ᵢ = LSTM( g′(xᵢ), h→ᵢ₋₁, c→ᵢ₋₁ ) (reads the set **left to right**)

•         h←ᵢ, c←ᵢ = LSTM( g′(xᵢ), h←ᵢ₊₁, c←ᵢ₊₁ ) (reads the set **right to left**)

**Intuition:** A *bidirectional* LSTM lets each item see everything before it **and** everything after it. The final embedding adds the forward state, the backward state, and the original features g′(xᵢ) (another skip connection). So each support example's embedding now reflects its neighbours in the set.

(Note: sets have no natural order. The LSTM treats the support set as a sequence only as a practical tool, and the approach follows the paper "Order Matters" for handling sets.)

## 10. Quick Concept Map

•         **Goal:** classify new classes from 1 (or a few) examples, with no retraining.

•         **Model:** ŷ = Σ a(x̂, xᵢ) yᵢ, a similarity-weighted vote over the support set.

•         **Similarity:** softmax of cosine similarity between learned embeddings.

•         **Upgrade (FCE):** embeddings that are aware of the whole support set (bi-LSTM for support items, attention-LSTM for the query).

•         **Training:** episodic. Fake many small N-way k-shot tasks, maximize log-probability of the correct labels (Eq. 2).

•         **Why it works:** a learned metric (neural net) plus a non-parametric lookup (instant new classes).

•         **Weakness:** cost grows with support size; fails if test tasks differ greatly from training tasks.

## 11. Self-Test Questions

1.       What is the difference between a parametric and a non-parametric model? Which is Matching Networks closer to, and why is it actually a hybrid?

2.       In a 5-way 1-shot task, how many labelled examples are in the support set? What accuracy does random guessing give?

3.       In Equation 1, what does each symbol (ŷ, a, xᵢ, yᵢ) mean? What happens if the labels yᵢ are one-hot?

4.       Why is a softmax used in the attention kernel? What does it guarantee about the weights?

5.       Why does cosine similarity ignore the *length* of vectors?

6.       What does it mean to "train the way you will be tested"? Describe one training episode step by step.

7.       What problem do Full Context Embeddings solve? On which dataset did they help?

8.       Why did Matching Nets perform slightly worse than the Inception baseline on the dogs split?

9.       Name two limitations of Matching Networks.

### Answer Hints

1.       Parametric = knowledge in weights; non-parametric = knowledge in stored data. Matching Nets learn weights (the embeddings) but classify via a lookup over the support set.

2.       5 examples; 20%.

3.       ŷ = predicted label distribution, a = attention weight, xᵢ/yᵢ = support example and its label. With one-hot labels, the output is the total attention given to each class.

4.       Softmax makes all weights positive and sum to 1, so they act like probabilities.

5.       Cosine only measures the angle between vectors, so scale doesn't matter.

6.       See Section 6.2.

7.       Each embedding was independent of the rest of the support set; FCE adds context. It helped on miniImageNet (\~2 points), not Omniglot.

8.       Training support sets were random and diverse, but dog classes are fine-grained and similar, a train/test distribution mismatch.

9.       Computation grows with support set size; struggles with fine-grained or biased label distributions; needs test tasks similar to training tasks.
