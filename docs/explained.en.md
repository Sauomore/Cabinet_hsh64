# HSH-64 Design Document

This document explains the HSH-64 structure and the design of that structure in plain language. The original idea behind the design was whether a string of numbers could encode semantics; what is explained here is the final version.
In earlier versions (see GitHub for details), the project used the HSH-20 and HSH-32 designs, and was ultimately iterated into $HSH-64$. Let us now start walking through it.

---

## 1. Vocabulary and Two Sets of Embeddings

Suppose for now that we have a fixed vocabulary:

$$
\mathcal{X} = \{ \mathbf{x}_1, \ldots, \mathbf{x}_N \}
$$

- $\mathcal{X}$: the vocabulary set;
- $N$: the total number of items in the vocabulary;
- $\mathbf{x}_i$: the student embedding of the $i$-th item.

It contains $N$ items (in the HSH-64 experiments, device constraints meant only a fairly small vocabulary was used, $N=3{,}109$). Each item $i$ corresponds to a $D$-dimensional continuous vector:

$$
\mathbf{x}_i \in \mathbb{R}^D
$$

- $\mathbf{x}_i$: the student embedding of the $i$-th item;
- $\mathbb{R}^D$: $D$-dimensional real space;
- $D$: the dimension of the student embedding ($D=512$ in bge-small).

**Meaning**: $\mathbf{x}_i$ is the vector emitted by a lightweight encoder (ultimately **bge-small**), of dimension $D=512$. It is small and fast, and serves as the input for actual online retrieval.

In addition, we have a set of "teacher" embeddings:

$$
\mathcal{T} = \{ \mathbf{t}_1, \ldots, \mathbf{t}_N \}
$$

- $\mathcal{T}$: the teacher embedding set;
- $\mathbf{t}_i$: the teacher embedding of the $i$-th item;
- $N$: the total number of items in the vocabulary (in one-to-one correspondence with the student set).

**Meaning**: $\mathbf{t}_i$ is produced by a stronger model (ultimately **bge-large**), and is used specifically to provide a high-precision "ground truth" for verifying accuracy during subsequent training. It is used only in the offline stage.

| Symbol | Source model | Role | Stage used |
|---|---|---|---|
| $\mathbf{x}_i$ | bge-small | Student vector, lightweight | Online inference |
| $\mathbf{t}_i$ | bge-large | Teacher vector, high precision | Offline training/evaluation |

---

## 2. Teacher Similarity Matrix

The teacher vectors are used to compute the cosine similarity between item $i$ and item $j$:

$$
s_{ij} = \frac{\mathbf{t}_i^\top \mathbf{t}_j}{\|\mathbf{t}_i\| \|\mathbf{t}_j\|}
$$

- $s_{ij}$: the cosine similarity between item $i$ and item $j$;
- $\mathbf{t}_i, \mathbf{t}_j$: the teacher embeddings of item $i$ and item $j$;
- $\mathbf{t}_i^\top \mathbf{t}_j$: the dot product of the two teacher embeddings;
- $\|\mathbf{t}_i\|$: the L2 norm (length) of vector $\mathbf{t}_i$.

**Why cosine similarity?**

Because it only cares whether the two vectors point in the same "direction" and ignores differences in length, which makes it suitable for measuring semantic similarity.

**Offline role**: In the offline stage, bge-large is used to compute the pairwise cosine similarity between all words in the vocabulary $\mathcal{X}$, stored as an $N \times N$ similarity matrix $S$:

$$
S = \begin{bmatrix}
s_{11} & s_{12} & \cdots & s_{1N} \\
s_{21} & s_{22} & \cdots & s_{2N} \\
\vdots  & \vdots  & \ddots & \vdots  \\
s_{N1} & s_{N2} & \cdots & s_{NN}
\end{bmatrix}
$$

- $S$: the $N \times N$ teacher cosine similarity matrix;
- $s_{ij}$: the element in row $i$, column $j$ of the matrix, i.e. the cosine similarity between item $i$ and item $j$;
- $N$: the vocabulary size.

Matrix $S$ later has two important roles:
1. Defining the "true top-$K$ neighbours" of each item;
2. Providing a supervision signal for the student model during training (for example, the distillation loss).


---

## 3. True top-$K$ Neighbour Set (Ground Truth)
Here we define the set of ground-truth answers.

The true top-$K$ neighbour set of item $i$ is denoted:

$$
\mathcal{G}_i^{(K)} = \argmax_{\mathcal{J} \subseteq [N] \setminus \{i\}, \, |\mathcal{J}|=K} \sum_{j \in \mathcal{J}} s_{ij}
$$

- $\mathcal{G}_i^{(K)}$: the true top-$K$ neighbour set of item $i$ (ground truth);
- $[N]$: the vocabulary index set $\{1, 2, \ldots, N\}$;
- $\mathcal{J}$: a subset of $K$ indices chosen from $[N]\setminus\{i\}$;
- $|\mathcal{J}|$: the number of elements in set $\mathcal{J}$;
- $K$: the size of the neighbour set;
- $s_{ij}$: the cosine similarity between item $i$ and item $j$;
- $\argmax$: take the $\mathcal{J}$ that maximises the expression that follows.

### 3.1 Symbol Breakdown

- $[N] = \{1, 2, \ldots, N\}$: the numbers of all words in the vocabulary.
- $[N] \setminus \{i\}$: exclude the query word $i$ itself — a word cannot count as its own neighbour.
- $\mathcal{J} \subseteq [N] \setminus \{i\}$: pick one subset (candidate combination) from the remaining $N-1$ words.
- $|\mathcal{J}| = K$: this subset must contain exactly $K$ words.
- $\sum_{j \in \mathcal{J}} s_{ij}$: add up the similarities of these $K$ words to $i$ to obtain a "total score".
- $\argmax$: find the combination that makes the total score the largest.

### 3.2 Intuition

What it actually does is very simple:

> Pick out the $K$ words most similar to item $i$.

Because all similarities $s_{ij}$ are non-negative (cosine similarity lies in $[-1, 1]$, but semantically related words are usually positive), the combination that maximises the sum is simply the top $K$ after sorting $s_{ij}$ from high to low.

### 3.3 Example: the top-10 neighbours of "apple"

Suppose the vocabulary also contains words such as "fruit", "Red Fuji", "banana", "phone" and "car", and bge-large computes their similarities to "apple" as follows:

| Candidate word | Similarity to "apple" $s_{ij}$ |
|---|---|
| fruit | 0.95 |
| Red Fuji | 0.90 |
| banana | 0.85 |
| phone | 0.30 |
| car | 0.10 |
| … | … |

If $K=10$, then $\mathcal{G}_{\text{苹果}}^{(10)}$ is the 10 words with the highest similarity, for example:

$$
\mathcal{G}_{\text{苹果}}^{(10)} = \{\text{水果}, \text{红富士}, \text{香蕉}, \ldots\}
$$

At this point the set $$\mathcal{G}_{\text{苹果}}^{(10)}$$ contains 10 elements, and this is the **ground-truth set** of the item "apple".

---

## 4. Hash Retrieval and Recall@$K$

The semantic hash function $h$ maps the lightweight student vector $\mathbf{x}_i$ to a binary code:

$$
h: \mathbb{R}^D \to \{0,1\}^M
$$

- $h$: the semantic hash function;
- $\mathbb{R}^D$: the $D$-dimensional continuous embedding space (where the student embeddings live);
- $\{0,1\}^M$: the $M$-dimensional binary code space;
- $M$: the binary code length ($M=52$ in HSH-64).

In HSH-64, $M=52$ is the semantic similarity code length, ultimately packed into a 64-bit integer.

The retrieval function returns the top-$K$ items with the smallest Hamming distance to the query code:

$$
\mathcal{R}_i^{(K)} = \argmin_{\mathcal{J} \subseteq [N] \setminus \{i\}, \, |\mathcal{J}|=K} \sum_{j \in \mathcal{J}} d_H\bigl(h(\mathbf{x}_i), h(\mathbf{x}_j)\bigr)
$$

- $\mathcal{R}_i^{(K)}$: the top-$K$ result set returned by the HSH-64 system for item $i$;
- $\mathcal{J}$: the candidate top-$K$ index subset;
- $K$: the number of returned results;
- $d_H(\cdot, \cdot)$: the Hamming distance function;
- $h(\mathbf{x}_i)$: the hash code of item $i$;
- $h(\mathbf{x}_j)$: the hash code of item $j$;
- here $i$ is the word being queried, and $j$ is another word compared against $i$;
- $\argmin$: take the $\mathcal{J}$ that minimises the total Hamming distance, i.e. the most similar one.

**Meaning**: $\mathcal{R}_i^{(K)}$ is the result set actually returned by the HSH-64 system.

### 4.1 Recall@$K$ Definition

The optimisation objective of the entire system is to maximise Recall@$K$:

$$
\text{Recall@}K = \frac{1}{N} \sum_{i=1}^{N} \frac{|\mathcal{R}_i^{(K)} \cap \mathcal{G}_i^{(K)}|}{K}
$$

- $\text{Recall@}K$: the average recall over the first $K$ results;
- $N$: the total number of items in the vocabulary;
- $\mathcal{R}_i^{(K)}$: the top-$K$ result set returned by the $HSH$ system;
- $\mathcal{G}_i^{(K)}$: the true top-$K$ neighbour set given by the teacher;
- $\cap$: the intersection of the two sets;
- $|\cdot|$: the number of elements in a set;
- $K$: the number of returned results.

### 4.2 Symbol Breakdown

- $\mathcal{R}_i^{(K)}$: the top-$K$ results returned by the hash model.
- $\mathcal{G}_i^{(K)}$: the ground-truth top-$K$ given by bge-large.
- $\mathcal{R}_i^{(K)} \cap \mathcal{G}_i^{(K)}$: the intersection of the two sets, i.e. the items that "hit the correct answer".
- $|\cdot|$: the number of elements in the intersection.
- $\frac{|\cap|}{K}$: turns the hit count into a ratio. For example, with $K=10$ and 7 hits, the Recall@10 of that query is 0.7.
- $\frac{1}{N} \sum_{i=1}^{N}$: averages over all $N$ query words, avoiding the situation where only the words that happen to perform well are counted.

### 4.3 Intuition

The question Recall@$K$ answers is:

> Of the first $K$ results returned by the system, how many are genuinely the semantically most relevant words?

The value range is $[0, 1]$, and the closer to 1 the better. It is the most central evaluation metric of HSH-64.

---

## 5. Summary

At this point we have fully defined the problem once:

1. **Input**: a vocabulary of $N$ words, together with each word's lightweight student vector $\mathbf{x}_i$ and strong teacher vector $\mathbf{t}_i$.
2. **Ground truth**: compute cosine similarity with the teacher vectors, and define the true top-$K$ neighbour set $\mathcal{G}_i^{(K)}$ for each word.
3. **Model output**: the hash function $h$ maps student vectors to binary codes, and the top-$K$ set $\mathcal{R}_i^{(K)}$ is returned by Hamming distance.
4. **Optimisation objective**: maximise Recall@$K$, i.e. make $\mathcal{R}_i^{(K)}$ and $\mathcal{G}_i^{(K)}$ overlap as much as possible.

We can now move on to the HSH-64 encoding structure, the training pipeline and the retrieval algorithm.

---

## 6. HSH-64 Encoding Structure

HSH-64 compresses the entire semantic information into a single 64-bit unsigned integer:

$$
c = \underbrace{(f_3 f_2 f_1 f_0)}_{\text{feat}: 4} \,\|\, \underbrace{(s_{51} \cdots s_1 s_0)}_{\text{sim}: 52} \,\|\, \underbrace{(a_7 \cdots a_1 a_0)}_{\text{abs}: 8}
$$

- $c$: one HSH-64 code (a 64-bit unsigned integer);
- $f_3 f_2 f_1 f_0$: the 4 bits of the feat field;
- $s_{51} \cdots s_1 s_0$: the 52 bits of the sim field;
- $a_7 \cdots a_1 a_0$: the 8 bits of the abs field;
- $\|$: the bit concatenation operator.

The three fields each have their own job:

| Field | Bits | Role | Example/capacity |
|---|---|---|---|
| **feat** | 4 | part-of-speech / coarse-grained category label | $2^4 = 16$ coarse classes |
| **sim** | 52 | core semantic similarity code | $2^{52} \approx 4.5 \times 10^{15}$ states |
| **abs** | 8 | perfect-hash identifier within a cluster | $2^8 = 256$ unique IDs |
| **Total** | 64 | one `u64`, one `POPCNT` | only 8 bytes per item |

The distance between two codes $c$ and $c'$ is computed with a single hardware popcount:

$$
d_H(c, c') = popcnt(c \oplus c')
$$

- $d_H(c, c')$: the Hamming distance between code $c$ and code $c'$;
- $popcnt(\cdot)$: the CPU popcount instruction, which counts the number of 1s in the binary representation;
- $\oplus$: the bitwise XOR operation;
- $c, c'$: the two HSH-64 codes to be compared.

### 6.1 The abs Field: Tie-Breaking Mechanism

`abs` is not a semantic signal but a **tie-breaker**.

Imagine two words that share the same `(feat, sim)` prefix and differ only in `abs`: they differ by at most 8 bits and still lie within a tight Hamming neighbourhood. Two words with different `sim` usually differ by more bits. Therefore:

- `abs` determines the ordering within a cluster;
- `abs` does not dominate cross-cluster semantic distance;
- `abs` can also be treated as a hard filter: require an exact `(feat, sim)` match before looking at `abs`, which realises a perfect-hash lookup within the bucket.

---

## 7. Deep Hash MLP Architecture

HSH-64 uses a tiny MLP to map bge-small's 512-dimensional vector into a 52-bit semantic code.

### 7.1 Forward Pass

1. **Centering**: subtract the training-set mean from the input
   $$
   \mathbf{x}' = \mathbf{x} - \boldsymbol{\mu}
   $$
   - $\mathbf{x}$: the original student embedding;
   - $\boldsymbol{\mu}$: the mean vector of the student embeddings over the training set;
   - $\mathbf{x}'$: the centred student embedding.

2. **MLP projection**: a single hidden layer with ReLU
   $$
   \mathbf{u} = \mathbf{W}_2 \, \sigma(\mathbf{W}_1 \mathbf{x}' + \mathbf{h}_1) + \mathbf{h}_2
   $$
   - $\mathbf{u}$: the continuous logits vector;
   - $\mathbf{W}_1$: the weight matrix from the input layer to the hidden layer;
   - $\mathbf{h}_1$: the hidden-layer bias;
   - $\sigma$: the ReLU activation function, short for Rectified Linear Unit;
   - $\mathbf{W}_2$: the weight matrix from the hidden layer to the output layer;
   - $\mathbf{h}_2$: the output-layer bias.
   - The hidden layer first performs a linear transform $\mathbf{W}_1 \mathbf{x}' + \mathbf{h}_1$ , and then applies ReLU. Its purpose is:

     - To introduce non-linearity: without an activation function, no matter how many layers are stacked, the neural network is equivalent to a linear transform, and its expressive power is very weak;
     - To keep computation simple: it only needs to test the sign, with no complicated exponential/trigonometric operations;
     - To alleviate vanishing gradients: in the positive region the gradient is always 1, so gradients are not easily attenuated during backpropagation.
     - Put simply, ReLU is "turn negative numbers into 0 and leave positive numbers unchanged".

   - There is a common misunderstanding here: the hidden layer includes $\sigma(\mathbf{W}_1 \mathbf{x}' + \mathbf{h}_1)$ ; in this expression $\sigma$ takes effect only on the hidden layer, while the weights $\mathbf{W}_2$ , as well as the bias $\mathbf{h}_2$ , are not restricted, which is to say that the final value range of $\mathbf{u}$ can be positive or negative, and this does not affect the input to the subsequent sign quantisation

3. **Sign quantisation**: turn continuous logits into a binary code
   $$
   \mathbf{b} = \frac{1 + \operatorname{sign}(\mathbf{u})}{2} \in \{0, 1\}^{52}
   $$
   - $\mathbf{b}$: the quantised 52-bit binary code;
   - $\operatorname{sign}(\mathbf{u})$: the sign function, outputting $+1$ for a positive number and $-1$ for a negative number;
   - $\{0,1\}^{52}$: the 52-dimensional binary vector space.

Here $\mathbf{u} \in \mathbb{R}^{52}$ is called logits, and $\mathbf{b} \in \{0,1\}^{52}$ is the final sim code.

### 7.2 Replaceable Output Layer

The hidden layers of the MLP are shared, but the output dimension $O$ changes with the stage:

- $O$: the dimension of the MLP output layer;
- **Stage 1** (teacher distillation): $O = 1024$, matching the output dimension of bge-large;
- **Stage 2** (hash training): $O = 52$, generating the final semantic code.

Stage 2 directly reuses the hidden-layer weights trained in stage 1 and replaces only the last layer, thereby leveraging the continuous projection learned in stage 1 for initialisation.

### 7.3 Straight-Through Estimator (STE)

Because the sign function $\operatorname{sign}(\cdot)$ has a gradient of zero almost everywhere, it cannot be backpropagated directly. The STE approach is:

- **Forward propagation**: use the real sign quantisation;
- **Backward propagation**: approximate the gradient as the identity mapping.

$$
\frac{\partial \mathbf{b}}{\partial \mathbf{u}} \approx \mathbf{I}
$$

- $\frac{\partial \mathbf{b}}{\partial \mathbf{u}}$: the Jacobian matrix of the binary code $\mathbf{b}$ with respect to the logits $\mathbf{u}$;
- $\mathbf{I}$: the identity matrix;
- meaning: during backpropagation, the gradient of the sign function is approximated as the identity mapping.

Aside: the definition of the Jacobian matrix. If there is a function b=f(u), whose input is a vector u and whose output is also a vector b, then the Jacobian matrix arranges all "partial derivatives of the outputs with respect to the inputs" into a matrix:
$$
\frac{\partial \mathbf{b}}{\partial \mathbf{u}} =
\begin{bmatrix}
\frac{\partial b_1}{\partial u_1} & \frac{\partial b_1}{\partial u_2} & \cdots & \frac{\partial b_1}{\partial u_{52}} \\
\frac{\partial b_2}{\partial u_1} & \frac{\partial b_2}{\partial u_2} & \cdots & \frac{\partial b_2}{\partial u_{52}} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial b_{52}}{\partial u_1} & \frac{\partial b_{52}}{\partial u_2} & \cdots & \frac{\partial b_{52}}{\partial u_{52}}
\end{bmatrix}
$$

Here:
$$
\mathbf{b} = \frac{1 + \operatorname{sign}(\mathbf{u})}{2}
$$

Each $\mathbf{b}_j$ depends only on the corresponding $\mathbf{u}_j$ :
when $\mathbf{u}_j$ < 0, $\mathbf{b}_j$ = 0 , and when $\mathbf{u}_j$ > 0, $\mathbf{b}_j$ = 1
so in theory:
$$
\frac{\partial \mathbf{b}_j}{\partial \mathbf{u}_j} = \mathbf{0}
$$
This means the true Jacobian matrix is a zero matrix almost everywhere.

But here, $STE$ approximates it as an identity matrix, because a zero matrix cannot backpropagate gradients.

So the meaning of the identity matrix $\mathbf{I}$ is: each $\mathbf{b}_j$ is affected only by the corresponding $\mathbf{u}_j$ , with an influence of 1, that is:
$$
\frac{\partial \mathbf{b}_j}{\partial \mathbf{u}_k} \approx \mathbf{0} (\mathcal{j} \ne \mathcal{k})
 , 
\frac{\partial \mathbf{b}_j}{\partial \mathbf{u}_j} \approx \mathbf{1} 
$$

In this way, the gradient of the loss with respect to the logits can be passed directly back to the MLP weights, enabling end-to-end training.


---

## 8. Three-Stage Training Pipeline

HSH-64 training is divided into three stages, which progressively distil "continuous teacher knowledge" into "high-quality binary codes".

### 8.1 Stage 1: Continuous Pre-training

**Goal**: make the MLP output as close as possible to bge-large's teacher vectors.

$$
\mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^{N} \| f_\theta(\mathbf{x}_i) - \mathbf{t}_i \|_2^2
$$

- $\mathcal{L}_{\text{MSE}}$: the mean squared error loss;
- $N$: the vocabulary size;
- $f_\theta(\cdot)$: the MLP function with parameters $\theta$;
- $\mathbf{x}_i$: the student embedding of the $i$-th item;
- $\mathbf{t}_i$: the teacher embedding of the $i$-th item;
- $\|\cdot\|_2$: the L2 norm, i.e. the modulus length of the vector; here what is computed is the straight-line distance (Euclidean distance) between the internal elements.


The 1024-dimensional vectors of bge-large contain a rich semantic structure. Distilling with MSE first is equivalent to letting the student model learn a high-quality continuous projection, laying a good foundation for the subsequent binarisation.

### 8.2 Stage 2: Discrete Fine-Tuning

Replace the output layer with 52 hyperplanes and train end-to-end with STE. The total loss consists of four terms:

$$
\mathcal{L} = \mathcal{L}_{\text{info}} + \lambda_q \mathcal{L}_{\text{quant}} + \lambda_b \mathcal{L}_{\text{balance}} + \lambda_p \mathcal{L}_{\text{pair}}
$$

- $\mathcal{L}$: the total loss of stage 2;
- $\mathcal{L}_{\text{info}}$: the InfoNCE loss;
- $\mathcal{L}_{\text{quant}}$: the quantisation loss;
- $\mathcal{L}_{\text{balance}}$: the balance loss;
- $\mathcal{L}_{\text{pair}}$: the pairwise loss;
- $\lambda_q, \lambda_b, \lambda_p$: the weight hyperparameters of the corresponding losses

#### 8.2.1 InfoNCE Loss

$$
\mathcal{L}_{\text{info}} = -\frac{1}{|\mathcal{P}^+|}\sum_{(i,j) \in \mathcal{P}^+} \log \frac{e^{\mathbf{u}_i^\top \mathbf{u}_j / \tau}}{\sum_{k} e^{\mathbf{u}_i^\top \mathbf{u}_k / \tau}}
$$

- $\mathcal{L}_{\text{info}}$: the InfoNCE loss;
- $\mathcal{P}^+$: the set of positive pairs (teacher top-$P$ neighbours);
- $|\mathcal{P}^+|$: the number of positive pairs;
- $(i,j)$: a positive pair;
- $\mathbf{u}_i, \mathbf{u}_j$: the continuous logits of item $i$ and item $j$;
- $\tau$: the temperature hyperparameter;
- $k$: the index that traverses all $N$ vocabulary items in the denominator.

Effect: bring semantically similar words closer together in logits space, and push dissimilar words further apart.

#### 8.2.2 Quantisation Loss

$$
\mathcal{L}_{\text{quant}} = \frac{1}{N} \sum_{i=1}^{N} \| \mathbf{u}_i - \mathbf{b}_i \|_1
$$

- $\mathcal{L}_{\text{quant}}$: the quantisation loss;
- $N$: the vocabulary size;
- $\mathbf{u}_i$: the continuous logits of item $i$;
- $\mathbf{b}_i$: the quantised binary code of item $i$;
- $\|\cdot\|_1$: the L1 norm (sum of absolute values).

Effect: encourage the logits to be as close to ±1 as possible, so that the information loss when quantising into binary codes is small.

#### 8.2.3 Balance Loss

$$
\mathcal{L}_{\text{balance}} = \frac{1}{M} \sum_{j=1}^{M} \left( \bar{b}_j - 0.5 \right)^2
$$

- $\mathcal{L}_{\text{balance}}$: the balance loss;
- $M$: the binary code length ($M=52$);
- $j$: the $j$-th bit position of the code;
- $\bar{b}_j$: the empirical mean of the $j$-th bit over all items;
- $0.5$: the ideal balance value (half 0, half 1).

Effect: prevent bit collapse. Ideally, each bit should be roughly half 0 and half 1 across all items, which is what maximises the information capacity.

#### 8.2.4 Pairwise Loss

$$
\mathcal{L}_{\text{pair}} = \frac{1}{|\mathcal{P}|} \sum_{(i,j) \in \mathcal{P}} \left( \mathbf{u}_i^\top \mathbf{u}_j - \mathbf{t}_i^\top \mathbf{t}_j \right)^2
$$

- $\mathcal{L}_{\text{pair}}$: the pairwise loss;
- $\mathcal{P}$: the set of sampled pairs;
- $|\mathcal{P}|$: the number of pairs;
- $\mathbf{u}_i, \mathbf{u}_j$: the student logits (after L2 normalisation);
- $\mathbf{t}_i, \mathbf{t}_j$: the teacher embeddings (after L2 normalisation);
- $\mathbf{u}_i^\top \mathbf{u}_j$: the inner product of the student logits;
- $\mathbf{t}_i^\top \mathbf{t}_j$: the inner product of the teacher embeddings.

Note: before computation, both $\mathbf{u}_i$ and $\mathbf{t}_i$ are first L2-normalised, so that the inner product is equivalent to cosine similarity.

Effect: align the similarity structure among the student logits directly with the teacher embeddings.

### 8.3 Stage 3: Recall-Oriented Post-Processing

Once MLP training is complete, the model weights are frozen and the binary codes themselves are optimised directly, with the goal of improving Recall@$K$.

#### 8.3.1 Positive and Negative Samples

For each query $i$:

- **Positive sample set** $\mathcal{P}_i^+$: the indices of the teacher top-$P$ neighbours;
- **Negative sample set** $\mathcal{P}_i^-$: the items that are in the current Hamming top-$K$ but not in the positive sample set.

#### 8.3.2 Pair Weights

$$
W_{ij} =
\begin{cases}
+w_{\text{pos}}, & j \in \mathcal{P}_i^+ \\
-w_{\text{neg}}, & j \in \mathcal{P}_i^- \\
0, & \text{其他}
\end{cases}
$$

- $W_{ij}$: the recall-oriented weight between item $i$ and item $j$;
- $w_{\text{pos}}$: the positive sample weight (hyperparameter);
- $w_{\text{neg}}$: the negative sample weight (hyperparameter);
- $\mathcal{P}_i^+$: the positive sample set of item $i$ (teacher top-$P$ neighbours);
- $\mathcal{P}_i^-$: the negative sample set of item $i$ (non-positive samples within the Hamming top-$K$).

#### 8.3.3 Objective Function

$$
\mathcal{O}(C) = \sum_{i,j} W_{ij} \cdot d_H(\mathbf{c}_i, \mathbf{c}_j)
$$

- $\mathcal{O}(C)$: the recall-oriented objective function, whose input is the entire code matrix $C$;
- $C$: the $N \times M$ binary code matrix;
- $W_{ij}$: the recall-oriented weight of the item pair $(i,j)$;
- $d_H(\mathbf{c}_i, \mathbf{c}_j)$: the Hamming distance between item $i$ and item $j$;
- $\mathbf{c}_i, \mathbf{c}_j$: the binary codes of item $i$ and item $j$.

**Intuition**:

- For positive samples ($W_{ij} > 0$): the smaller the Hamming distance, the smaller the objective function → pull them closer together;
- For negative samples ($W_{ij} < 0$): the larger the Hamming distance, the smaller the objective function → push them further apart.

#### 8.3.4 Greedy Bit Flipping

The algorithm iteratively tries flipping each bit of each code, and selects the flip that makes $\mathcal{O}(C)$ decrease the most, until there is no improvement. This process converges to a 1-optimal state, i.e. no single-bit flip can further reduce the objective function.

---

## 9. Asymmetric Distance Scoring

Sign quantisation discards the magnitude information of the continuous logits. To mitigate this, HSH-64 uses **asymmetric scoring** during re-ranking:

$$
s_{\text{asym}}(\mathbf{u}_q, \mathbf{b}_d) = \sum_{j=1}^{M} u_{q,j} \cdot (2 b_{d,j} - 1)
$$

- $s_{\text{asym}}$: the asymmetric similarity score;
- $\mathbf{u}_q$: the continuous logits of the query item (computed online);
- $\mathbf{b}_d$: the binary code of the database item (precomputed);
- $u_{q,j}$: the $j$-th dimension of the query logits;
- $b_{d,j}$: the $j$-th bit of the database code;
- $M$: the code length ($M=52$).

In this way, bits with high confidence in the query projection receive larger weights, providing finer-grained ranking than pure Hamming distance. The online cost is only one dot product per candidate.

---

## 10. Adaptive Multi-Index Hashing (MIH)

To make online retrieval fast enough, HSH-64 uses MIH to split the 52-bit sim code into multiple segments and build an inverted index for each segment.

### 10.1 Segmentation and Indexing

- The 52-bit sim code is split into $B=13$ segments;
- Each segment is 4 bits, giving $2^4 = 16$ possible segment codes;
- Each segment corresponds to an inverted index, which records which items take that segment code on that segment.

### 10.2 Query Process

Given a query code $\mathbf{q}$:

- $\mathbf{q}$: the HSH-64 code of the query item;
- $\mathbf{q}_b$: the $b$-th segment of the query code ($b=1,\ldots,B$);
- $r$: the current total Hamming radius;
- $B$: the total number of segments ($B=13$);
- $\lfloor r/B \rfloor$: the maximum Hamming radius allowed per segment (rounded down).

Query steps:

1. For each segment $\mathbf{q}_b$, enumerate all segment codes within the Hamming radius $\lfloor r/B \rfloor$;
2. Look up the inverted index and collect candidate item IDs;
3. Merge the candidate lists of all segments and deduplicate.

### 10.3 Adaptive Radius Expansion

Starting from $r=0$, gradually expand the radius until the following are satisfied:

- $r$: the current total Hamming radius;
- $K$: the final number of top-$K$ results needed;
- $\alpha$: the coarse-ranking factor (default $\alpha=10$);
- $\max(K, \alpha K)$: the minimum target size of the candidate pool;
- $\delta$: the saturation threshold (for example 10%), which controls when radius expansion stops.

Stopping conditions:

- The candidate pool size is at least $\max(K, \alpha K)$ (guaranteeing recall);
- The proportion of newly added candidates is below the threshold $\delta$ (indicating that further radius expansion yields very little).

With the default coarse-ranking factor $\alpha = 10$, about 100 candidates are first recalled, and then the top-10 are taken by re-ranking.

---

## 11. Two-Stage Retrieval Architecture

The online retrieval of HSH-64 is divided into two stages:

### 11.1 Coarse Ranking (Required)

1. The query word is encoded by bge-small into a 512-dimensional vector;
2. A small MLP projects it to generate an HSH-64 code;
3. MIH quickly recalls a candidate pool (about 187 items).

The online stack is very small:

- bge-small: about 600 MB (HuggingFace format) or a smaller quantised version;
- MLP projector: 585 KB minimum;
- codebook: 25 KB;
- MIH index: 81 KB.

### 11.2 Re-ranking (Optional)

For the small number of candidates recalled by coarse ranking, bge-large is used to compute the exact cosine similarity and re-rank them. Because the candidate pool is very small, the cost of this step is controllable.

- MIH coarse ranking only: latency about 0.54 ms;
- MIH + bge-large re-ranking: latency about 3.21 ms;
- bge-large re-ranking can be omitted, which suits deployment on edge devices.

---

## 12. Model Ensemble

To further improve recall without increasing online latency, HSH-64 trains multiple Deep Hash models with different random seeds and hidden dimensions. At retrieval time the candidate sets of the models are merged, and then fused using the following strategies:

- **Frequency**: candidates recalled by more models rank first;
- **Average rank**: candidates with a better average rank across the models rank first;
- **Minimum Hamming distance**: candidates with the smallest distance in any model rank first.

Experiments show that a small-scale ensemble of four models improves Recall@10 by about 2–3 percentage points over a single model.

---

## 13. Overview of Experimental Results

The main results on a 3,109-word Chinese vocabulary:

| Configuration | Recall@10 | Notes |
|---|---|---|
| h256 single model | 0.7382 | lightweight version, MLP only 585 KB |
| h512 single model | 0.7404 | best single model, MLP 1.17 MB |
| Four-model ensemble | 0.7724 | candidate set merging + re-ranking |
| MIH coarse ranking + bge-large re-ranking (four-model ensemble) | 0.8970 | full two-stage system |

Key conclusions:

- The 52-bit semantic code already retains a strong semantic signal;
- Post-processing and ensembling can significantly improve recall;
- The two-stage architecture reduces bge-large calls from 3,109 to about 187, balancing speed and accuracy.

---

## 14. Overall Summary

The core problem that HSH-64 solves is:

> How to compress high-dimensional semantic vectors into an extremely short binary code while maintaining high recall

Through this design, HSH-64 achieves a Recall@10 of 0.8970 on a 3,109-word Chinese vocabulary, while keeping storage and computational overhead extremely low

At this point the entire HSH-64 system has been explained, and the content of this document was edited based on the changelog
