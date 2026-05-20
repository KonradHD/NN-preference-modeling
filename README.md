# NN-preference-modeling

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg?style=for-the-badge&logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-Engineering_Thesis-success.svg?style=for-the-badge)


## The Core Problem: Multi-Criteria Decision Making (MCDM)

Real-world decision-making is rarely driven by a single factor. Selecting the optimal solution—whether it involves software architecture, business strategy, or resource allocation—requires balancing multiple, often conflicting criteria (e.g., minimizing cost while maximizing both performance and security). This domain, known as **Multi-Criteria Decision Making (MCDM)**, forms the backbone of modern operational research and systems engineering.

**Analytic Hierarchy Process (AHP)** - one of the most prominent MCDM method, which simplifies complex problems into a series of pairwise comparisons evaluated by human experts.

**The Engineering Challenge:**
The fundamental vulnerability in AHP is **human inconsistency**. Experts are rarely perfectly logical in their subjective judgments. Classical analytical solvers, such as the Eigenvalue Method (EVM), are highly sensitive to this cognitive noise. When the human Consistency Ratio - CR rises, the mathematical foundations of classical AHP break down, leading to distorted and illogical priority rankings. 

### Figurative Example: Smartphone Selection Process


To understand the input/output pipeline, consider a decision-making scenario where a user is choosing a phone based on three criteria: Feature A - Battery, Feature B - Memory, and Feature C - Camera.

**1. Input Data: Pairwise Comparison Matrix** The user evaluates the features against each other using **Saaty's Scale** (a standard AHP scale from 1 to 9, where 1 indicates equal importance).

| | Feature A | Feature B | Feature C |
| :--- | :--- | :--- | :--- |
| **Feature A** | 1 | 2 | 1/3 |
| **Feature B** | 1/2 | 1 | 1/4 |
| **Feature C** | 3 | 4 | 1 |

*Interpretation:* The value **2** indicates that Feature A is twice as important as Feature B. Conversely, the value **1/3** means Feature A is considered three times less important than Feature C.

**2. Target Output: Predicted Weight Vector** Instead of relying on sensitive analytical solvers, the `DeepAHPNet` model ingests the matrix above and predicts the final, normalized priority weight vector. This vector explicitly defines the overall importance of each element in the final choice:

| Feature A | Feature B | Feature C |
| :---: | :---: | :---: |
| 0.3 | 0.1 | 0.6 |

*Conclusion:* In this scenario, the network correctly identifies Feature C (0.6) as the dominant deciding factor, followed by Feature A (0.3) and Feature B (0.1).

This project addresses this critical flaw by replacing brittle analytical solvers with a robust deep learning architecture trained to extract true hierarchical priorities even from highly contradictory matrices.



### Neural Network Solution

The objective of this thesis is to design and train a neural network to estimate the priority weight vector based on pairwise comparison matrices. This vector is generated using a softmax function and optimized through a custom loss function. Rather than focusing directly on individual pairwise comparisons, this loss function optimizes the entire output representation of the weight vector by incorporating additional constraints that extend beyond the standard consistency criterion. **The expected outcome is the development of a ranking calculation method characterized by greater stability** (i.e., robustness to perturbations and noise) in its results. The core evaluation and conclusion of the thesis will involve benchmarking the weight values predicted by the model against the results of classical, analytical methods. This includes testing specific properties, such as comparing the stability of solutions derived classically versus those obtained via the neural network.


## Data pipeline Architecture

The data pipeline in this project is built for high efficiency, mathematical validity and using efficiency. It bypasses traditional, slow CPU-bound loops by utilizing heavily vectorized NumPy and PyTorch implementations.


### 1. Matrix Generation & Noise Injection
* **Ground Truth:** Ideal, perfectly consistent matrices are synthesized directly from synthetic target priority vectors using Saaty's relational definition ($a_{ij} = w_i / w_j$).
* **Noise Simulation:** Controlled human inconsistency is introduced by disturbing the matrix elements according to predefined target **Consistency Rates (CR)**. This creates the exact noisy matrices that the network is tasked to solve.

### 2. Dynamic Permutation Augmentation 
To prevent the model from memorizing fixed position biases and spatial patterns within the matrices, the `AHPDatasets` applies an on-the-fly **Symmetric Permutation Rule** for training data.
* A random index permutation vector is generated at each fetch step (e.g., $P = [2, 0, 4, 1, 3]$ for $N=5$).
* The matrix rows and columns are shuffled simultaneously and symmetrically: 
  $$\mathbf{M}_{aug} = \mathbf{M}[P, :][:, P]$$
* **Critical Step:** The Target Ground Truth weight vector is strictly permuted using the same vector ($\mathbf{w}_{aug} = \mathbf{w}[P]$) to maintain perfect mathematical alignment.

### 3. Logarithmic Scaling
Instead of feeding raw Saaty scale values (which range non-linearly from $1/9$ to $9$), the dataset applies an element-wise natural logarithm:
$$\mathbf{M}_{log} = \ln(\mathbf{M} + 10^{-8})$$
* **Engineering Benefit:** This linearizes the reciprocal relationships. Because $\ln(1/x) = -\ln(x)$, an entry of $9$ becomes $\approx 2.19$, and an entry of $1/9$ becomes $\approx -2.19$. This centers the data distributions perfectly around $0$, significantly stabilizing the gradient flow and accelerating training convergence.

### 4. Upper Triangular Extraction (Dimensionality Reduction)
Since an AHP matrix is completely deterministic based on its upper triangle ($a_{ii}=1$ and $a_{ji} = 1/a_{ij}$), feeding the full $N \times N$ matrix to a dense network wastes computational capacity. 
* Inside the model's `forward` pass, a highly optimized `triu_indices` buffer slices out only the strictly independent comparison elements.
* For a $5 \times 5$ matrix, the input vector drops from **25 inputs** down to just **10 inputs**:
  $$\text{Input Dim} = \frac{N \times (N - 1)}{2}$$
* This final compressed vector is what gets injected into the `RobustDenseEncoder` feature extraction layers for the siamese architecture. 


## Model architectures

### Siamese Architecture 

* **Strict Weight Sharing:** The model instantiates a single backbone encoder (`AdvancedAHPEncoder`). Both input branches ($M_1$ and $M_2$) pass through this exact same sub-network. This guarantees that the geometric transformations and learned preference representations are perfectly invariant to whichever branch receives the matrix, preventing asymmetric feature drifting.
* **Deep Residual Blocks (ResNet-Style):** The core of the encoder utilizes stacked `ResidualBlock` modules. By expanding the internal representation dimensionalities ($H_{dim} \times 2$) and employing **Skip Connections** ($F(x) + x$), the network effectively mitigates gradient vanishing and allows deeper layer configurations to converge stably.
* **Modern Regularization Funnel:** To safeguard the weights against collapsing into uniform distributions, each block integrates modern scaling components:
  * **Layer Normalization (`LayerNorm`):** Stabilizes internal covariate shifts across dense layers during contrastive updates.
  * **GELU Activations:** Provides a smoother, non-monotonic gradient surface compared to standard ReLUs, boosting ranking alignment.
  * **Dropout Interleaving:** Introduces structural stochastics to suppress over-reliance on individual noise artifacts.


### Graph Architecture 


To fully exploit the relational nature of the Analytic Hierarchy Process, this project introduces a native **Graph Architecture** (`GraphAHPNet`). Instead of flattening the input matrix or treating it as a 2D image, this topology treats the AHP matrix as a fully connected, weighted directed graph. In this paradigm, each decision criterion acts as a **node**, and the pairwise comparisons act as **edge weights**.


* **Relational Message Passing:** The custom `GraphAHPConv` layer utilizes Batch Matrix Multiplication (`torch.bmm`) to perform neighborhood aggregation. This allows each criterion to dynamically update its internal representation based on the incoming "votes" (comparisons) from all other criteria in the matrix.
* **Learnable Node Embeddings:** The network does not rely on static input features. It initializes a set of continuous, learnable parameters (`initial_node_features`) for the criteria nodes. These baseline embeddings are iteratively refined through the graph convolution layers based on the graph's topology.
* **Stable Graph Propagations:** Deep GNNs are notoriously prone to oversmoothing and vanishing gradients. This architecture combats these issues by wrapping every graph convolution with **Layer Normalization (`LayerNorm`)**, **Dropout**, and **GELU** activations, ensuring stable training dynamics even with high-variance logarithmic edge weights.
* **Node-Level Readout (The Head):** After multi-hop relational aggregations across $N$ layers, the shared Multi-Layer Perceptron (MLP) head projects each node's high-dimensional hidden state down to a single scalar logit. This flawlessly maps the graph structure directly to the required priority weight vector generation via the final Softmax layer.


## Loss function 

To ensure the network learns both the exact numerical values and the logical hierarchy of the criteria, the model optimizes a custom composite loss function. The total loss is defined as a weighted sum of three distinct penalty components:

$$L_{total} = \alpha \cdot L_{reconstruction} + \beta \cdot L_{stability} + \gamma \cdot L_{COP}$$

* **$L_{reconstruction}$:** supervised component (typically Mean Squared Error) that directly penalizes the numerical deviation between the network's predicted priority weight vector and the ground truth.
* **$L_{stability}$:** a stabilizing regularizer designed to ensure that minor perturbations or noise in the input data do not cause drastic, disproportionate shifts in the final priority weights.
* **$L_{COP}$:** pairwise ranking hinge loss that strictly enforces the topological hierarchy, ensuring that the qualitative order of priorities designated by the expert is mathematically preserved even under high inconsistency.

### **$L_{stability}$**


The $L_{stability}$ term acts as a crucial regularizer by leveraging **Kullback-Leibler (KL) Divergence**. When the network encounters highly contradictory information (noise within the AHP matrix), it might attempt to artificially minimize error by becoming overly confident—collapsing its predictions to heavily favor a single criterion. 

To prevent this extreme polarization, the loss calculates the divergence between the predicted priority weight vector ($w$) and a uniform distribution ($u$):

$$L_{stability} = D_{KL}(w||u) = \sum_{i=1}^{n} w_i \ln(w_i \cdot n)$$

By minimizing this divergence, the network is strictly penalized for generating extreme, "spiky" probability distributions. Instead, it is mathematically forced to strive for a more balanced, higher-entropy weight vector unless the underlying matrix data provides strong, logically consistent evidence to justify a definitive ranking.


### **$L_{COP}$** 

This component strictly forces the neural network to preserve the true priority ranking of individual criteria, acting as a rigorous topological constraint. 

For every pair where the expert explicitly stated that criterion $i$ is more important than criterion $j$ (i.e., the matrix value $a_{ij} > 1$), the network's raw output logits ($s_i$ and $s_j$) are evaluated:

$$L_{COP} = \sum_{i,j:a_{ij}>1} \max(0, -(s_i - s_j) + m)$$

Where $m$ represents a configurable safety margin. If the network fails to rank $s_i$ higher than $s_j$ by at least this margin, it accumulates a linear penalty. This ensures the model learns the correct relative hierarchy, not just the absolute values.

*💡 **Academic Foundation:** This pairwise ranking loss mechanism is directly adapted from the learning-to-rank methodology introduced in the seminal paper **"Optimizing Search Engines using Clickthrough Data"** by Thorsten Joachims.*

### **$L_{reconstruction}$**

This is the primary data-fitting component of the objective function. It is implemented using the standard **Mean Squared Error (MSE)**. Its purpose is to directly penalize the absolute numerical deviation between the network's predicted priority weight vector and the ideal ground truth values, ensuring the model learns the precise quantitative scale of the priorities.

## Results analysis 

### Criteria 5




## Conclusions  







## Author 
- Konrad Ćwiękas