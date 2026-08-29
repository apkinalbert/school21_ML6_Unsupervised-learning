# Dimensionality Reduction: Linear Methods and Manifold Learning

## 1. Preamble: Why Dimensionality Reduction?

In supervised learning, each observation is described by a feature vector and has a target variable that we want to predict. In unsupervised learning, there is no target variable: instead, we try to discover structure and useful representations directly from the data.

Dimensionality reduction is one of the central unsupervised-learning tasks. Its goal is to transform data from a high-dimensional feature space into a lower-dimensional space while preserving the properties of the data that are important for the task.

A typical motivation is visualization: a dataset with hundreds of features cannot be directly visualized, whereas a two- or three-dimensional representation can be plotted. Dimensionality reduction can also reduce computational cost, remove noise and redundancy, compress data, and provide useful representations for subsequent machine-learning algorithms.

---

# 2. The Curse of Dimensionality

The **curse of dimensionality** refers to a collection of phenomena that make learning from data increasingly difficult as the dimensionality of the feature space grows.

Suppose observations are distributed in a $d$-dimensional region. If we want to maintain the same resolution in every coordinate, and divide each coordinate into $n$ intervals, then the total number of cells is

$$
n^d.
$$

Thus, increasing the dimension from $d$ to $d+1$ multiplies the number of cells by $n$.

For example,

$$
10^2 = 100,
\qquad
10^5 = 100\,000,
\qquad
10^{10} = 10\,000\,000\,000.
$$

This is why maintaining a comparable density of observations requires exponentially more data as dimensionality increases.

The practical consequences include:

1. **Distances become less informative.** Points tend to become far apart, and the contrast between nearest and farthest neighbors can decrease.
2. **Sparse data.** The available observations occupy only a tiny fraction of the possible feature space.
3. **Higher risk of overfitting.** With many features, a model can discover accidental correlations with the target.
4. **Higher computational cost.** More features usually mean larger matrices, more operations, and more memory.
5. **More difficult estimation.** Estimating an unknown function accurately becomes increasingly data-hungry.

The source specifically emphasizes that adding automatically generated features can be dangerous. If many transformed features are added, some may contain little useful information while others can create accidental correlations that a flexible model interprets as real patterns, increasing overfitting.

---

# 3. Which Previously Studied Models Are Affected by the Curse of Dimensionality?

The curse is **not equally relevant to every model**. Its importance depends on how the model represents the feature space, estimates parameters, and uses distances or local neighborhoods.

## 3.1 k-Nearest Neighbors

The curse is particularly important for **k-nearest neighbors (k-NN)**.

k-NN relies directly on distances such as

$$
d(x,x_i)
=
\sqrt{\sum_{j=1}^{d}(x_j-x_{ij})^2}.
$$

As $d$ grows, irrelevant or noisy coordinates contribute to the distance. Neighborhoods become less local and less meaningful.

Therefore:

- nearest neighbors may no longer be genuinely similar;
- a much larger dataset may be required;
- prediction becomes more computationally expensive.

This is one of the clearest examples of a model suffering from the curse of dimensionality.

## 3.2 Linear Regression and Logistic Regression

The curse can also affect **linear regression** and **logistic regression**, although the mechanism is different.

For linear regression,

$$
y = X\beta + \varepsilon,
$$

the number of parameters is proportional to the number of features. As $d$ grows, we need more data to estimate the coefficients reliably.

With many irrelevant or correlated features, variance can increase and the model can overfit. Ridge and Lasso regularizations help.

Lasso can additionally perform feature selection by driving some coefficients exactly to zero.

## 3.3 Polynomial Features

Polynomial feature expansion is particularly vulnerable.

For $d$ original features, the number of polynomial terms can grow combinatorially with degree. For example, a degree-2 expansion contains terms such as

$$
1,\quad
x_1,\ldots,x_d,\quad
x_1^2,\ldots,x_d^2,\quad
x_ix_j.
$$

The number of degree-$p$ monomials grows rapidly with both $d$ and $p$.

Consequently, blindly generating large numbers of features can make the problem substantially harder.

## 3.4 Decision Trees

Decision trees are generally **less directly affected** than k-NN because they do not require a global distance metric.

Nevertheless, irrelevant features can still cause problems. A tree may select a noisy feature because of a chance correlation with the target. With many candidate features, such accidental correlations become more likely.

Thus the curse can manifest as:

- overfitting;
- larger trees;
- slower training;
- less stable splits.

Random forests and other ensembles reduce some of these effects, but they do not make irrelevant features harmless.

## 3.5 SVMs

The answer depends strongly on the kernel.

A linear SVM can work well in high-dimensional spaces, especially when the data are sparse. However, nonlinear kernels can become computationally expensive because kernel methods often depend on pairwise similarities between observations.

Distance-based and neighborhood-based behavior also makes high-dimensional geometry important for many nonlinear kernels.

## 3.6 Summary

The curse is especially important for models that depend on:

- distances;
- local neighborhoods;
- density estimation;
- large numbers of parameters relative to available data.

It is particularly severe for k-NN and other local methods. Linear and nonlinear parametric models can also suffer through increased estimation variance and overfitting.

---

# 4. Dimensionality Reduction: Main Families

The methods considered here can be organized into two broad groups:

1. **Matrix-factorization / linear methods**
   - Singular Value Decomposition (SVD)
   - Principal Component Analysis (PCA)
   - Non-Negative Matrix Factorization (NMF)

2. **Nonlinear / manifold-learning methods**
   - Kernel methods as a bridge to nonlinear representations
   - t-SNE
   - UMAP
   - Locally Linear Embedding (LLE)

The key distinction is that linear methods search for a linear low-dimensional representation, whereas manifold-learning methods attempt to preserve nonlinear geometric structure.

---

# 5. Singular Value Decomposition (SVD)

## 5.1 Definition

Let

$$
M\in\mathbb{R}^{n\times m}.
$$

Its singular value decomposition is

$$
M=U\Sigma V^T.
$$

A full SVD has

$$
U\in\mathbb{R}^{n\times n},
\qquad
\Sigma\in\mathbb{R}^{n\times m},
\qquad
V\in\mathbb{R}^{m\times m}.
$$

For a compact representation, let

$$
r=\operatorname{rank}(M).
$$

Then

$$
M=U_r\Sigma_rV_r^T,
$$

where

$$
U_r\in\mathbb{R}^{n\times r},
\quad
\Sigma_r\in\mathbb{R}^{r\times r},
\quad
V_r\in\mathbb{R}^{m\times r}.
$$

The matrices $U$ and $V$ have orthonormal columns:

$$
U^TU=I,
\qquad
V^TV=I.
$$

$\Sigma$ is diagonal:

$$
\Sigma=
\operatorname{diag}(\sigma_1,\sigma_2,\ldots,\sigma_r),
$$

with

$$
\sigma_1\geq\sigma_2\geq\cdots\geq\sigma_r\geq0.
$$

The $\sigma_i$ are the **singular values**.

---

## 5.2 Geometric interpretation

SVD can be interpreted as a sequence of transformations:

1. an orthogonal transformation (rotation/reflection);
2. scaling along orthogonal directions;
3. another orthogonal transformation.

The singular values describe how strongly the transformation stretches different directions.

Large singular values correspond to directions carrying a large amount of the matrix's "energy".

---

## 5.3 Low-rank approximation

Keeping only the first $k$ singular values gives

$$
M_k
=
U_k\Sigma_kV_k^T.
$$

This matrix has rank at most $k$.

The fundamental result is the **Eckart–Young theorem**:

$$
M_k
=
\arg\min_{\operatorname{rank}(\tilde M)\leq k}
\|M-\tilde M\|_F.
$$

Here,

$$
\|A\|_F
=
\sqrt{\sum_{i,j}A_{ij}^2}
$$

is the Frobenius norm.

Thus the truncated SVD is the best rank-$k$ approximation in Frobenius norm.

The approximation error is

$$
\|M-M_k\|_F^2
=
\sum_{i>k}\sigma_i^2.
$$

This is one of the main reasons SVD is useful for compression and dimensionality reduction.

---

# 6. PCA

## 6.1 What PCA does

**Principal Component Analysis (PCA)** searches for orthogonal directions that successively explain the maximum possible variance of centered data.

Suppose the data matrix is

$$
X\in\mathbb{R}^{n\times d},
$$

where each row is an observation.

First, center each feature:

$$
X_c=X-\mathbf{}\mu,
$$

where $\mu$ is the vector of feature means.

The covariance matrix is

$$
C
=
\frac{1}{n-1}X_c^TX_c.
$$

PCA finds eigenvectors of $C$:

$$
Cv_i=\lambda_i v_i.
$$

The eigenvalues satisfy

$$
\lambda_1\geq\lambda_2\geq\cdots\geq\lambda_d.
$$

The first principal component is the direction $v_1$ maximizing the projected variance:

$$
v_1
=
\arg\max_{\|v\|=1}
\operatorname{Var}(X_cv).
$$

The second component is the orthogonal direction explaining the largest remaining variance, and so on.

---

## 6.2 PCA through SVD

The covariance matrix does not need to be explicitly constructed.

Compute

$$
X_c=U\Sigma V^T.
$$

Then

$$
X_c^TX_c
=
V\Sigma^TU^T U\Sigma V^T
=
V\Sigma^2V^T.
$$

Therefore:

- the columns of $V$ are the principal directions;
- the squared singular values are proportional to the PCA eigenvalues:

$$
\lambda_i
=
\frac{\sigma_i^2}{n-1}.
$$

The low-dimensional coordinates are

$$
Z=X_cV_k.
$$

Using the SVD,

$$
Z
=
U_k\Sigma_k.
$$

---

# 7. PCA vs SVD

This is one of the central distinctions in the project.

### SVD

SVD is a **general matrix factorization**:

$$
M=U\Sigma V^T.
$$

It can be applied to any real matrix and does not inherently require the matrix to represent centered observations.

### PCA

PCA is a **statistical dimensionality-reduction method**. It is defined in terms of variance and covariance and normally begins by centering the observations.

PCA can be computed using SVD of the centered data matrix.

Therefore:

> **SVD is an algebraic decomposition; PCA is a statistical method that can be implemented using SVD.**

The distinction matters because applying SVD to an arbitrary uncentered data matrix is not automatically equivalent to performing PCA.

---

# 8. Choosing the Number of Principal Components

A common criterion is the **explained variance ratio**.

For component $i$,

$$
\mathrm{EVR}_i
=
\frac{\lambda_i}{\sum_j\lambda_j}.
$$

Using singular values,

$$
\mathrm{EVR}_i
=
\frac{\sigma_i^2}{\sum_j\sigma_j^2}.
$$

The cumulative explained variance is

$$
\mathrm{CEV}(k)
=
\frac{\sum_{i=1}^{k}\sigma_i^2}
{\sum_{i=1}^{r}\sigma_i^2}.
$$

One can select the smallest $k$ such that

$$
\mathrm{CEV}(k)\geq \tau,
$$

for a chosen threshold $\tau$, such as $0.90$, $0.95$, or $0.99$.

---

# 9. PCA: Advantages and Disadvantages

## Advantages

- Simple and mathematically well understood.
- Computationally efficient.
- Produces orthogonal components.
- Useful for visualization.
- Can remove redundant dimensions.
- Provides an optimal linear low-rank reconstruction under squared-error criteria.
- Often useful as preprocessing for other algorithms.
- Can substantially reduce storage and computational cost.

## Disadvantages

- Only captures **linear** structure.
- Sensitive to feature scaling.
- Sensitive to outliers.
- Components can be difficult to interpret.
- Maximum variance is not necessarily the same as maximum predictive information.
- The choice of number of components is task-dependent.

---

# 10. Non-Negative Matrix Factorization (NMF)

## 10.1 Basic idea

NMF factorizes a non-negative matrix into two non-negative matrices.

Given

$$
X\in\mathbb{R}_{\geq0}^{n\times m},
$$

we seek

$$
X\approx WH,
$$

where

$$
W\in\mathbb{R}_{\geq0}^{n\times k},
\qquad
H\in\mathbb{R}_{\geq0}^{k\times m}.
$$

The factorization can be obtained by minimizing an objective such as

$$
\min_{W,H\geq0}
\|X-WH\|_F^2.
$$

---

## 10.2 Interpretation

NMF is especially useful when the entries of $X$ represent quantities that naturally cannot be negative.

Examples include:

- pixel intensities;
- word counts;
- audio magnitude spectra;
- non-negative measurements.

Because $W$ and $H$ are non-negative, the representation tends to be **additive**.

A data vector can be represented approximately as

$$
x_i
\approx
\sum_{j=1}^{k}W_{ij}h_j,
$$

where $h_j$ are non-negative basis components.

This can produce interpretable parts-based representations.

For example, in image analysis, one component may correspond to one recurring visual pattern, and an image can be represented as an additive combination of such patterns.

---

# 11. NMF vs SVD

Both methods can produce low-rank approximations:

$$
X\approx X_k.
$$

However, their mathematical constraints and interpretations are different.

## SVD

SVD gives

$$
X\approx U_k\Sigma_kV_k^T.
$$

The factors can contain **positive and negative values**.

The decomposition is orthogonal and has a strong optimality property: truncated SVD gives the best rank-$k$ approximation under the Frobenius norm.

## NMF

NMF gives

$$
X\approx WH,
$$

with

$$
W\geq0,\qquad H\geq0.
$$

The representation is additive rather than cancellation-based.

NMF is therefore often more interpretable for inherently non-negative data.

## Main difference

> **SVD is an unconstrained orthogonal matrix factorization with optimal low-rank approximation properties, whereas NMF imposes non-negativity and often produces additive, parts-based representations.**

NMF generally does **not** have the same unique optimal closed-form solution as truncated SVD.

---

# 12. When Should NMF Be Used?

NMF is particularly attractive when:

1. the input matrix is non-negative;
2. additive components have a meaningful interpretation;
3. interpretability is more important than the optimal squared-error rank-$k$ approximation;
4. sparse or parts-based representations are desirable.

Typical applications include:

- topic modeling;
- document-term matrices;
- image decomposition;
- audio source separation;
- recommender systems;
- feature extraction.

---

# 13. Linear vs Nonlinear Dimensionality Reduction

PCA, SVD-based reduction, and NMF fundamentally work with linear combinations of the original features.

Suppose the original observation is

$$
x\in\mathbb{R}^d.
$$

A linear embedding has the form

$$
z=Wx
$$

or, after centering,

$$
z=W(x-\mu).
$$

But real datasets can have nonlinear structure.

For example, observations may lie close to a curved surface embedded in a high-dimensional space.

A linear method may fail because no single linear subspace can represent the geometry accurately.

This motivates **manifold learning**.

---

# 14. Manifold Learning

A manifold is, informally, a space that may be globally curved but locally resembles ordinary Euclidean space.

Suppose high-dimensional observations satisfy

$$
x=f(z),
$$

where

$$
z\in\mathbb{R}^k,
\qquad
k\ll d.
$$

The observed data may therefore be generated from a low-dimensional latent coordinate system through a nonlinear mapping $f$.

The goal of manifold learning is to recover a useful low-dimensional representation $z$ while preserving some aspect of the underlying geometry.

Important manifold-learning methods include:

- t-SNE;
- UMAP;
- LLE.

---

# 15. t-SNE

**t-distributed Stochastic Neighbor Embedding (t-SNE)** is primarily a visualization method for mapping high-dimensional observations into two or three dimensions.

Its main goal is to preserve **local neighborhood structure**.

---

## 15.1 High-dimensional similarities

For each observation $x_i$, t-SNE defines a conditional probability describing how likely $x_j$ is to be considered a neighbor of $x_i$.

A typical formulation uses a Gaussian kernel:

$$
p_{j|i}
=
\frac{
\exp\left(-\frac{\|x_i-x_j\|^2}{2\sigma_i^2}\right)
}{
\sum_{k\neq i}
\exp\left(-\frac{\|x_i-x_k\|^2}{2\sigma_i^2}\right)
}.
$$

The bandwidth $\sigma_i$ is related to the desired neighborhood size through the **perplexity** parameter.

A symmetric probability is commonly constructed as

$$
p_{ij}
=
\frac{p_{j|i}+p_{i|j}}{2n}.
$$

---

## 15.2 Low-dimensional similarities

Each observation receives a low-dimensional coordinate $y_i$.

Instead of a Gaussian distribution, t-SNE uses a Student's $t$ distribution with one degree of freedom:

$$
q_{ij}
=
\frac{
(1+\|y_i-y_j\|^2)^{-1}
}{
\sum_{k\neq l}
(1+\|y_k-y_l\|^2)^{-1}
}.
$$

The heavy tails are important because they allow moderately distant points in the original space to be represented farther apart in the low-dimensional space.

---

## 15.3 Optimization

t-SNE minimizes the Kullback–Leibler divergence between the high-dimensional and low-dimensional similarity distributions:

$$
C
=
D_{KL}(P\|Q)
=
\sum_{i\neq j}
p_{ij}
\log
\frac{p_{ij}}{q_{ij}}.
$$

The low-dimensional coordinates are optimized using gradient-based optimization.

A schematic gradient is

$$
\frac{\partial C}{\partial y_i}
=
4
\sum_j
(p_{ij}-q_{ij})
(1+\|y_i-y_j\|^2)^{-1}
(y_i-y_j).
$$

The exact optimization procedure includes additional practical techniques such as early exaggeration and specialized numerical implementations.

---

# 16. t-SNE: Advantages and Disadvantages

## Advantages

- Excellent visualization of local neighborhoods.
- Can reveal nonlinear cluster structure.
- Works well for complex high-dimensional embeddings.
- Produces intuitive 2D or 3D plots.

## Disadvantages

- Computationally expensive for large datasets, although modern implementations are substantially optimized.
- Results can depend on hyperparameters such as perplexity and initialization.
- Global distances and cluster sizes are not necessarily meaningful.
- Different runs can produce different layouts.
- The apparent separation between clusters can be visually stronger than the evidence for actual global separation.
- Not naturally suited to representing new unseen observations without additional methodology.
- It is generally not the best choice when the goal is reconstruction or preserving global geometry.

---

# 17. UMAP

Uniform Manifold Approximation and Projection (UMAP) is another nonlinear dimensionality-reduction method, mainly used for visualization.

Like t-SNE, UMAP tries to preserve local neighborhoods when mapping high-dimensional observations into a lower-dimensional space.

The main difference is how local relationships are represented.

## 17.1 Basic idea

For each observation, UMAP finds its $k$ nearest neighbors and constructs a graph:

$$
G=(V,E).
$$

Each observation is a vertex, and an edge connects observations that are close to each other.

Thus, instead of assigning a similarity probability to every pair of observations as in t-SNE, UMAP works primarily with nearest-neighbor relationships.

Conceptually:

$$
\text{t-SNE:}
\qquad
x_i,x_j
\rightarrow
p_{ij}
$$

while

$$
\text{UMAP:}
\qquad
x_i
\rightarrow
\text{$k$ nearest neighbors}.
$$

UMAP then finds low-dimensional coordinates $y_i$ that preserve these neighborhood relationships as well as possible.

## 17.2 UMAP vs t-SNE

The simplest way to think about the difference is:

| | t-SNE | UMAP |
|---|---|---|
| How similarity is represented | Probability $p_{ij}$ between pairs | Nearest-neighbor graph |
| Main goal | Preserve local neighborhoods | Preserve local neighborhoods and graph structure |
| Typical output | 2D/3D | 2D/3D or higher dimensions |
| Speed | Usually slower | Usually faster |
| New observations | Not naturally supported | Can transform new observations |
| Main hyperparameter | Perplexity | Number of neighbors (`n_neighbors`) |

---

# 18. Locally Linear Embedding (LLE)

**Locally Linear Embedding (LLE)** is a nonlinear dimensionality-reduction algorithm based on the assumption that a nonlinear manifold is approximately linear in a sufficiently small neighborhood.

Its key idea is:

> If each point can be reconstructed from its nearest neighbors using local linear combinations, then the same reconstruction relationships should remain valid after mapping the data to a lower-dimensional space.

---

# 19. Structure of the LLE Algorithm

LLE consists of **three main conceptual steps**.

## Step 1: Find nearest neighbors

For every point $x_i$, find its $k$ nearest neighbors:

$$
\mathcal N(i)=\{x_{i_1},\ldots,x_{i_k}\}.
$$

The neighborhood can be determined using Euclidean distance or another appropriate metric.

---

## Step 2: Compute reconstruction weights

For each point, find weights $w_{ij}$ that reconstruct $x_i$ from its neighbors:

$$
x_i
\approx
\sum_{j\in\mathcal N(i)}
w_{ij}x_j.
$$

The weights are obtained by minimizing

$$
\epsilon(W)
=
\sum_i
\left\|
x_i-
\sum_{j\in\mathcal N(i)}
w_{ij}x_j
\right\|^2
$$

subject to

$$
\sum_{j\in\mathcal N(i)}w_{ij}=1.
$$

The weights for non-neighboring points are set to zero.

The constraint

$$
\sum_j w_{ij}=1
$$

makes the reconstruction invariant to translations of the local coordinate system.

---

## Step 3: Find low-dimensional coordinates

Now we seek low-dimensional points $y_i\in\mathbb R^k$ that preserve the same reconstruction weights:

$$
y_i
\approx
\sum_j w_{ij}y_j.
$$

Therefore we minimize

$$
\Phi(Y)
=
\sum_i
\left\|
y_i-
\sum_jw_{ij}y_j
\right\|^2.
$$

In matrix form,

$$
\Phi(Y)
=
\operatorname{Tr}
\left(
Y^T(I-W)^T(I-W)Y
\right).
$$

The solution is obtained from the eigenvectors corresponding to the smallest nonzero eigenvalues of

$$
M=(I-W)^T(I-W).
$$

The eigenvector corresponding to eigenvalue zero represents the trivial constant solution and is discarded.

The next $k$ eigenvectors provide the low-dimensional embedding.

---

# 20. Why LLE Works

Imagine a curved manifold embedded in a high-dimensional space.

Globally, the manifold may be highly nonlinear. But locally, a small neighborhood of the manifold can often be approximated by a linear patch.

LLE does not attempt to discover one global linear projection.

Instead, it:

1. learns the local geometry around every point;
2. represents each point through local reconstruction weights;
3. finds a low-dimensional configuration preserving those local relationships.

Thus LLE preserves **local linear structure** rather than global Euclidean distances.

---

# 21. LLE: Advantages and Disadvantages

## Advantages

- Captures nonlinear manifold structure.
- Elegant mathematical formulation.
- Focuses on local geometry.
- Does not require explicitly estimating a global nonlinear mapping.
- Often works well when the manifold is smooth and locally approximately linear.

## Disadvantages

- Sensitive to the choice of number of neighbors $k$.
- Sensitive to noise and outliers.
- The neighborhood graph is critical.
- Can fail when neighborhoods do not represent the manifold correctly.
- The optimization can become numerically unstable for poorly conditioned local neighborhoods.
- The basic algorithm is not naturally designed for straightforward out-of-sample prediction.
- Global structure may not be preserved.

---

# 22. Comparison of the Main Methods

| Method | Linear? | Main idea | Preserves | Typical use |
|---|---:|---|---|---|
| SVD | Yes | Matrix factorization | Low-rank structure | Compression, factorization |
| PCA | Yes | Maximize variance | Global linear variance structure | Reduction, visualization, preprocessing |
| NMF | Yes | Non-negative factorization | Additive structure | Parts-based representations |
| t-SNE | Nonlinear | Match neighborhood probabilities | Local neighborhoods | Visualization |
| UMAP | Nonlinear | Preserve a nearest-neighbor graph | Local manifold structure | Visualization, embeddings |
| LLE | Nonlinear | Preserve local reconstruction weights | Local linear geometry | Manifold learning |

---
