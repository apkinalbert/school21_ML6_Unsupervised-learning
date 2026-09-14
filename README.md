# Unsupervised Learning & Dimensionality Reduction

This project explores unsupervised learning and dimensionality reduction techniques, with a focus on practical applications.

## Installation

Install all required dependencies:

```bash
pip3 install -r requirements.txt
```

Run the notebook or Python scripts after installing the dependencies.

## What was done

* Built a **sparse user–book interaction matrix** from the Book Recommendation dataset.
* Trained **Linear Regression** and **Random Forest** models to predict user age from sparse high-dimensional features.
* Applied **PCA** and **UMAP** for dimensionality reduction and compared model performance and training time before and after compression.
* Reduced **MNIST** digit representations to 2D using:

  * PCA
  * SVD
  * Randomized SVD
  * t-SNE
  * UMAP
  * LLE
* Compared the resulting embeddings using metrics measuring the separation of digit classes.
* Implemented **SVD-based image compression**, investigating low-rank approximations, singular value spectra, and explained variance.
* Applied **SVD to video frames** to extract a low-rank representation of the background.
* Prepared a dimensionality reduction model for reproducible inference on unseen handwritten digits.

## Main techniques

`PCA` · `SVD` · `Randomized SVD` · `NMF` · `t-SNE` · `UMAP` · `LLE` · `Sparse Matrices` · `Low-Rank Approximation`

## Datasets

* Book Recommendation Dataset
* MNIST
* Background Models Challenge video dataset
