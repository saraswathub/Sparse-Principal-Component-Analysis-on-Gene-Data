# Sparse Principal Component Analysis (SPCA) on Gene Data

Welcome to this repository! This project implements Sparse Principal Component Analysis (SPCA) applied to gene expression data, providing a powerful tool for dimensionality reduction and feature selection in biological datasets. Below, you'll find an explanation of SPCA, how it differs from traditional Principal Component Analysis (PCA), and details about this implementation.

## What is Sparse Principal Component Analysis (SPCA)?

Sparse Principal Component Analysis (SPCA) is an extension of Principal Component Analysis (PCA), a widely used technique for reducing the dimensionality of large datasets. While PCA transforms the original variables into a new set of uncorrelated variables (principal components) that explain the maximum variance in the data, SPCA introduces sparsity into these components. This means that instead of all original variables contributing to each principal component (as in PCA), only a subset of variables has non-zero loadings in SPCA. 

In the context of gene expression data, where datasets often include thousands of genes (features), SPCA helps identify a smaller, interpretable subset of genes that are most influential in explaining the variance, making it particularly useful for biological interpretation and downstream analysis.

### Key Features of SPCA:
- **Sparsity**: Many loadings (coefficients) in the principal components are forced to zero, resulting in components that depend on fewer variables.
- **Interpretability**: By focusing on a subset of genes, SPCA makes it easier to identify biologically relevant features.
- **Dimensionality Reduction**: Like PCA, SPCA reduces the number of dimensions while preserving important patterns in the data.

## How is SPCA Different from PCA?

While PCA and SPCA share the common goal of dimensionality reduction, they differ significantly in their approach and outcomes. Here’s a breakdown of the key differences:

| Feature                | PCA                              | SPCA                              |
|------------------------|----------------------------------|-----------------------------------|
| **Loadings**           | All variables contribute to each principal component (non-zero loadings). | Only a subset of variables has non-zero loadings, introducing sparsity. |
| **Interpretability**   | Components are linear combinations of all original features, often hard to interpret biologically. | Sparse components highlight a small number of key features (e.g., genes), improving interpretability. |
| **Objective**          | Maximize explained variance without constraints on loadings. | Maximize explained variance while enforcing sparsity constraints (e.g., L1 penalty). |
| **Computation**        | Solved via eigenvalue decomposition or singular value decomposition (SVD). | Requires optimization techniques (e.g., penalized regression) to enforce sparsity. |
| **Use Case**           | General dimensionality reduction. | Feature selection and interpretability in high-dimensional data like gene expression. |

### Why Use SPCA for Gene Data?
In traditional PCA, each principal component is a linear combination of all genes, which can make it challenging to pinpoint which genes are driving the variance. For example, if you have 20,000 genes, a PCA component might include small contributions from all of them, obscuring biological insights. SPCA, by contrast, might identify that only 50 genes are critical for a given component, allowing researchers to focus on those for further study—say, in understanding disease mechanisms or identifying biomarkers.

## Repository Contents

- **`spca_gene_analysis.py`**: Python script implementing SPCA on gene expression data using a sample dataset.
- **`data/`**: Directory containing example gene expression data (e.g., a CSV file with genes as columns and samples as rows).
- **`requirements.txt`**: List of Python dependencies (e.g., NumPy, scikit-learn, pandas).
- **`README.md`**: This file!

## References

- Zou, H., Hastie, T., & Tibshirani, R. (2006). Sparse Principal Component Analysis. *Journal of Computational and Graphical Statistics*.
- Scikit-learn documentation: [SparsePCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.SparsePCA.html).
