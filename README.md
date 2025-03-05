This project implements Sparse Principal Component Analysis (SPCA) applied to gene expression data, providing a powerful tool for dimensionality reduction and feature selection in biological datasets. Below, you'll find an explanation of SPCA, how it differs from traditional Principal Component Analysis (PCA), and details about this implementation.
What is Sparse Principal Component Analysis (SPCA)?

Sparse Principal Component Analysis (SPCA) is an extension of Principal Component Analysis (PCA), a widely used technique for reducing the dimensionality of large datasets. While PCA transforms the original variables into a new set of uncorrelated variables (principal components) that explain the maximum variance in the data, SPCA introduces sparsity into these components. This means that instead of all original variables contributing to each principal component (as in PCA), only a subset of variables has non-zero loadings in SPCA.

In the context of gene expression data, where datasets often include thousands of genes (features), SPCA helps identify a smaller, interpretable subset of genes that are most influential in explaining the variance, making it particularly useful for biological interpretation and downstream analysis.
Key Features of SPCA:
