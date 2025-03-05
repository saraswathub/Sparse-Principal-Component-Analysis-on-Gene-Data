import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, SparsePCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import argparse

def load_dataset(file_path):
    """Load a dataset from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        print(f"Dataset loaded successfully from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def preprocess_data(data, target_column=None):
    """Preprocess data: separate features and standardize."""
    if target_column and target_column in data.columns:
        X = data.drop(columns=[target_column])
    else:
        X = data.copy()
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns)

def apply_pca(data, n_components=2):
    """Apply PCA to the dataset."""
    pca = PCA(n_components=n_components, random_state=42)
    pca_result = pca.fit_transform(data)
    loadings = pd.DataFrame(pca.components_.T, index=data.columns, 
                           columns=[f"PC{i+1}" for i in range(n_components)])
    return pca_result, loadings

def apply_spca(data, n_components=2, alpha=1.0):
    """Apply Sparse PCA to the dataset."""
    spca = SparsePCA(n_components=n_components, alpha=alpha, random_state=42)
    spca_result = spca.fit_transform(data)
    loadings = pd.DataFrame(spca.components_.T, index=data.columns, 
                           columns=[f"PC{i+1}" for i in range(n_components)])
    return spca_result, loadings

def plot_transformed_data(pca_result, spca_result, title="PCA vs SPCA Transformed Data"):
    """Plot the transformed data from PCA and SPCA."""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
    plt.title("PCA Transformed Data")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    
    plt.subplot(1, 2, 2)
    plt.scatter(spca_result[:, 0], spca_result[:, 1], alpha=0.5, color='orange')
    plt.title("SPCA Transformed Data")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig("transformed_data_comparison.png")
    plt.show()

def plot_loadings(pca_loadings, spca_loadings, gene_limit=20):
    """Plot the loadings of PCA vs SPCA for the first principal component."""
    plt.figure(figsize=(12, 5))
    
    # PCA Loadings
    plt.subplot(1, 2, 1)
    pca_pc1 = pca_loadings["PC1"].iloc[:gene_limit]
    pca_pc1.plot(kind="bar", title="PCA Loadings (PC1)")
    plt.xlabel("Features")
    plt.ylabel("Loading Value")
    
    # SPCA Loadings
    plt.subplot(1, 2, 2)
    spca_pc1 = spca_loadings["PC1"].iloc[:gene_limit]
    spca_pc1.plot(kind="bar", title="SPCA Loadings (PC1)", color='orange')
    plt.xlabel("Features")
    plt.ylabel("Loading Value")
    
    plt.tight_layout()
    plt.savefig("loadings_comparison.png")
    plt.show()

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Apply PCA and SPCA to a dataset.")
    parser.add_argument("file_path", type=str, help="Path to the CSV dataset")
    parser.add_argument("--target", type=str, default=None, help="Target column name (optional)")
    parser.add_argument("--n_components", type=int, default=2, help="Number of components")
    parser.add_argument("--alpha", type=float, default=1.0, help="SPCA sparsity parameter")
    args = parser.parse_args()

    # Load and preprocess data
    data = load_dataset(args.file_path)
    if data is None:
        return
    
    X_scaled = preprocess_data(data, target_column=args.target)
    
    # Apply PCA
    print("Applying PCA...")
    pca_result, pca_loadings = apply_pca(X_scaled, n_components=args.n_components)
    
    # Apply SPCA
    print("Applying Sparse PCA...")
    spca_result, spca_loadings = apply_spca(X_scaled, n_components=args.n_components, alpha=args.alpha)
    
    # Compare sparsity
    pca_nonzero = (pca_loadings != 0).sum().sum()
    spca_nonzero = (spca_loadings != 0).sum().sum()
    print(f"\nPCA non-zero loadings: {pca_nonzero}")
    print(f"SPCA non-zero loadings: {spca_nonzero}")
    
    # Save loadings
    pca_loadings.to_csv("pca_loadings.csv")
    spca_loadings.to_csv("spca_loadings.csv")
    print("Loadings saved to 'pca_loadings.csv' and 'spca_loadings.csv'.")
    
    # Visualize results
    print("Generating visualizations...")
    plot_transformed_data(pca_result, spca_result, title=f"PCA vs SPCA (alpha={args.alpha})")
    plot_loadings(pca_loadings, spca_loadings)

if __name__ == "__main__":
    main()
