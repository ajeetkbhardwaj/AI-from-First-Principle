import numpy as np

class PrincipalComponentAnalysis():
    def __init__(self, k=3):
        self.k = k

    def reduce(self, X: np.ndarray) -> np.ndarray:
        # 1. Standardize the data
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

        # 2. Compute the covariance matrix
        cov_matrix = np.cov(X, ddof=0, rowvar=False)

        # 3. Compute the eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # 4. Sort the eigenvalues and eigenvectors in descending order (the PCA is the eigenvector with the highest eigenvalue)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # 5. Select the first k eigenvectors
        eigenvectors = eigenvectors[:, :self.k]

        # 6. Compute the new data
        reduced_data = np.matmul(X, eigenvectors)

        # How much of the variance is explained by the first k eigenvectors?
        explained_variance = np.sum(eigenvalues[:self.k]) / np.sum(eigenvalues)

        return reduced_data, explained_variance

        
        
