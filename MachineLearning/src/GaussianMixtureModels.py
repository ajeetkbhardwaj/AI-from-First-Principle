import numpy as np
from src.utils import pdf_multivariate_normal_distribution


# ----- Gaussian Mixture Model -----
class GaussianMixtureModel:

    def __init__(self, k: int, X: np.ndarray, max_iter: int = 100, tol: float = 1e-6) -> None:
        self.k = k
        self.X = X
        self.n, self.d = X.shape
        self.max_iter = max_iter
        self.tol = tol

        # Initialize parameters
        np.random.seed(42)
        self.mu = X[np.random.choice(self.n, k, replace=False)]  # better init
        self.cov_matrix = np.array([np.cov(X, rowvar=False)] * k)
        self.pi = np.ones(k) / k  # equal weights initially
        self.r = np.zeros((self.n, k))

    # ----- E-Step -----
    def _e_step(self) -> None:
        for j in range(self.k):
            for i in range(self.n):
                self.r[i, j] = self.pi[j] * pdf_multivariate_normal_distribution(
                    self.X[i], self.mu[j], self.cov_matrix[j]
                )
        # Normalize responsibilities
        self.r = self.r / np.sum(self.r, axis=1, keepdims=True)

    # ----- M-Step -----
    def _m_step(self) -> None:
        N_k = np.sum(self.r, axis=0)

        for j in range(self.k):
            # Update means
            self.mu[j] = np.sum(self.r[:, j].reshape(-1, 1) * self.X, axis=0) / N_k[j]

            # Update covariance
            diff = self.X - self.mu[j]
            self.cov_matrix[j] = (self.r[:, j].reshape(-1, 1) * diff).T @ diff / N_k[j]

            # Regularize covariance (to avoid singular matrix)
            self.cov_matrix[j] += 1e-6 * np.eye(self.d)

            # Update mixing coefficients
            self.pi[j] = N_k[j] / self.n

    # ----- Log-likelihood -----
    def _compute_log_likelihood(self) -> float:
        ll = 0.0
        for i in range(self.n):
            tmp = 0
            for j in range(self.k):
                tmp += self.pi[j] * pdf_multivariate_normal_distribution(self.X[i], self.mu[j], self.cov_matrix[j])
            ll += np.log(tmp)
        return ll

    # ----- Fit -----
    def fit(self, verbose: bool = True):
        prev_ll = None
        for iteration in range(self.max_iter):
            self._e_step()
            self._m_step()
            ll = self._compute_log_likelihood()

            if verbose:
                print(f"Iteration {iteration + 1}, Log-Likelihood: {ll:.4f}")

            # Convergence check
            if prev_ll is not None and abs(ll - prev_ll) < self.tol:
                print("Converged at iteration", iteration + 1)
                break
            prev_ll = ll

    # ----- Predict cluster for new data -----
    def predict(self, X_new):
        probs = np.zeros((X_new.shape[0], self.k))
        for j in range(self.k):
            for i in range(X_new.shape[0]):
                probs[i, j] = self.pi[j] * pdf_multivariate_normal_distribution(
                    X_new[i], self.mu[j], self.cov_matrix[j]
                )
        return np.argmax(probs, axis=1)
