"""
Utilities function for implementation of machine learning algorithms from first principles.





"""

import numpy as np

def LinearCombination(X, weights, bias):
    """
    Computes the linear combination of inputs and weights with bias.

    Parameters:
    X (numpy.ndarray): Input features.
    weights (numpy.ndarray): Weights for each feature.
    bias (float): Bias term.

    Returns:
    numpy.ndarray: Result of the linear combination.
    """
    return X @ weights + bias

def Sigmoid(Z):
    """
    Computes the sigmoid activation function.

    Parameters:
    Z (numpy.ndarray): Input values.

    Returns:
    numpy.ndarray: Sigmoid of the input values.
    """
    exp = np.exp(Z - np.max(Z))  # for numerical stability
    for i in range(len(Z)):
        exp[i] /= np.sum(exp[i])
    return exp

def LabelOneHotEncoding(y, num_classes):
    """
    Converts labels to one-hot encoded format.

    Parameters:
    y (numpy.ndarray): Array of labels.
    num_classes (int): Number of classes.

    Returns:
    numpy.ndarray: One-hot encoded labels.
    """
    one_hot = np.zeros((len(y), num_classes))
    for i, label in enumerate(y):
        one_hot[i, label] = 1
    return one_hot


# ----- Multivariate Normal Distribution PDF -----

def pdf_multivariate_normal_distribution(x, mean, cov):
    """Compute multivariate normal density function."""
    d = len(x)
    cov_det = np.linalg.det(cov)
    cov_inv = np.linalg.inv(cov)
    norm_const = 1.0 / np.sqrt((2 * np.pi) ** d * cov_det)
    x_mu = (x - mean).reshape(1, -1)
    result = np.exp(-0.5 * x_mu @ cov_inv @ x_mu.T)
    return float(norm_const * result)