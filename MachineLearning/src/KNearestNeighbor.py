"""
Algorithms : K-Nearest Keighbors(KNNs) - Classifier 
1. Input : (X, Y), k=3(default)
2. Output : Y labels
3. Utilities : Euclidean Distance function
4. computing the distance between point and its selected k nbds points
4. sorting the distances and returning it's indices of first k nbds
5. find/extract the labels of the first k nbds
6. return these labels

"""
import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    """
    args:
       Input : (x1, x2)

    output: scalar distance
    """
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KNNClassifier:
    """
    
    
    """
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        pred = [self._predict(x) for x in X]
        return np.array(pred)
    
    def _predict(self, X):
        # Compute distances between x and all examples in the training set
        distances = [euclidean_distance(X, x_train) for x_train in self.X_train]
        # Sort by distance and return indices of the first k neighbors
        k_idx = np.argsort(distances)[: self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_neighbor_labels = [self.y_train[i] for i in k_idx]
        # return the most common class label
        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0]
    def accuracy(y_true, pred):
        accuracy = np.sum(y_true == pred)/len(y_true)
        return accuracy
    

# Quest-1 : How to change KNNClassifier to KNNRegressor ?
# 1. By changing KNN to predict the mean of the neighbors  
# 2. Using  Root Mean Squared Error to evaluate predictions. 
class KNNRegressor:
    def __init__(self, k):
        self.k = k
    