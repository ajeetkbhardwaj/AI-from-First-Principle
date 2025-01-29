""" 
Decison Tree : 
"""

# 0. dependencies
import numpy as np
import random 
from tdqm import tdqm

from sklearn import datasets
from sklearn.model_selection import train_test_split


# 1. Tree Node class
class TreeNode:
    def __init__(self, feature_index, thresold, left_child = None, right_child = None, *, leaf_value=None):
        self.feature_index = feature_index
        self.thresold = thresold
        self.left_child = left_child
        self.right_child = right_child
        self.leaf_value = leaf_value

    def IsLeafNode(self):
        return self.leaf_value is not None
    
# 2. Decision Tree Class

class DecisionTree:
    def __init__(self, min_samples_leaf=2, max_depth=100, n_features=None):
        self.min_sample_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y, l):
        self.n_features = X.shap[1] if not self.n_features else min(self.n_features, )
        self.root = self._grow_tree(X, y, level = 0)

    def _grow_tree(self, X, y, level):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # If we have reached a stopping criteria, return a leaf node
        if (level == self.max_depth or n_labels == 1 or n_samples < self.min_samples_leaf):
            leaf_value = self._most_common_label(y)
            return TreeNode(None, None, leaf_value=leaf_value)

        # Select random features to consider (we don't have to in the case of a Decision Tree, but it's mandatory in the case of a Random Forest)
        feature_idxs = np.random.choice(n_features, self.n_features, replace=False)

        # Greedily select the best split according to information gain
        best_feature_idx, best_threshold = self._best_split_criteria(X, y, feature_idxs)
        # Split the data using the threshold
        left_idxs, right_idxs = self._split(X[:, best_feature_idx], best_threshold)

        # Assign data points to the left or right child
        left_child = self._grow_tree(X[left_idxs, :], y[left_idxs], level+1)
        right_child = self._grow_tree(X[right_idxs, :], y[right_idxs], level+1)
        return TreeNode(best_feature_idx, best_threshold, left_child, right_child)

    def _best_split_criteria(self, X, y, feature_idxs):
        best_gain = -1
        split_criteria = None
        # Iterate through all possible splits and select the pair (feature, threshold) that maximizes information gain
        for feature_idx in feature_idxs:
            feature = X[:, feature_idx]
            thresholds = np.unique(feature)
            for threshold in thresholds:
                gain = self._information_gain(y, feature, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_criteria = (feature_idx, threshold)
        return split_criteria
    
    def _information_gain(self, y, feature, threshold):
        # Calculate the entropy of the parent node
        parent_entropy = self._entropy(y)

        # Calculate the entropy of the left and right child nodes
        left_idxs, right_idxs = self._split(feature, threshold)
        left_entropy = self._entropy(y[left_idxs])
        right_entropy = self._entropy(y[right_idxs])
        
        # Calculate the information gain
        information_gain = parent_entropy - (len(left_idxs)/len(y))*left_entropy - (len(right_idxs)/len(y))*right_entropy
        return information_gain
    
    def _entropy(self, y):
        # Calculate the entropy of a node according to the formula -sum(p_i*log(p_i)) where p_i is the probability of the i-th class
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        entropy = sum(probabilities * -np.log2(probabilities))
        return entropy
    
    def _split(self, X_column, split_thresh):
        # The index of the data points that are less than or equal to the threshold
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        # The index of the data points that are greater than the threshold
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs
    
    def _most_common_label(self, y):
        # Return the most common label in the data
        return np.bincount(y).argmax()
    
    def predict(self, X):
        return np.array([self._predict(x) for x in X])
    
    def _predict(self, x):
        # Traverse the tree to find the leaf node that the data point belongs to
        node = self.root
        while not node.is_leaf_node():
            if x[node.feature_idx] <= node.threshold:
                node = node.left_child
            else:
                node = node.right_child
        return node.leaf_value
    

class RandomForest:

    def __init__(self, n_trees=10, min_samples_leaf=2, max_depth=100, n_features=None):
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.n_features = n_features
        self.n_trees = n_trees

    def fit(self, X, Y):
        self.trees = []
        tree_iterator = tqdm(range(self.n_trees), desc='Training trees')
        for _ in tree_iterator:
            tree = DecisionTree(min_samples_leaf=self.min_samples_leaf, max_depth=self.max_depth, n_features=self.n_features)
            X_sample, Y_sample = self._sample(X, Y)
            tree.fit(X_sample, Y_sample)
            self.trees.append(tree)
    
    def _sample(self, X, Y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], Y[indices]
    
    def _most_common(self, Y):
        return np.bincount(Y).argmax()
    
    def predict(self, X, agg_type='mean'):
        predictions = []
        tree_iterator = tqdm(self.trees, desc='Predicting')
        for tree in tree_iterator:
            predictions.append(tree.predict(X))
        predictions = np.array(predictions)
        if agg_type == 'mean':
            return np.mean(predictions, axis=0)
        elif agg_type == 'majority':
            return np.apply_along_axis(self._most_common, axis=0, arr=predictions)
        else:
            raise ValueError('agg_type must be either mean or majority')

class Stump:

    def __init__(self):
        self.feature_idx = None
        self.threshold = None
        self.alpha = None
        self.p = None

    def predict(self, X):
        n_samples = X.shape[0]
        X_column = X[:, self.feature_idx]
        # Initially all the samples are classified as 1
        predictions = np.ones(n_samples)
        if self.p == 1:
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column > self.threshold] = -1
        return predictions
    
class AdaBoost:

    def __init__(self, num_stumps=5):
        self.num_stumps = num_stumps
        self.stumps = []

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initial weights are 1 / n_samples
        w = np.full(n_samples, (1 / n_samples))

        for _ in range(self.num_stumps):
            stump = Stump()
            min_error = float('inf')

            # Iterate through every feature and threshold to find the best decision stump
            for feature_idx in range(n_features):
                X_column = X[:, feature_idx]
                thresholds = np.unique(X_column)

                for threshold in thresholds:
                    p = 1
                    prediction = np.ones(n_samples)
                    prediction[X_column < threshold] = -1

                    # Error = sum of weights of misclassified samples
                    error = sum(w[y != prediction])

                    # If error is over 50% we flip the prediction so that error will be 1 - error
                    if error > 0.5:
                        error = 1 - error
                        p = -1

                    # Store the best configuration
                    if error < min_error:
                        stump.feature_idx = feature_idx
                        stump.threshold = threshold
                        min_error = error
                        stump.p = p

            # Calculate alpha
            EPS = 1e-10
            stump.alpha = 0.5 * np.log((1.0 - min_error + EPS) / (min_error + EPS))

            # Calculate predictions and update weights
            predictions = stump.predict(X)

            # Only misclassified samples have non-zero weights
            w *= np.exp(-stump.alpha * y * predictions)
            w /= np.sum(w)

            # Save stump
            self.stumps.append(stump)
    
    def predict(self, X):
        preds = [stump.alpha * stump.predict(X) for stump in self.stumps]
        preds = np.sum(preds, axis=0)
        preds = np.sign(preds)
        return preds