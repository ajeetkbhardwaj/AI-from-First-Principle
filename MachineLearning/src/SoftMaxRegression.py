import numpy as np
import utils

class SoftMaxRegression():
    """
    
    """
    def __init__(self, learning_rate=0.01, num_iterations=100):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Fit the SoftMax Regression model to the training data.
        Parameters:
            X (np.ndarray): Training data features.
            y (np.ndarray): Training data labels.

        Returns:
            None

        """
        # find the number of samples and features
        self.n_samples, self.n_features = X.shape
        # find the number of classes
        self.n_classes = len(np.unique(y))
        
        # initialize weights and bias
        self.weights = np.random.random((self.n_features, self.n_classes))
        self.bias = np.random.random(self.n_classes)
        
        Loss = []
        
        # training loop with gradient descent
        for _ in range(self.num_iterations):
            # Compute the linear combination of inputs and weights
            linear_model = utils.LinearCombination(X, self.weights, self.bias)
            
            # Apply the sigmoid activation function
            y_predicted = utils.Sigmoid(linear_model)
            
            # one-hot encode the labels
            y_one_hot = utils.LabelOneHotEncoding(y, self.n_classes)
            
            # Compute gradients
            dw = (1 / self.n_samples) * np.dot(X.T, (y_predicted - y_one_hot))
            db = (1 / self.n_samples) * np.sum(y_predicted - y_one_hot, axis=0)
            
            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # computing loss
            loss = - np.mean(np.log(y_predicted[np.arange(len(y)), y]))
            Loss.append(loss)

            if (self.num_iterations % 10 == 0):
                print(f"Iteration {self.num_iterations}, Loss: {loss}".format(self.num_iterations, loss))

        return self.weights, self.bias, Loss

    def predict(self, X):
        """
        Predict using the SoftMax Regression model.
        
        Parameters:
            X (np.ndarray): Input features for prediction.
        
        Returns:
            np.ndarray: Predicted class labels.
        """
        linear_model = utils.LinearCombination(X, self.weights, self.bias)
        y_predicted = utils.Sigmoid(linear_model)
        return np.argmax(y_predicted, axis=1)

    def score(self, X, y):
        """
        Calculate the accuracy of the SoftMax Regression model.
        
        Parameters:
            X (np.ndarray): Input features for scoring.
            y (np.ndarray): True labels for scoring.
        
        Returns:
            float: Accuracy of the model.
        """
        y_predicted = self.predict(X)
        return np.mean(y_predicted == y)