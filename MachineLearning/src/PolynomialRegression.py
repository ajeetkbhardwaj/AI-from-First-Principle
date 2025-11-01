"""
Polynomial Regression :

"""
import numpy as np

class PolynomialRegression():
    def __init__(self):
        pass

    def fit(X, y, batch_size, degrees, epochs, lr):
        X = PolyFeatureSet(X, degrees)
        m, n = X.shape
        # parameter initialiazation
        w = np.zeros((n, 1))
        b = 0
        
        # reshape the y to mx1
        y = y.reshape((m , 1))

        # loss initialization
        loss = []
        for epoch in range(epochs):
            for i in range((m-1)//batch_size + 1):
              start_i = i*batch_size
              end_i = start_i + batch_size
              x_batch = X[start_i:end_i]
              y_batch = y[start_i:end_i]
              y_pred = np.dot(x_batch, w) + b
              dw, db = _Grad(x_batch, y_batch, y_pred)
              w -= lr*dw
              b -= lr*db
            l = _MSELoss(y, np.dot(X, w) + b)
            loss.append(l)

    def predict(X, w, b, degrees):
        X1 = PolyFeatureSet(X, degrees)
        return np.dot(X1, w) + b
    
    def r2_score(y, y_hat):
       sse = np.sum((np.array(y_hat)-np.array(y))**2)
       tss = np.sum((np.array(y)-np.mean(np.array(y)))**2)
       return 1 - (sse / tss )


    def _MSELoss(y_pred, y_true):
        """
        Compute Mean Squared Error (MSE) loss between the true values and the predictions. 
        L(w, b) = sum((y_true - y_pred)^2)
        """
    
        mse_loss = np.mean((y_true - y_pred)**2)
        return mse_loss
    
    def _Grad(X, y_pred, y_true):
        """
        dw = (1/num_rows)*np.dot(X.T, (y_pred - y_true))

        """
        # number of rows = number of training examples
        num_rows = X.shape[0]

        # gradient of loss function wrt weights w and b
        dw = (1/num_rows)*np.dot(X.T, (y_pred, y_true))
        db = (1/num_rows)*np.sum((y_pred - y_true))

        return dw, db
    
    def PolyFeatureSet(X, degrees):
        """
        Transformation of the input dataset by adding higher features.
        It is  a function called polynomial where relationship between dependent and independent variables
        are of nth degree polynomials
        
        Example : 
            Feature Set : [X]
            Task : 2nd degree polynomial then
            New Feature Set : [X, X**2]
        """
        t = X.copy()
        for i in range(degrees):
            X = np.append(X, t**i, axis=1)
        return X
