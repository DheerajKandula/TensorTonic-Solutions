import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    X = np.array(X)
    y = np.array(y)
    m,n = X.shape
    w = np.zeros(n)
    b = 0.0
    for  _ in range(steps):
        z = np.dot(X, w) + b
        f = _sigmoid(z)
        dJ_dw = (1/m) * (np.dot(X.T, f-y))
        dJ_db = (1/m) * (np.sum(f-y))
        w -= lr*dJ_dw
        b -= lr*dJ_db
    return w, b