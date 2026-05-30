import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))

        w = np.zeros(X.shape[1])
        b = 0

        # iterating over each epoch:
        i = 0
        while i < epochs:
            # get y_pred
            y_pred = X @ w + b

            # Calculate loss
            loss = np.mean(np.square(y_pred - y))

            n = X.shape[0]
            # Calculate gradient
            dw = (2 / n) * X.T @ (y_pred - y)
            db = (2 / n) * np.sum(y_pred - y)

            # Update weights using GDs
            w = w - lr * dw
            b = b - lr * db

            # Increment the epoch
            i += 1

        return np.round(w,5), np.round(b,5)
        pass
