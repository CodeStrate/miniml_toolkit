import numpy as np
from loguru import logger

class SimpleLinearRegression:
    def __init__(self, learning_rate: int = 0.01, gradient: bool = False, lambda_: float=0.001): # weight are beta1/theta1/m and biases are beta0/theta0/c
        self.learning_rate = learning_rate
        self.gradient = gradient
        self.lambda_ = lambda_
        self.weight = 0.0
        self.bias = 0.0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, iterations: int = 1000):
        if not self.gradient:
        # OLS fitting
            x_bar = np.mean(X_train)
            y_bar = np.mean(y_train)
            numerator = np.sum((X_train - x_bar) * (y_train - y_bar))
            denominator = np.sum((X_train - x_bar)**2)
            self.weight = numerator / denominator
            self.bias = y_bar - self.weight * x_bar
        
        else:
            # reset w and b from any previous OLS run
            self.weight = 0.0
            self.bias = 0.0
        # gradient descent
            for iter in range(iterations):
                # predict
                y_pred = self.weight * X_train + self.bias
                error = y_train - y_pred

                if iter % 100 == 0:
                    logger.info(f"Average Error after {iter} iterations: {np.mean(error)}")

                # update (gradient descent)
                d_m = (-2 / len(X_train)) * np.sum(X_train * error) # + 2 * self.lambda_ * self.weight # differentiate loss wrt weight or m
                d_b = (-2 / len(X_train)) * np.sum(error)           # differentiate loss wrt bias or b

                self.weight -= self.learning_rate * d_m
                self.bias -= self.learning_rate * d_b

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.weight * X_test + self.bias
        


