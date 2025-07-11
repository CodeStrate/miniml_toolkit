import numpy as np
from loguru import logger

class MultipleLinearRegression:
    def __init__(self, learning_rate: int = 0.01, gradient: bool = True, lambda_: float=0.001):
        self.learning_rate = learning_rate
        self.lambda_ = lambda_
        self.weights = None
        self.gradient= gradient
        self.bias = 0.0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, iterations: int = 1100):
        # OLS fitting
        if not self.gradient:
            X_train_bias = np.c_[np.ones(X_train.shape[0]), X_train] # add a bias column of 1s to X_train
            self.weights = np.linalg.pinv(X_train_bias.T @ X_train_bias) @ X_train_bias.T @ y_train # use pinv to handle singularity
                                                                                                    # i.e. D(X) = 0, then inv(X) not possible

        else:
        # gradient descent
            # reset w and b from prev OLS runs
            n_samples = X_train.shape[0]
            n_features = X_train.shape[1]
            self.weights = np.zeros(n_features)
            self.bias = 0.0

            for iter in range(iterations):
                # predict
                y_pred = X_train @ self.weights + self.bias # MLR equation is X * W + b
                error = y_train - y_pred

                if iter % 100 == 0:
                    logger.info(f"Average Error after {iter} iterations: {np.mean(error)}")

                # update (gradient descent)
                d_m = (-2 / n_samples) * (X_train.T @ error) + 2 * self.lambda_ * self.weights
                d_b = (-2 / n_samples) * np.sum(error)          

                self.weights -= self.learning_rate * d_m
                self.bias -= self.learning_rate * d_b

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if self.gradient:
            return X_test @ self.weights + self.bias
        
    # augment X_test with bias too (OLS)
        else:
            X_test = np.c_[np.ones(X_test.shape[0]), X_test]
            return X_test @ self.weights


