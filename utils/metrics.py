import numpy as np

def r2_score(y_pred: np.ndarray, y_test: np.ndarray) -> float:
    y_test_bar = np.mean(y_test)
    residual_squared_error_sum = np.sum((y_test - y_pred)**2)
    total_squared_dev_sum = np.sum((y_test - y_test_bar)**2)
    return np.round(1 - (residual_squared_error_sum/total_squared_dev_sum), 3)

def mean_squared_error(y_pred: np.ndarray, y_test: np.ndarray, squared: bool = True) -> float:
    if squared:
        return np.round(np.sum((y_test - y_pred)**2) / len(y_test), 3)
    else:
        return np.round(np.sqrt(np.sum((y_test - y_pred)**2) / len(y_test)), 3)