from miniML_models.models.multiple_linear_regression import MultipleLinearRegression
from utils.standard_scaler import StandardScaler
from utils.metrics import mean_squared_error, r2_score
from utils.model_utils import train_test_split
from loguru import logger
import pandas as pd
import numpy as np

cols = ['CRIM', 'ZS', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df = pd.read_csv("housing.csv", delimiter=r"\s+", names=cols)
df.info()

def run_model(df: pd.DataFrame):

    X = np.array(df.iloc[:, 0:13])
    y = np.array(df['MEDV'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # #instantiate model
    mlr = MultipleLinearRegression(gradient=False)
    mlr.fit(X_train_scaled, y_train)

    #run predictions
    y_pred = mlr.predict(X_test_scaled)

    # return r2 score
    logger.info(f"The MSE is : {mean_squared_error(y_pred, y_test)}")
    logger.info(f"The RMSE is : {mean_squared_error(y_pred, y_test, False)}")
    logger.info(f"The R2 Score is : {r2_score(y_pred, y_test)}")

if __name__ == "__main__":
    run_model(df)