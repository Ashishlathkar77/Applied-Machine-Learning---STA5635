import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def ols_regression(X_train, y_train, X_test, y_test, lambda_=0.01):
    X_train_with_intercept = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    X_test_with_intercept = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

    n_features = X_train_with_intercept.shape[1]
    I = np.eye(n_features)
    I[0, 0] = 0  

    XtX = np.dot(X_train_with_intercept.T, X_train_with_intercept) + lambda_ * I
    XtX_inv = np.linalg.inv(XtX)
    XtY = np.dot(X_train_with_intercept.T, y_train)
    coef = np.dot(XtX_inv, XtY)

    y_train_pred = np.dot(X_train_with_intercept, coef)
    y_test_pred = np.dot(X_test_with_intercept, coef)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    log_det = np.linalg.slogdet(XtX)[1]

    return train_mse, test_mse, train_r2, test_r2, log_det

data = pd.read_csv('/content/abalone.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

n_splits = 10
ols_train_mse = []
ols_test_mse = []
ols_train_r2 = []
ols_test_r2 = []
ols_log_det = []

for _ in range(n_splits):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=None)
    train_mse, test_mse, train_r2, test_r2, log_det = ols_regression(X_train, y_train, X_test, y_test, lambda_=0.01)
    ols_train_mse.append(train_mse)
    ols_test_mse.append(test_mse)
    ols_train_r2.append(train_r2)
    ols_test_r2.append(test_r2)
    ols_log_det.append(log_det)

print(f"OLS Regression - Average Training MSE: {np.mean(ols_train_mse)}, Average Test MSE: {np.mean(ols_test_mse)}")
print(f"OLS Regression - Average Training R^2: {np.mean(ols_train_r2)}, Average Test R^2: {np.mean(ols_test_r2)}")
print(f"OLS Regression - Average Log Det of X^T.X + λ.Ip: {np.mean(ols_log_det)}")
print(f"OLS Regression - Training MSE Std Dev: {np.std(ols_train_mse)}, Test MSE Std Dev: {np.std(ols_test_mse)}")
print(f"OLS Regression - Training R^2 Std Dev: {np.std(ols_train_r2)}, Test R^2 Std Dev: {np.std(ols_test_r2)}")
print(f"OLS Regression - Log Det Std Dev: {np.std(ols_log_det)}")
