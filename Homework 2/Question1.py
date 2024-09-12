import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv('/content/abalone.csv', header=None)
X = df.iloc[:, :-1].values 
y = df.iloc[:, -1].values  

def split_data(X, y, test_size=0.1, random_state=None):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

from sklearn.metrics import mean_squared_error

def null_model(y_train, y_test):
    mean_train = np.mean(y_train)
    y_pred_train = np.full_like(y_train, mean_train)
    y_pred_test = np.full_like(y_test, mean_train)
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    return mse_train, mse_test

def average_null_model_mse(X, y, iterations=10):
    mse_train_list = []
    mse_test_list = []
    for i in range(iterations):
        X_train, X_test, y_train, y_test = split_data(X, y, random_state=i)
        mse_train, mse_test = null_model(y_train, y_test)
        mse_train_list.append(mse_train)
        mse_test_list.append(mse_test)
    return np.mean(mse_train_list), np.mean(mse_test_list)

null_train_mse, null_test_mse = average_null_model_mse(X, y)
print(f'Average training MSE (null model): {null_train_mse}')
print(f'Average testing MSE (null model): {null_test_mse}')
