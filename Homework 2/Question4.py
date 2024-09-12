from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

def rf_regression(X_train, y_train, X_test, y_test, n_trees):
    model = RandomForestRegressor(n_estimators=n_trees, oob_score=True, random_state=42)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    oob_r2 = model.oob_score_ 

    return train_mse, test_mse, train_r2, test_r2, oob_r2

def average_rf_metrics(X, y, iterations=10, n_trees_list=[10, 30, 100, 300]):
    rf_data = {
        'n_estimators': [],
        'avg_tr_mse': [],
        'std_tr_mse': [],
        'avg_ts_mse': [],
        'std_ts_mse': [],
        'avg_tr_r2': [],
        'std_tr_r2': [],
        'avg_ts_r2': [],
        'std_ts_r2': [],
        'avg_oob_r2': [],
        'std_oob_r2': []
    }

    for n_trees in n_trees_list:
        train_mse = []
        test_mse = []
        train_r2 = []
        test_r2 = []
        oob_r2 = []

        for i in range(iterations):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=i)
            t_mse, ts_mse, t_r2, ts_r2, oob_r = rf_regression(X_train, y_train, X_test, y_test, n_trees)
            train_mse.append(t_mse)
            test_mse.append(ts_mse)
            train_r2.append(t_r2)
            test_r2.append(ts_r2)
            oob_r2.append(oob_r)

        rf_data['n_estimators'].append(n_trees)
        rf_data['avg_tr_mse'].append(np.mean(train_mse))
        rf_data['std_tr_mse'].append(np.std(train_mse))
        rf_data['avg_ts_mse'].append(np.mean(test_mse))
        rf_data['std_ts_mse'].append(np.std(test_mse))
        rf_data['avg_tr_r2'].append(np.mean(train_r2))
        rf_data['std_tr_r2'].append(np.std(train_r2))
        rf_data['avg_ts_r2'].append(np.mean(test_r2))
        rf_data['std_ts_r2'].append(np.std(test_r2))
        rf_data['avg_oob_r2'].append(np.mean(oob_r2))
        rf_data['std_oob_r2'].append(np.std(oob_r2))

    return pd.DataFrame(rf_data)

rf_metrics = average_rf_metrics(X, y, 10)

print(rf_metrics.to_string(index=False))
