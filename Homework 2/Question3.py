from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

def regression_tree_metrics(X_train, y_train, X_test, y_test, max_depth):
    model = DecisionTreeRegressor(max_depth=max_depth)
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    return mse_train, mse_test, r2_train, r2_test

def average_tree_metrics(X, y, iterations=10, max_depth=7):
    depth_range = range(1, max_depth + 1)
    mse_train_avg = []
    mse_test_avg = []
    r2_train_avg = []
    r2_test_avg = []
    for depth in depth_range:
        mse_train_list = []
        mse_test_list = []
        r2_train_list = []
        r2_test_list = []
        for i in range(iterations):
            X_train, X_test, y_train, y_test = split_data(X, y, random_state=i)
            mse_train, mse_test, r2_train, r2_test = regression_tree_metrics(X_train, y_train, X_test, y_test, depth)
            mse_train_list.append(mse_train)
            mse_test_list.append(mse_test)
            r2_train_list.append(r2_train)
            r2_test_list.append(r2_test)
        mse_train_avg.append(np.mean(mse_train_list))
        mse_test_avg.append(np.mean(mse_test_list))
        r2_train_avg.append(np.mean(r2_train_list))
        r2_test_avg.append(np.mean(r2_test_list))

    return {
        'depth': depth_range,
        'mse_train_avg': mse_train_avg,
        'mse_test_avg': mse_test_avg,
        'r2_train_avg': r2_train_avg,
        'r2_test_avg': r2_test_avg
    }

tree_metrics = average_tree_metrics(X, y)

import matplotlib.pyplot as plt

def plot_tree_metrics(tree_metrics, null_train_mse, null_test_mse):
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(tree_metrics['depth'], tree_metrics['mse_train_avg'], marker='o', label='Training MSE')
    plt.plot(tree_metrics['depth'], tree_metrics['mse_test_avg'], marker='o', label='Testing MSE')
    plt.axhline(y=null_train_mse, color='r', linestyle='--', label='Null Model MSE (Train)')
    plt.axhline(y=null_test_mse, color='b', linestyle='--', label='Null Model MSE (Test)')
    plt.xlabel('Tree Depth')
    plt.ylabel('MSE')
    plt.title('Tree Depth vs MSE')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(tree_metrics['depth'], tree_metrics['r2_train_avg'], marker='o', label='Training R2')
    plt.plot(tree_metrics['depth'], tree_metrics['r2_test_avg'], marker='o', label='Testing R2')
    plt.xlabel('Tree Depth')
    plt.ylabel('R2')
    plt.title('Tree Depth vs R2')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_tree_metrics(tree_metrics, null_train_mse, null_test_mse)
