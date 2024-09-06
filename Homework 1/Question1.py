import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


madelon_train_data_path = '/content/Madelon/madelon_train.data'
madelon_test_data_path = '/content/Madelon/madelon_valid.data'
madelon_train_labels_path = '/content/Madelon/madelon_train.labels'
madelon_test_labels_path = '/content/Madelon/madelon_valid.labels'


def load_data(train_path, test_path, train_labels_path, test_labels_path):
    X_train = np.loadtxt(train_path)
    X_test = np.loadtxt(test_path)
    Y_train = np.loadtxt(train_labels_path)
    Y_test = np.loadtxt(test_labels_path)
    return X_train, X_test, Y_train, Y_test


def calc_dt_errors(X_train, Y_train, X_test, Y_test, max_depth):
    tr_errs, te_errs = [], []
    for d in range(1, max_depth + 1):
        dt = DecisionTreeClassifier(max_depth=d, random_state=42)
        dt.fit(X_train, Y_train)
        tr_errs.append(1 - accuracy_score(Y_train, dt.predict(X_train)))
        te_errs.append(1 - accuracy_score(Y_test, dt.predict(X_test)))
    return tr_errs, te_errs


def find_optimal_depth(te_errs):
    min_err = min(te_errs)
    return te_errs.index(min_err) + 1, min_err


def plot_errors(x_vals, tr_errs, te_errs, title):
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(x_vals, tr_errs, label='Train Error', marker='o')
    plt.plot(x_vals, te_errs, label='Test Error', marker='o')
    plt.xlabel('Depth')
    plt.ylabel('Error Rate')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def print_error_summary(x_vals, tr_errs, te_errs):
    print(f"{'Depth':<10}{'Train Error':<15}{'Test Error':<15}")
    for x, tr_err, te_err in zip(x_vals, tr_errs, te_errs):
        print(f"{x:<10}{tr_err:<15.4f}{te_err:<15.4f}")


if __name__ == "__main__":


    X_train, X_test, Y_train, Y_test = load_data(madelon_train_data_path, madelon_test_data_path, madelon_train_labels_path, madelon_test_labels_path)
  
    max_depth = 12
    tr_errs, te_errs = calc_dt_errors(X_train, Y_train, X_test, Y_test, max_depth)
    best_depth, min_te_err = find_optimal_depth(te_errs)
    plot_errors(range(1, max_depth + 1), tr_errs, te_errs, 'Decision Tree Errors vs Depth (Madelon)')
    print(f'Optimal Tree Depth: {best_depth}')
    print(f'Minimum Test Error: {min_te_err:.4f}')
    print_error_summary(range(1, max_depth + 1), tr_errs, te_errs)
