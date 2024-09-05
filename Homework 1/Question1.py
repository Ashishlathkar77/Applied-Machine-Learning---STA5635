import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def load_data(train_data_path, test_data_path, train_labels_path, test_labels_path):
    X_train = np.loadtxt(train_data_path)
    X_test = np.loadtxt(test_data_path)
    Y_train = np.loadtxt(train_labels_path)
    Y_test = np.loadtxt(test_labels_path)
    return X_train, X_test, Y_train, Y_test

def calculate_errors(X_train, Y_train, X_test, Y_test, max_depth):
    train_errors = []
    test_errors = []
    for d in range(1, max_depth + 1):
        clf = DecisionTreeClassifier(max_depth=d, random_state=42)
        clf.fit(X_train, Y_train)
        
        train_pred = clf.predict(X_train)
        test_pred = clf.predict(X_test)
        train_errors.append(1 - accuracy_score(Y_train, train_pred))
        test_errors.append(1 - accuracy_score(Y_test, test_pred))
    return train_errors, test_errors

def find_optimal_depth(test_errors):
    min_error = min(test_errors)
    best_depth = test_errors.index(min_error) + 1
    return best_depth, min_error

def plot_errors(train_errors, test_errors, max_depth):
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(range(1, max_depth + 1), train_errors, label='Train Error', marker='o')
    plt.plot(range(1, max_depth + 1), test_errors, label='Test Error', marker='o')
    plt.xlabel('Tree Depth')
    plt.ylabel('Misclassification Rate')
    plt.title('Error Rates vs Tree Depth')
    plt.legend()
    plt.grid(True)
    plt.show()

def print_error_summary(train_errors, test_errors):
    print(f"{'Depth':<6} {'Train Error':<15} {'Test Error':<15}")
    for i in range(len(train_errors)):
        print(f"{i + 1:<6} {train_errors[i]:<15.4f} {test_errors[i]:<15.4f}")

train_data_path = '/content/Madelon/madelon_train.data'
test_data_path = '/content/Madelon/madelon_valid.data'
train_labels_path = '/content/Madelon/madelon_train.labels'
test_labels_path = '/content/Madelon/madelon_valid.labels'

max_depth = 12

X_train, X_test, Y_train, Y_test = load_data(train_data_path, test_data_path, train_labels_path, test_labels_path)

train_errors, test_errors = calculate_errors(X_train, Y_train, X_test, Y_test, max_depth)

best_depth, min_test_error = find_optimal_depth(test_errors)

plot_errors(train_errors, test_errors, max_depth)

print(f'Optimal Tree Depth: {best_depth}')
print(f'Minimum Test Error: {min_test_error:.4f}')
print_error_summary(train_errors, test_errors)
