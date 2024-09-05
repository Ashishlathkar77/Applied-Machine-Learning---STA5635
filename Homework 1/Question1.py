import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

X_train_data = np.loadtxt('/content/Madelon/madelon_train.data')
X_test_data = np.loadtxt('/content/Madelon/madelon_valid.data')
Y_train_labels = np.loadtxt('/content/Madelon/madelon_train.labels')
Y_test_labels = np.loadtxt('/content/Madelon/madelon_valid.labels')

train_errors_list = []
test_errors_list = []

for depth in range(1, 13):
    decision_tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    decision_tree.fit(X_train_data, Y_train_labels)
    
    train_predictions = decision_tree.predict(X_train_data)
    test_predictions = decision_tree.predict(X_test_data)
    
    train_errors_list.append(1 - accuracy_score(Y_train_labels, train_predictions))
    test_errors_list.append(1 - accuracy_score(Y_test_labels, test_predictions))

min_test_error_value = min(test_errors_list)
optimal_tree_depth = test_errors_list.index(min_test_error_value) + 1

plt.figure(figsize=(10, 6), dpi=300)
plt.plot(range(1, 13), train_errors_list, label='Training Error', marker='o')
plt.plot(range(1, 13), test_errors_list, label='Test Error', marker='o')
plt.xlabel('Depth of Tree')
plt.ylabel('Misclassification Error')
plt.title('Training and Test Errors vs Tree Depth')
plt.legend()
plt.grid(True)
plt.show()

print(f'Optimal Tree Depth: {optimal_tree_depth}')
print(f'Minimum Test Error: {min_test_error_value:.4f}')

print(f"{'Depth':<6} {'Training Error':<15} {'Test Error':<15}")
for index in range(12):
    print(f"{index + 1:<6} {train_errors_list[index]:<15.4f} {test_errors_list[index]:<15.4f}")
