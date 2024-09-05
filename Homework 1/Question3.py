import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

train_data_path = '/content/Madelon/madelon_train.data'
test_data_path = '/content/Madelon/madelon_valid.data'
train_labels_path = '/content/Madelon/madelon_train.labels'
test_labels_path = '/content/Madelon/madelon_valid.labels'

tree_counts = [3, 10, 30, 100, 300]

X_train, X_test, Y_train, Y_test = load_data(train_data_path, test_data_path, train_labels_path, test_labels_path)

def calculate_rf_errors(X_train, Y_train, X_test, Y_test, tree_counts, feature_subset_size):
    training_errors = []
    testing_errors = []
    for num_trees in tree_counts:
        rf = RandomForestClassifier(n_estimators=num_trees, max_features=feature_subset_size, random_state=42)
        rf.fit(X_train, Y_train)
        
        train_pred = rf.predict(X_train)
        test_pred = rf.predict(X_test)
        
        training_errors.append(1 - accuracy_score(Y_train, train_pred))
        testing_errors.append(1 - accuracy_score(Y_test, test_pred))
    return training_errors, testing_errors

num_features = X_train.shape[1]
feature_subset_size = int(np.sqrt(num_features))
training_errors, testing_errors = calculate_rf_errors(X_train, Y_train, X_test, Y_test, tree_counts, feature_subset_size)

plt.figure(figsize=(10, 6), dpi=300)
plt.plot(tree_counts, training_errors, label='Training Error', marker='o')
plt.plot(tree_counts, testing_errors, label='Test Error', marker='o')
plt.xlabel('Number of Trees (k)')
plt.ylabel('Misclassification Error')
plt.title('Training and Test Errors vs Number of Trees for Madelon Dataset')
plt.legend()
plt.grid(True)
plt.show()

print(f"{'Number of Trees (k)':<20}{'Training Error':<20}{'Test Error':<20}")
for idx in range(len(tree_counts)):
    print(f"{tree_counts[idx]:<20}{training_errors[idx]:<20.4f}{testing_errors[idx]:<20.4f}")
