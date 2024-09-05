import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

X_train_data = np.loadtxt('/content/Madelon/madelon_train.data')
X_test_data = np.loadtxt('/content/Madelon/madelon_valid.data')
Y_train_labels = np.loadtxt('/content/Madelon/madelon_train.labels')
Y_test_labels = np.loadtxt('/content/Madelon/madelon_valid.labels')

num_features = X_train_data.shape[1]
feature_subset_size = int(np.sqrt(num_features))

tree_counts = [3, 10, 30, 100, 300]

training_errors = []
testing_errors = []

for num_trees in tree_counts:
    random_forest = RandomForestClassifier(n_estimators=num_trees, max_features=feature_subset_size, random_state=42)
    random_forest.fit(X_train_data, Y_train_labels)
    
    training_predictions = random_forest.predict(X_train_data)
    testing_predictions = random_forest.predict(X_test_data)
    
    training_errors.append(1 - accuracy_score(Y_train_labels, training_predictions))
    testing_errors.append(1 - accuracy_score(Y_test_labels, testing_predictions))

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
