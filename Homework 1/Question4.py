import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

X_train_data = np.loadtxt('/content/Madelon/madelon_train.data')
X_test_data = np.loadtxt('/content/Madelon/madelon_valid.data')
Y_train_labels = np.loadtxt('/content/Madelon/madelon_train.labels')
Y_test_labels = np.loadtxt('/content/Madelon/madelon_valid.labels')

num_features = X_train_data.shape[1]
subset_size = int(np.log2(num_features))

tree_counts = [3, 10, 30, 100, 300]

train_errors = []
test_errors = []

for num_trees in tree_counts:
    rf_model = RandomForestClassifier(n_estimators=num_trees, max_features=subset_size, random_state=42)
    rf_model.fit(X_train_data, Y_train_labels)
    
    train_preds = rf_model.predict(X_train_data)
    test_preds = rf_model.predict(X_test_data)
    
    train_errors.append(1 - accuracy_score(Y_train_labels, train_preds))
    test_errors.append(1 - accuracy_score(Y_test_labels, test_preds))

plt.figure(figsize=(10, 6), dpi=300)
plt.plot(tree_counts, train_errors, label='Training Error', marker='o')
plt.plot(tree_counts, test_errors, label='Test Error', marker='o')
plt.xlabel('Number of Trees (k)')
plt.ylabel('Misclassification Error')
plt.title('Training and Test Errors vs Number of Trees with log2(Features) Subset')
plt.legend()
plt.grid(True)
plt.show()

print(f"{'Number of Trees (k)':<20}{'Training Error':<20}{'Test Error':<20}")
for idx in range(len(tree_counts)):
    print(f"{tree_counts[idx]:<20}{train_errors[idx]:<20.4f}{test_errors[idx]:<20.4f}")
