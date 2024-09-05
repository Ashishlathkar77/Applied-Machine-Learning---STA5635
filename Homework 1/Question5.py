import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

train_data = np.loadtxt('/content/Madelon/madelon_train.data')
test_data = np.loadtxt('/content/Madelon/madelon_valid.data')
train_labels = np.loadtxt('/content/Madelon/madelon_train.labels')
test_labels = np.loadtxt('/content/Madelon/madelon_valid.labels')

tree_counts = [3, 10, 30, 100, 300]

train_errors = []
test_errors = []

for num_trees in tree_counts:
    rf_model = RandomForestClassifier(n_estimators=num_trees, max_features=None, random_state=42)
    rf_model.fit(train_data, train_labels)
    
    train_predictions = rf_model.predict(train_data)
    test_predictions = rf_model.predict(test_data)
    
    train_errors.append(1 - accuracy_score(train_labels, train_predictions))
    test_errors.append(1 - accuracy_score(test_labels, test_predictions))

plt.figure(figsize=(10, 6), dpi=300)
plt.plot(tree_counts, train_errors, label='Training Error', marker='o')
plt.plot(tree_counts, test_errors, label='Test Error', marker='o')
plt.xlabel('Number of Trees (k)')
plt.ylabel('Misclassification Error')
plt.title('Training and Test Errors vs Number of Trees with All Features')
plt.legend()
plt.grid(True)
plt.show()

print(f"{'Number of Trees (k)':<20}{'Training Error':<20}{'Test Error':<20}")
for idx in range(len(tree_counts)):
    print(f"{tree_counts[idx]:<20}{train_errors[idx]:<20.4f}{test_errors[idx]:<20.4f}")
