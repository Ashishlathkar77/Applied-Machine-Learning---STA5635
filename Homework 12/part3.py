def mahalanobis_distance(x, mu_k, Sigma_k_inv):
    diff = x - mu_k
    return np.dot(np.dot(diff.T, Sigma_k_inv), diff)

Sigma_k_inv = {}
for k in range(10):
    Sigma_k = np.dot(W_k[k], W_k[k].T) + sigma2_k[k] * np.eye(W_k[k].shape[0])
    Sigma_k_inv[k] = np.linalg.inv(Sigma_k)

def predict_class(x, mu_k, Sigma_k_inv):
    distances = [mahalanobis_distance(x, mu_k[k], Sigma_k_inv[k]) for k in range(10)]
    return np.argmin(distances)

train_predictions = np.array([predict_class(x, mu_k, Sigma_k_inv) for x in X_train])
train_misclassification_error = 1 - accuracy_score(y_train, train_predictions)

test_predictions = np.array([predict_class(x, mu_k, Sigma_k_inv) for x in X_test])
test_misclassification_error = 1 - accuracy_score(y_test, test_predictions)

print(f"Training Misclassification Error: {train_misclassification_error:.4f}")
print(f"Test Misclassification Error: {test_misclassification_error:.4f}")
