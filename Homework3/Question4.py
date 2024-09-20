import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


X_train = np.loadtxt('/content/gisette_train.data')
y_train = np.loadtxt('/content/gisette_train.labels')
X_test = np.loadtxt('/content/gisette_valid.data')
y_test = np.loadtxt('/content/gisette_valid.labels')


mean_X_train = np.mean(X_train, axis=0)
std_X_train = np.std(X_train, axis=0)
std_X_train[std_X_train == 0] = 1  


X_train_normalized = (X_train - mean_X_train) / std_X_train
X_test_normalized = (X_test - mean_X_train) / std_X_train


weights = np.zeros(X_train_normalized.shape[1])


lambda_reg = 0.001


huber_threshold = 0.1


def huberized_svm_loss(scores, h):
    return np.where(scores >= 1,
                    0,
                    np.where(scores <= 1 - h,
                             1 - scores - h / 2,
                             (1 - scores) ** 2 / (2 * h)))


def huberized_svm_loss_derivative(scores, h):
    return np.where(scores >= 1,
                    0,
                    np.where(scores <= 1 - h,
                             -1,
                             -(1 - scores) / h))


def calculate_loss_and_gradient(weights, features, labels, lambda_reg, h):
    num_samples = features.shape[0]
    scores = labels * (features @ weights)  
    loss_values = huberized_svm_loss(scores, h)
    total_loss = np.mean(loss_values) + lambda_reg * np.dot(weights, weights)
   
    loss_derivative = huberized_svm_loss_derivative(scores, h)
    gradient = (1 / num_samples) * np.sum((loss_derivative * labels)[:, np.newaxis] * features, axis=0) + 2 * lambda_reg * weights
    return total_loss, gradient


iterations = 300
learning_rate = 0.01


loss_history = []


for iteration in range(iterations):
    current_loss, gradient = calculate_loss_and_gradient(weights, X_train_normalized, y_train, lambda_reg, huber_threshold)
    loss_history.append(current_loss)
    weights -= learning_rate * gradient  
    if iteration % 10 == 0:
        print(f"Iteration {iteration}, Loss: {current_loss:.4f}")


train_scores = X_train_normalized @ weights
train_predictions = np.sign(train_scores)
train_error_rate = np.mean(train_predictions != y_train)
print(f"Training misclassification error: {train_error_rate:.4f}")


test_scores = X_test_normalized @ weights
test_predictions = np.sign(test_scores)
test_error_rate = np.mean(test_predictions != y_test)
print(f"Test misclassification error: {test_error_rate:.4f}")


plt.figure()
plt.plot(range(iterations), loss_history, label='Huberized SVM Loss')
plt.xlabel('Iteration Number')
plt.ylabel('Training Loss')
plt.title('Training Loss vs Iteration Number')
plt.legend()
plt.grid(True)
plt.show()


fpr_train, tpr_train, _ = roc_curve(y_train, train_scores)
roc_auc_train = auc(fpr_train, tpr_train)


fpr_test, tpr_test, _ = roc_curve(y_test, test_scores)
roc_auc_test = auc(fpr_test, tpr_test)


plt.figure()
plt.plot(fpr_train, tpr_train, label='Training ROC (AUC = %0.2f)' % roc_auc_train)
plt.plot(fpr_test, tpr_test, label='Test ROC (AUC = %0.2f)' % roc_auc_test)
plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()
