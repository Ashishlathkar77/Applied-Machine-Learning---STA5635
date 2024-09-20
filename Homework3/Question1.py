import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.preprocessing import StandardScaler


def data_preprocess(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_std = pd.DataFrame(scaler.fit_transform(X_train))
    X_test_std = pd.DataFrame(scaler.transform(X_test))
 
    y_train = (y_train + 1) // 2 * 2 - 1
    y_test = (y_test + 1) // 2 * 2 - 1
   
    return X_train_std, y_train, X_test_std, y_test


def hinge_loss(X, y, w):
    return np.maximum(0, 1 - y * np.dot(X, w))


def grad(X, y, w, lambda_):
    margin = 1 - y * np.dot(X, w)
    indicator = (margin > 0).astype(int)
    grad_w = -np.dot(X.T, y * indicator) / X.shape[0] + 2 * lambda_ * w
    return grad_w
def minimize_hinge_loss(X_train, y_train, X_test, y_test, iter=300, learning_rate=0.01, lambda_=0.001):
    w = np.zeros(X_train.shape[1])
   
    train_loss = []
    train_errors = []
    test_errors = []
   
    fpr_train, tpr_train, auc_train = [], [], []
    fpr_test, tpr_test, auc_test = [], [], []
   
    for i in range(iter):
        loss = np.mean(hinge_loss(X_train, y_train, w)) + lambda_ * np.dot(w, w)
        train_loss.append(loss)
      
        w -= learning_rate * grad(X_train, y_train, w, lambda_)
     
        train_pred = np.sign(np.dot(X_train, w))
        test_pred = np.sign(np.dot(X_test, w))
       
        train_error = 1 - accuracy_score(y_train, train_pred)
        test_error = 1 - accuracy_score(y_test, test_pred)
       
        train_errors.append(train_error)
        test_errors.append(test_error)
       
        fpr, tpr, _ = roc_curve(y_train, np.dot(X_train, w))
        roc_auc = auc(fpr, tpr)
        fpr_train.append(fpr)
        tpr_train.append(tpr)
        auc_train.append(roc_auc)
       
        fpr, tpr, _ = roc_curve(y_test, np.dot(X_test, w))
        roc_auc = auc(fpr, tpr)
        fpr_test.append(fpr)
        tpr_test.append(tpr)
        auc_test.append(roc_auc)
   
    plt.figure(figsize=(10, 6))
    plt.plot(range(iter), train_loss)
    plt.title("Training Loss vs Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Training Loss")
    plt.grid(True)
    plt.show()
 
    best_iter = np.argmin(test_errors)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr_train[best_iter], tpr_train[best_iter], label=f"Train ROC (AUC = {auc_train[best_iter]:.4f})")
    plt.plot(fpr_test[best_iter], tpr_test[best_iter], label=f"Test ROC (AUC = {auc_test[best_iter]:.4f})")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.show()


    print(f"Training Misclassification Error: {train_errors[best_iter]:.4f}")
    print(f"Test Misclassification Error: {test_errors[best_iter]:.4f}")


X_train = np.loadtxt('/content/gisette_train.data')
y_train = np.loadtxt('/content/gisette_train.labels')
X_test = np.loadtxt('/content/gisette_valid.data')
y_test = np.loadtxt('/content/gisette_valid.labels')


X_train_std, y_train_std, X_test_std, y_test_std = data_preprocess(X_train, y_train, X_test, y_test)


minimize_hinge_loss(X_train_std, y_train_std, X_test_std, y_test_std, iter=300, learning_rate=0.01, lambda_=0.001)
