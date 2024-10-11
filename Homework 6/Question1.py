import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def load_data(train_data_path, train_labels_path, valid_data_path, valid_labels_path):
    X_train = np.loadtxt(train_data_path)
    y_train = np.loadtxt(train_labels_path)
    X_test = np.loadtxt(valid_data_path)
    y_test = np.loadtxt(valid_labels_path)
    return X_train, y_train, X_test, y_test

def convert_labels(y):
    return 2 * y - 1

def create_bins(X, num_bins=10):
    bins = np.zeros_like(X)
    for i in range(X.shape[1]):
        bins[:, i] = np.digitize(X[:, i], np.histogram(X[:, i], bins=num_bins)[1][:-1])
    return bins

def weak_learner(X, y, weights, num_bins=10):
    best_feature = None
    best_bin = None
    min_loss = float('inf')
    
    n_samples, n_features = X.shape
    
    for feature in range(n_features):
        bins = np.digitize(X[:, feature], np.histogram(X[:, feature], bins=num_bins)[1][:-1])
        
        for b in range(num_bins):
            h = np.ones(n_samples)
            h[bins != b] = -1
            
            loss = np.sum(weights * np.log(1 + np.exp(-y * h)))
            
            if loss < min_loss:
                min_loss = loss
                best_feature = feature
                best_bin = b

    return best_feature, best_bin

def logitboost(X_train, y_train, X_test, y_test, k_values, num_bins=10):
    n_train, n_features = X_train.shape
    y_train_ = convert_labels(y_train)
    y_test_ = convert_labels(y_test)

    f_train = np.zeros(n_train)
    f_test = np.zeros(X_test.shape[0])
    
    train_losses = []
    test_losses = []
    train_errors = []
    test_errors = []
    
    for k in k_values:
        losses = []
        for iteration in range(k):
        
            weights = np.exp(-y_train_ * f_train) / (1 + np.exp(-y_train_ * f_train))
            
            best_feature, best_bin = weak_learner(X_train, y_train_, weights, num_bins)

            bins = np.digitize(X_train[:, best_feature], np.histogram(X_train[:, best_feature], bins=num_bins)[1][:-1])
            h_train = np.ones(n_train)
            h_train[bins != best_bin] = -1
            
            bins_test = np.digitize(X_test[:, best_feature], np.histogram(X_test[:, best_feature], bins=num_bins)[1][:-1])
            h_test = np.ones(X_test.shape[0])
            h_test[bins_test != best_bin] = -1
            
            f_train += h_train
            f_test += h_test
            
            loss = np.sum(np.log(1 + np.exp(-y_train_ * f_train)))
            losses.append(loss)
        
        train_losses.append(losses)
        
        train_error = np.mean(np.sign(f_train) != y_train_)
        test_error = np.mean(np.sign(f_test) != y_test_)
        
        train_errors.append(train_error)
        test_errors.append(test_error)
        
        print(f"Iteration {k}: Train Error: {train_error}, Test Error: {test_error}")
        
    return train_losses, train_errors, test_errors, f_test

def plot_train_loss_vs_iteration(train_losses):
    plt.plot(train_losses[-1], label='Training Loss')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss vs Iteration Number for k=500')
    plt.show()

def plot_misclassification_errors(k_values, train_errors, test_errors):
    plt.plot(k_values, train_errors, label='Training Error')
    plt.plot(k_values, test_errors, label='Test Error')
    plt.xlabel('Boosting Iterations (k)')
    plt.ylabel('Misclassification Error')
    plt.legend()
    plt.title('Misclassification Error vs Boosting Iterations')
    plt.show()

def plot_roc_curve(y_test, f_test):
    fpr, tpr, _ = roc_curve(y_test, f_test)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for k=300')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    
    train_data_path = '/content/gisette/gisette_train.data'
    train_labels_path = '/content/gisette/gisette_train.labels'
    valid_data_path = '/content/gisette/gisette_valid.data'
    valid_labels_path = '/content/gisette/gisette_valid.labels'
    
    X_train, y_train, X_test, y_test = load_data(train_data_path, train_labels_path, valid_data_path, valid_labels_path)
    
    num_bins = 10
    X_train_binned = create_bins(X_train, num_bins)
    X_test_binned = create_bins(X_test, num_bins)
    
    k_values = [10, 30, 100, 300, 500]
    
    train_losses, train_errors, test_errors, f_test = logitboost(X_train_binned, y_train, X_test_binned, y_test, k_values, num_bins)
    
    plot_train_loss_vs_iteration(train_losses)
    
    plot_misclassification_errors(k_values, train_errors, test_errors)
    
    plot_roc_curve(y_test, f_test)
