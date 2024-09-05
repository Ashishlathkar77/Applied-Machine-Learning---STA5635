train_data_path = '/content/Madelon/madelon_train.data'
test_data_path = '/content/Madelon/madelon_valid.data'
train_labels_path = '/content/Madelon/madelon_train.labels'
test_labels_path = '/content/Madelon/madelon_valid.labels'

tree_counts = [3, 10, 30, 100, 300]

def load_data(train_data_path, test_data_path, train_labels_path, test_labels_path):
    return (np.loadtxt(train_data_path), np.loadtxt(test_data_path),
            np.loadtxt(train_labels_path), np.loadtxt(test_labels_path))

def calculate_rf_errors(X_train, Y_train, X_test, Y_test, tree_counts):
    train_errors = []
    test_errors = []
    for num_trees in tree_counts:
        rf_model = RandomForestClassifier(n_estimators=num_trees, max_features=None, random_state=42)
        rf_model.fit(X_train, Y_train)
        train_errors.append(1 - accuracy_score(Y_train, rf_model.predict(X_train)))
        test_errors.append(1 - accuracy_score(Y_test, rf_model.predict(X_test)))
    return train_errors, test_errors

def plot_errors(tree_counts, train_errors, test_errors):
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(tree_counts, train_errors, label='Training Error', marker='o')
    plt.plot(tree_counts, test_errors, label='Test Error', marker='o')
    plt.xlabel('Number of Trees (k)')
    plt.ylabel('Misclassification Error')
    plt.title('Training and Test Errors vs Number of Trees with All Features')
    plt.legend()
    plt.grid(True)
    plt.show()

def print_error_summary(tree_counts, train_errors, test_errors):
    print(f"{'Number of Trees (k)':<20}{'Training Error':<20}{'Test Error':<20}")
    for k, train_err, test_err in zip(tree_counts, train_errors, test_errors):
        print(f"{k:<20}{train_err:<20.4f}{test_err:<20.4f}")

X_train, X_test, Y_train, Y_test = load_data(train_data_path, test_data_path, train_labels_path, test_labels_path)
train_errors, test_errors = calculate_rf_errors(X_train, Y_train, X_test, Y_test, tree_counts)
plot_errors(tree_counts, train_errors, test_errors)
print_error_summary(tree_counts, train_errors, test_errors)
