train_data_path = '/content/Madelon/madelon_train.data'
test_data_path = '/content/Madelon/madelon_valid.data'
train_labels_path = '/content/Madelon/madelon_train.labels'
test_labels_path = '/content/Madelon/madelon_valid.labels'

tree_counts = [3, 10, 30, 100, 300]

X_train, X_test, Y_train, Y_test = load_data(train_data_path, test_data_path, train_labels_path, test_labels_path)

num_features = X_train.shape[1]
subset_size = int(np.log2(num_features))

def calculate_rf_errors(X_train, Y_train, X_test, Y_test, tree_counts, subset_size):
    train_errors = []
    test_errors = []
    for num_trees in tree_counts:
        rf_model = RandomForestClassifier(n_estimators=num_trees, max_features=subset_size, random_state=42)
        rf_model.fit(X_train, Y_train)

        train_preds = rf_model.predict(X_train)
        test_preds = rf_model.predict(X_test)
        train_errors.append(1 - accuracy_score(Y_train, train_preds))
        test_errors.append(1 - accuracy_score(Y_test, test_preds))
    return train_errors, test_errors

train_errors, test_errors = calculate_rf_errors(X_train, Y_train, X_test, Y_test, tree_counts, subset_size)

def plot_errors(tree_counts, train_errors, test_errors):
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(tree_counts, train_errors, label='Training Error', marker='o')
    plt.plot(tree_counts, test_errors, label='Test Error', marker='o')
    plt.xlabel('Number of Trees (k)')
    plt.ylabel('Misclassification Error')
    plt.title('Training and Test Errors vs Number of Trees with log2(Features) Subset')
    plt.legend()
    plt.grid(True)
    plt.show()

def print_error_summary(tree_counts, train_errors, test_errors):
    print(f"{'Number of Trees (k)':<20}{'Training Error':<20}{'Test Error':<20}")
    for idx in range(len(tree_counts)):
        print(f"{tree_counts[idx]:<20}{train_errors[idx]:<20.4f}{test_errors[idx]:<20.4f}")

plot_errors(tree_counts, train_errors, test_errors)
print_error_summary(tree_counts, train_errors, test_errors)
