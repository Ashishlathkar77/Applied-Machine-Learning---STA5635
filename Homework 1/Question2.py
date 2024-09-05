train_data_path = '/content/Satimage/X.dat'
test_data_path = '/content/Satimage/Xtest.dat'
train_labels_path = '/content/Satimage/Y.dat'
test_labels_path = '/content/Satimage/Ytest.dat'

max_depth = 12

X_train, X_test, Y_train, Y_test = load_data(train_data_path, test_data_path, train_labels_path, test_labels_path)

train_errors, test_errors = calculate_errors(X_train, Y_train, X_test, Y_test, max_depth)

optimal_tree_depth, minimum_test_error = find_optimal_depth(test_errors)

plot_errors(train_errors, test_errors, max_depth)

print(f'Optimal Tree Depth: {optimal_tree_depth}')
print(f'Minimum Test Error: {minimum_test_error:.4f}')
print_error_summary(train_errors, test_errors)
