satimage_train_data_path = '/content/Satimage/X.dat'
satimage_test_data_path = '/content/Satimage/Xtest.dat'
satimage_train_labels_path = '/content/Satimage/Y.dat'
satimage_test_labels_path = '/content/Satimage/Ytest.dat'


def load_data(train_path, test_path, train_labels_path, test_labels_path):
    X_train = np.loadtxt(train_path)
    X_test = np.loadtxt(test_path)
    Y_train = np.loadtxt(train_labels_path)
    Y_test = np.loadtxt(test_labels_path)
    return X_train, X_test, Y_train, Y_test


if __name__ == "__main__":
    X_train, X_test, Y_train, Y_test = load_data(satimage_train_data_path, satimage_test_data_path, satimage_train_labels_path, satimage_test_labels_path)


    max_depth = 12
    tr_errs, te_errs = calc_dt_errors(X_train, Y_train, X_test, Y_test, max_depth)
    best_depth, min_te_err = find_optimal_depth(te_errs)
    plot_errors(range(1, max_depth + 1), tr_errs, te_errs, 'Decision Tree Errors vs Depth (Satimage)')
    print(f'Optimal Tree Depth: {best_depth}')
    print(f'Minimum Test Error: {min_te_err:.4f}')
    print_error_summary(range(1, max_depth + 1), tr_errs, te_errs)
