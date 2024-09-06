def calc_rf_errors_all(X_train, Y_train, X_test, Y_test, tree_counts):
    tr_errs, te_errs = [], []
    for num_trees in tree_counts:
        rf = RandomForestClassifier(n_estimators=num_trees, max_features=None, random_state=42)
        rf.fit(X_train, Y_train)
        tr_errs.append(1 - accuracy_score(Y_train, rf.predict(X_train)))
        te_errs.append(1 - accuracy_score(Y_test, rf.predict(X_test)))
    return tr_errs, te_errs


if __name__ == "__main__":
   
    X_train, X_test, Y_train, Y_test = load_data(madelon_train_data_path, madelon_test_data_path, madelon_train_labels_path, madelon_test_labels_path)


    tree_counts = [3, 10, 30, 100, 300]
    tr_errs, te_errs = calc_rf_errors_all(X_train, Y_train, X_test, Y_test, tree_counts)
    plot_errors(tree_counts, tr_errs, te_errs, 'Random Forest Errors vs Number of Trees with All Features (Madelon)')
    print_error_summary(tree_counts, tr_errs, te_errs)
