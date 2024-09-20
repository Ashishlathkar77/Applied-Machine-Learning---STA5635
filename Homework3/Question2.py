X_train = np.loadtxt('/content/madelon_train.data')
y_train = np.loadtxt('/content/madelon_train.labels')
X_test = np.loadtxt('/content/madelon_valid.data')
y_test = np.loadtxt('/content/madelon_valid.labels')


X_train_std, y_train_std, X_test_std, y_test_std = data_preprocess(X_train, y_train, X_test, y_test)


minimize_hinge_loss(X_train_std, y_train_std, X_test_std, y_test_std, iter=300, learning_rate=0.01, lambda_=0.001)
