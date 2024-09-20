X_train = np.genfromtxt('/content/dexter_train.csv', delimiter=',')
y_train = np.genfromtxt('/content/dexter_train.labels')
X_test = np.genfromtxt('/content/dexter_valid.csv', delimiter=',')
y_test = np.genfromtxt('/content/dexter_valid.labels')


X_train_std, y_train_std, X_test_std, y_test_std = data_preprocess(X_train, y_train, X_test, y_test)


minimize_hinge_loss(X_train_std, y_train_std, X_test_std, y_test_std, iter=300, learning_rate=0.01, lambda_=0.001)
