q = 20
mu_k = {}
W_k = {}
sigma2_k = {}

for k in range(10):
    X_class_k = X_train[y_train == k]
    pca = PCA(n_components=q)
    pca.fit(X_class_k)
    mu_k[k] = np.mean(X_class_k, axis=0)
    W_k[k] = pca.components_.T
    reconstructed_data = pca.inverse_transform(pca.transform(X_class_k))
    residual = X_class_k - reconstructed_data
    sigma2_k[k] = np.mean(np.var(residual, axis=0))

print("Noise variance (σ^2_k) for each class:")
for k in range(10):
    print(f"Class {k}: σ^2_k = {sigma2_k[k]:.6f}")
