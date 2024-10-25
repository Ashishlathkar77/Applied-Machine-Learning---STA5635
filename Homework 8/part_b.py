import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment
from numpy.linalg import inv

num_runs = 10
num_samples = 500

kl_divergences = []
kmeans_iso_accuracies = []
kmeans_full_accuracies = []
em_accuracies = []
kmeans_iso_ari = []
kmeans_full_ari = []
em_ari = []

for run in range(num_runs):
    M = np.random.normal(0, 1, (2, 2))
    U, _, _ = np.linalg.svd(M)

    D = np.diag([100, 1])
    Sigma = U @ D @ U.T

    X_Q = np.random.multivariate_normal(mean=[0, 0], cov=Sigma, size=num_samples)
    X_P = np.random.multivariate_normal(mean=[10, 0], cov=Sigma, size=num_samples)
    X = np.vstack((X_Q, X_P))
    y_true = np.hstack((np.zeros(num_samples), np.ones(num_samples)))

    def kl_divergence(mu1, Sigma1, mu2, Sigma2):
        d = len(mu1)
        inv_Sigma2 = inv(Sigma2)
        term1 = np.log(np.linalg.det(Sigma2) / np.linalg.det(Sigma1))
        term2 = np.trace(inv_Sigma2 @ Sigma1)
        term3 = (mu2 - mu1).T @ inv_Sigma2 @ (mu2 - mu1)
        return 0.5 * (term1 - d + term2 + term3)

    mu_P = np.array([10, 0])
    mu_Q = np.array([0, 0])
    D_KL = kl_divergence(mu_P, Sigma, mu_Q, Sigma)
    kl_divergences.append(D_KL)

    kmeans_iso = KMeans(n_clusters=2, n_init=10, random_state=run)
    y_kmeans_iso = kmeans_iso.fit_predict(X)

    kmeans_full = KMeans(n_clusters=2, n_init=10, random_state=run, algorithm='lloyd')
    y_kmeans_full = kmeans_full.fit_predict(X)

    gmm = GaussianMixture(n_components=2, covariance_type='full', n_init=1, random_state=run)
    y_em = gmm.fit_predict(X)

    def compute_accuracy(y_true, y_pred):
        cm = contingency_matrix(y_true, y_pred)
        row_ind, col_ind = linear_sum_assignment(-cm)
        accuracy = cm[row_ind, col_ind].sum() / cm.sum()
        return accuracy

    acc_iso = compute_accuracy(y_true, y_kmeans_iso)
    ari_iso = adjusted_rand_score(y_true, y_kmeans_iso)
    kmeans_iso_accuracies.append(acc_iso)
    kmeans_iso_ari.append(ari_iso)

    acc_full = compute_accuracy(y_true, y_kmeans_full)
    ari_full = adjusted_rand_score(y_true, y_kmeans_full)
    kmeans_full_accuracies.append(acc_full)
    kmeans_full_ari.append(ari_full)

    acc_em = compute_accuracy(y_true, y_em)
    ari_em = adjusted_rand_score(y_true, y_em)
    em_accuracies.append(acc_em)
    em_ari.append(ari_em)

    if run < 4:
        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 0], X[:, 1], c=y_kmeans_iso, cmap='viridis', s=10)
        plt.title(f'Run {run+1}: K-Means (Isotropic)')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()

        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 0], X[:, 1], c=y_kmeans_full, cmap='viridis', s=10)
        plt.title(f'Run {run+1}: K-Means (Full Covariance)')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()

        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 0], X[:, 1], c=y_em, cmap='viridis', s=10)
        plt.title(f'Run {run+1}: EM Clustering')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(kl_divergences, kmeans_iso_accuracies, color='red', label='K-Means Isotropic')
plt.scatter(kl_divergences, kmeans_full_accuracies, color='green', label='K-Means Full Covariance')
plt.scatter(kl_divergences, em_accuracies, color='blue', label='EM')
plt.title('Accuracy vs. KL Divergence')
plt.xlabel('KL Divergence D_KL(P || Q)')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(kl_divergences, kmeans_iso_ari, color='red', label='K-Means Isotropic')
plt.scatter(kl_divergences, kmeans_full_ari, color='green', label='K-Means Full Covariance')
plt.scatter(kl_divergences, em_ari, color='blue', label='EM')
plt.title('Adjusted Rand Index vs. KL Divergence')
plt.xlabel('KL Divergence D_KL(P || Q)')
plt.ylabel('Adjusted Rand Index')
plt.legend()
plt.show()

import pandas as pd

results_df = pd.DataFrame({
    'Run': np.arange(1, num_runs + 1),
    'KL Divergence': kl_divergences,
    'K-Means Iso Acc': kmeans_iso_accuracies,
    'K-Means Iso ARI': kmeans_iso_ari,
    'K-Means Full Acc': kmeans_full_accuracies,
    'K-Means Full ARI': kmeans_full_ari,
    'EM Acc': em_accuracies,
    'EM ARI': em_ari
})

print(results_df)
