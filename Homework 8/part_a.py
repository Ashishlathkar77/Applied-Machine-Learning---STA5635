import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment

sigma = 3
a_values = [0, 1, 2, 3, 4]
num_runs = 10
num_samples = 500

a_all = []
kmeans_accuracies_all = []
em_accuracies_all = []
kmeans_ari_all = []
em_ari_all = []

for a in a_values:
   
    X_Q = np.random.normal(loc=0, scale=sigma, size=(num_samples, 2))
    y_Q = np.zeros(num_samples, dtype=int)  
 
    mu = np.array([a, 0])
    X_a = np.random.normal(loc=mu, scale=1.0, size=(num_samples, 2))
    y_a = np.ones(num_samples, dtype=int)  

    X = np.vstack((X_Q, X_a))
    y_true = np.hstack((y_Q, y_a))

    if a == 0:
        X_plot = X.copy()
        y_true_plot = y_true.copy()

    for run in range(num_runs):

        random_state = run

        kmeans = KMeans(n_clusters=2, n_init=1, random_state=random_state)
        y_kmeans = kmeans.fit_predict(X)

        gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=random_state)
        y_em = gmm.fit_predict(X)

        def compute_accuracy(y_true, y_pred):
            cm = contingency_matrix(y_true, y_pred)
            row_ind, col_ind = linear_sum_assignment(-cm)
            accuracy = cm[row_ind, col_ind].sum() / cm.sum()
            return accuracy

        kmeans_accuracy = compute_accuracy(y_true, y_kmeans)
        kmeans_ari = adjusted_rand_score(y_true, y_kmeans)
        kmeans_accuracies_all.append(kmeans_accuracy)
        kmeans_ari_all.append(kmeans_ari)

        em_accuracy = compute_accuracy(y_true, y_em)
        em_ari = adjusted_rand_score(y_true, y_em)
        em_accuracies_all.append(em_accuracy)
        em_ari_all.append(em_ari)

        a_all.append(a)

        if a == 0 and run == 0:
            y_kmeans_plot = y_kmeans.copy()
            y_em_plot = y_em.copy()

plt.figure(figsize=(8, 6))
plt.scatter(X_plot[:, 0], X_plot[:, 1], c=y_kmeans_plot, cmap='viridis', s=10)
plt.title('K-Means Clustering Result for a=0')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(X_plot[:, 0], X_plot[:, 1], c=y_em_plot, cmap='viridis', s=10)
plt.title('EM Clustering Result for a=0')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(a_all, kmeans_accuracies_all, color='red', label='K-Means')
plt.scatter(a_all, em_accuracies_all, color='black', label='EM')
plt.title('Accuracy vs. a')
plt.xlabel('a')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(a_all, kmeans_ari_all, color='red', label='K-Means')
plt.scatter(a_all, em_ari_all, color='black', label='EM')
plt.title('Adjusted Rand Index vs. a')
plt.xlabel('a')
plt.ylabel('Adjusted Rand Index')
plt.legend()
plt.show()
