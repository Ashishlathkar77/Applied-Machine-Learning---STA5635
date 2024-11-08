import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
from PIL import Image

# Step 1: Load and preprocess the image
image = Image.open('scene2.jpg')
image = image.resize((128, 82))  
image_array = np.array(image) / 255.0  

height, width, _ = image_array.shape

# Step 2: Construct the affinity matrix
def compute_affinity_matrix(image_array, sigma=0.1):

    pixels = image_array.reshape(-1, 3) 
    
    A = sp.lil_matrix((height * width, height * width))
    
    for i in range(height):
        for j in range(width):
            idx = i * width + j
            neighbors = []

            if i > 0:  
                neighbors.append((i - 1, j))
            if i < height - 1:  
                neighbors.append((i + 1, j))
            if j > 0:  
                neighbors.append((i, j - 1))
            if j < width - 1:  
                neighbors.append((i, j + 1))

            for ni, nj in neighbors:
                nidx = ni * width + nj
                diff = pixels[idx] - pixels[nidx]
                distance_sq = np.sum(diff**2)
                affinity = np.exp(-distance_sq / (2 * sigma**2))
                A[idx, nidx] = affinity
    
    return A

# Step 3: Perform spectral clustering
def spectral_clustering(A, n_clusters=15):

    A = A.tocsr()  
    degree_matrix = np.diag(np.array(A.sum(axis=1)).flatten()) 
    laplacian_matrix = degree_matrix - A  

    u, s, vt = svds(laplacian_matrix, k=n_clusters)

    clustering = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors').fit(u)
    
    return clustering.labels_

# Step 4: Visualize the clustering
def visualize_clusters(labels, height, width, n_clusters):
    label_image = labels.reshape((height, width))
    plt.imshow(label_image, cmap='tab20', interpolation='nearest')
    plt.title(f'Clustering with {n_clusters} clusters')
    plt.colorbar()
    plt.show()

# Step 5: Reconstruct the image using cluster centroids
def reconstruct_image(labels, image_array, n_clusters):

    reconstructed_image = np.zeros_like(image_array)
    for k in range(n_clusters):
        cluster_mask = (labels == k)
        cluster_pixels = image_array.reshape(-1, 3)[cluster_mask]
        mean_rgb = np.mean(cluster_pixels, axis=0)
        reconstructed_image.reshape(-1, 3)[cluster_mask] = mean_rgb
    
    return reconstructed_image

A = compute_affinity_matrix(image_array)
labels_15 = spectral_clustering(A, n_clusters=15)

# Step 6: Display clustering with 15 clusters
visualize_clusters(labels_15, height, width, n_clusters=15)

reconstructed_image_15 = reconstruct_image(labels_15, image_array, n_clusters=15)
plt.imshow(reconstructed_image_15)
plt.title('Reconstructed Image with 15 Clusters')
plt.show()

labels_25 = spectral_clustering(A, n_clusters=25)

# Step 7: Display clustering with 25 clusters
visualize_clusters(labels_25, height, width, n_clusters=25)

reconstructed_image_25 = reconstruct_image(labels_25, image_array, n_clusters=25)
plt.imshow(reconstructed_image_25)
plt.title('Reconstructed Image with 25 Clusters')
plt.show()
