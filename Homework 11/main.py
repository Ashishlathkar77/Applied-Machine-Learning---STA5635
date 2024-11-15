Part a)
import os
import numpy as np
import matplotlib.pyplot as plt
from zipfile import ZipFile
from PIL import Image

def load_and_preprocess_images(zip_path, img_size=(128, 128)):
    images = []
   
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("horse_images")
   
    for file_name in os.listdir("horse_images"):
        if file_name.endswith(".png"):
            img_path = os.path.join("horse_images", file_name)
            img = Image.open(img_path).convert("L").resize(img_size)
            img_array = np.array(img) / 255.0
            images.append(img_array.flatten())
   
    return np.array(images)

zip_path = "horses.zip"
horse_images = load_and_preprocess_images(zip_path)

U, S, Vt = np.linalg.svd(horse_images, full_matrices=False)

singular_values = S[2:]

plt.figure(figsize=(10, 6))
plt.plot(singular_values, marker='o')
plt.title("Singular Values (After Discarding the Two Largest)")
plt.xlabel("Index")
plt.ylabel("Singular Value")
plt.yscale("log")
plt.show()

Part b)
Vt_2D = Vt[:2, :]  # First two principal components

projected_2D = horse_images @ Vt_2D.T  # Project onto 2D space

plt.figure(figsize=(10, 8))
plt.scatter(projected_2D[:, 0], projected_2D[:, 1], alpha=0.6, color='blue', edgecolor='k')
plt.title("2D Projection of Horse Images onto First Two Principal Components")
plt.xlabel("1st Principal Component")
plt.ylabel("2nd Principal Component")
plt.grid(True)
plt.show()

Part c)
mean_horse = np.mean(horse_images, axis=0)
horse_images_centered = horse_images - mean_horse

projected_2D_horses = horse_images_centered @ Vt_2D.T

bird_image = Image.open("bird.png").convert("L").resize((128, 128))
bird_array = np.array(bird_image) / 255.0
bird_array_flat = bird_array.flatten()

bird_centered = bird_array_flat - mean_horse

projected_2D_bird = bird_centered @ Vt_2D.T

plt.figure(figsize=(10, 8))
plt.scatter(projected_2D_horses[:, 0], projected_2D_horses[:, 1], color='black', label="Horses", alpha=0.6, edgecolor='k')
plt.scatter(projected_2D_bird[0], projected_2D_bird[1], color='red', marker='x', s=100, label="Bird")
plt.title("2D Projection of Horse Images and Bird Image on First Two Principal Components")
plt.xlabel("1st Principal Component")
plt.ylabel("2nd Principal Component")
plt.legend()
plt.grid(True)
plt.show()

Part d)
horse_image_path = "horse_images/horse070.png"
horse_img = Image.open(horse_image_path).convert("L").resize((128, 128))
horse_img_array = np.array(horse_img) / 255.0
horse_img_flat = horse_img_array.flatten()

horse_img_centered = horse_img_flat - mean_horse

Vt_32 = Vt[:32, :]
projection_32 = horse_img_centered @ Vt_32.T
reconstruction_32 = projection_32 @ Vt_32
reconstructed_img = reconstruction_32 + mean_horse

reconstructed_img_reshaped = reconstructed_img.reshape(128, 128)
binary_reconstruction = (reconstructed_img_reshaped > 0.5).astype(float)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(horse_img_array, cmap='gray')
plt.title("Original Horse Image (horse070.png)")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(binary_reconstruction, cmap='gray')
plt.title("Binary Reconstruction (32 PCs, Threshold 0.5)")
plt.axis('off')

plt.show()

Part e)
bird_image = Image.open("bird.png").convert("L").resize((128, 128))
bird_array = np.array(bird_image) / 255.0
bird_flat = bird_array.flatten()

bird_centered = bird_flat - mean_horse

projection_32_bird = bird_centered @ Vt_32.T
reconstruction_32_bird = projection_32_bird @ Vt_32
reconstructed_bird = reconstruction_32_bird + mean_horse

reconstructed_bird_reshaped = reconstructed_bird.reshape(128, 128)
binary_reconstruction_bird = (reconstructed_bird_reshaped > 0.5).astype(float)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(bird_array, cmap='gray')
plt.title("Original Bird Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(binary_reconstruction_bird, cmap='gray')
plt.title("Binary Reconstruction of Bird (32 PCs, Threshold 0.5)")
plt.axis('off')

plt.show()

Part f)
distances_horses = []
second_pc_coords_horses = []

for horse in horse_images_centered:
    projection_32 = horse @ Vt_32.T
    reconstruction_32 = projection_32 @ Vt_32
    residual = horse - reconstruction_32
    distance = np.linalg.norm(residual)
    distances_horses.append(distance)
    second_pc_coord = projection_32[1]
    second_pc_coords_horses.append(second_pc_coord)

projection_32_bird = bird_centered @ Vt_32.T
reconstruction_32_bird = projection_32_bird @ Vt_32
residual_bird = bird_centered - reconstruction_32_bird
distance_bird = np.linalg.norm(residual_bird)
second_pc_coord_bird = projection_32_bird[1]

plt.figure(figsize=(10, 6))

plt.scatter(second_pc_coords_horses, distances_horses, color='black', label="Horses", alpha=0.6, edgecolor='k')
plt.scatter(second_pc_coord_bird, distance_bird, color='red', marker='x', s=100, label="Bird")

plt.title("Distance to 32-PC Plane vs. Second PC Coordinate")
plt.xlabel("Second Principal Component Coordinate")
plt.ylabel("Distance to 32-PC Plane")
plt.legend()
plt.grid(True)
plt.show()

Part g)
plt.figure(figsize=(10, 6))
plt.hist(distances_horses, bins=20, color='blue', edgecolor='black', alpha=0.7)
plt.title("Histogram of Distances to 32-PC Plane for Horse Images")
plt.xlabel("Distance to 32-PC Plane")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
