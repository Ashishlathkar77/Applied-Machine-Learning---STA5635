import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

def load_image(image_path):
    image = Image.open(image_path).convert('L')
    image = np.array(image) / 255.0
    height, width = image.shape
    return image, width, height

def preprocess_data(image, width, height):

    x = np.arange(0, width)
    y = np.arange(0, height)
    xx, yy = np.meshgrid(x, y)
    coordinates = np.vstack([xx.ravel(), yy.ravel()]).T
    intensities = image.ravel()

    coordinates = (coordinates - np.mean(coordinates, axis=0)) / np.std(coordinates, axis=0)
    intensities = (intensities - np.mean(intensities)) / np.std(intensities)

    X = torch.tensor(coordinates, dtype=torch.float32).cuda()
    Y = torch.tensor(intensities, dtype=torch.float32).unsqueeze(1).cuda()

    return X, Y

image_path = '/content/horse033b.png'
image, width, height = load_image(image_path)
X, Y = preprocess_data(image, width, height)

class NeuralNet(nn.Module):
    def __init__(self, hidden_layers):
        super(NeuralNet, self).__init__()
        layers = []
        input_size = 2

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size

        layers.append(nn.Linear(input_size, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def train_nn(model, X, Y, epochs=300, batch_size=64, lr=0.003, lr_decay_epoch=100):

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    losses = []

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(X.size(0))
        epoch_loss = 0

        for i in range(0, X.size(0), batch_size):
            indices = permutation[i:i + batch_size]
            batch_x, batch_y = X[indices], Y[indices]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % lr_decay_epoch == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5

        avg_loss = epoch_loss / (X.size(0) // batch_size)
        losses.append(avg_loss)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}')

    return losses

def plot_results(losses, model, X, width, height):

    plt.figure()
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')
    plt.show()

    model.eval()
    with torch.no_grad():
        reconstructed = model(X).cpu().numpy().reshape(height, width)
        plt.imshow(reconstructed, cmap='gray')
        plt.title('Reconstructed Image')
        plt.show()

model_1layer = NeuralNet([128]).cuda()
losses_1layer = train_nn(model_1layer, X, Y)
plot_results(losses_1layer, model_1layer, X, width, height)

model_2layers = NeuralNet([32, 128]).cuda()
losses_2layers = train_nn(model_2layers, X, Y)
plot_results(losses_2layers, model_2layers, X, width, height)

model_3layers = NeuralNet([32, 64, 128]).cuda()
losses_3layers = train_nn(model_3layers, X, Y)
plot_results(losses_3layers, model_3layers, X, width, height)

model_4layers = NeuralNet([32, 64, 128, 128]).cuda()
losses_4layers = train_nn(model_4layers, X, Y)
plot_results(losses_4layers, model_4layers, X, width, height)
