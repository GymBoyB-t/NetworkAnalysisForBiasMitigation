import torch
import torchvision
import torchvision.transforms as transforms

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=len(train_dataset),
    shuffle=False
)

images, labels = next(iter(train_loader))

images = images.numpy()
labels = labels.numpy()



samples_per_class = 500

selected_images = []
selected_labels = []

for digit in range(10):
    idx = np.where(labels == digit)[0][:samples_per_class]
    
    selected_images.append(images[idx])
    selected_labels.append(labels[idx])

selected_images = np.concatenate(selected_images)
selected_labels = np.concatenate(selected_labels)

print(selected_images.shape)  # (5000, 1, 28, 28)

X = selected_images.reshape(len(selected_images), -1)
print(X.shape)  # (5000, 784)

similarity_matrix = cosine_similarity(X)
print(similarity_matrix.shape)  # (5000, 5000)

G = nx.from_numpy_array(similarity_matrix)

for i, label in enumerate(selected_labels):
    G.nodes[i]["label"] = int(label)


print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

# example edge
u, v, w = list(G.edges(data=True))[0]
print("Edge example:", u, v, w)


A = nx.to_numpy_array(G)