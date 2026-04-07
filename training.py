import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import main

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(32 * 7 * 7, 128)  # MNIST: 28x28 → 7x7 after pooling
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 28 → 14
        x = self.pool(F.relu(self.conv2(x)))  # 14 → 7
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

def train_model(X, y, epochs=25, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = torch.tensor(X, dtype=torch.float32).reshape(-1, 1, 28, 28)
    y = torch.tensor(y, dtype=torch.long)

    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SimpleCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    return model


subset_save_path = r"C:\Users\user\Desktop\DSCI_Final_Project\subsets"
model_save_path = r"C:\Users\user\Desktop\DSCI_Final_Project\models"

os.makedirs(model_save_path, exist_ok=True)

total_subset_size = 500
samples_per_class = 500

subset_files = [
    "random_subset",
    "deg_high",
    "deg_low",
    "clust_high",
    "clust_low",
    "pathlen_high",
    "pathlen_low",
    "diam_high",
    "diam_low",
    "density_high",
    "density_low",
    "class_net_filter"
]

if __name__ == "__main__":
    for name in subset_files:
        print(f"\nTraining on {name}...")
        
        path = f"{subset_save_path}/{name}{main.settingsText}.npz"

        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue


        data = np.load(path)

        X, y = data["X"], data["y"]

        model = train_model(X, y, epochs=25)

        model_path = os.path.join(model_save_path, f"{name}{main.settingsText}.pt")
        torch.save(model.state_dict(), model_path)

        print(f"Saved model: {model_path}")


    train_data = np.load(f"{subset_save_path}/{main.train_subset_path}.npz")
    X_train, y_train = train_data["X"], train_data["y"]

    model = train_model(X_train, y_train, epochs=25)
    torch.save(model.state_dict(), os.path.join(model_save_path, f"train_full{main.settingsText}.pt"))