import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import main
from training import SimpleCNN


# ----------------------------
# Paths
# -----------------------------
subset_save_path = r"C:\Users\user\Desktop\DSCI_Final_Project\subsets"
model_save_path = r"C:\Users\user\Desktop\DSCI_Final_Project\models"
plot_save_path = r"C:\Users\user\Desktop\DSCI_Final_Project\accuracyPlots"

os.makedirs(plot_save_path, exist_ok=True)

samples_per_class = 500
total_subset_size = 500

# -----------------------------
# Load test set
# -----------------------------
test_data = np.load(
    f"{subset_save_path}/{main.test_subset_path}.npz"
)

X_test = torch.tensor(test_data["X"], dtype=torch.float32).reshape(-1, 1, 28, 28)
y_test = torch.tensor(test_data["y"], dtype=torch.long)


train_data = np.load(
    f"{subset_save_path}/{main.train_subset_path}.npz"
)

unique, counts = np.unique(train_data["y"], return_counts=True)
total = len(train_data["y"])

print("Class distribution from train set:")
for cls, count in zip(unique, counts):
    print(f"Class {cls}: {count} samples ({count/total:.4f})")



# -----------------------------
# Subsets / models
# -----------------------------
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
    "train_full",
    "class_net_filter"
]


# -----------------------------
# Evaluation loop
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""
for name in subset_files:
    print(f"\nEvaluating {name}...")

    # Load model
    model = SimpleCNN().to(device)

    model_path = os.path.join(model_save_path, f"{name}{main.settingsText}.pt")
    
    if not os.path.exists(model_path):
        print(f"File not found: {model_path}")
        continue

    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Inference
    with torch.no_grad():
        outputs = model(X_test.to(device))
        preds = torch.argmax(outputs, dim=1).cpu()

    # Accuracy
    accuracy = (preds == y_test).float().mean().item()
    print(f"{name}... Accuracy: {accuracy:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test.numpy(), preds.numpy())

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"{name} Confusion Matrix")
"""

for name in subset_files:
    print(f"\nEvaluating {name}...")

    # Load model
    model = SimpleCNN().to(device)

    model_path = os.path.join(model_save_path, f"{name}{main.settingsText}.pt")
    
    if not os.path.exists(model_path):
        print(f"File not found: {model_path}")
        continue

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Inference
    with torch.no_grad():
        outputs = model(X_test.to(device))
        preds = torch.argmax(outputs, dim=1).cpu()

    # Accuracy
    accuracy = (preds == y_test).float().mean().item()
    print(f"{name}... Accuracy: {accuracy:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test.numpy(), preds.numpy())

    # ---- CONFUSION MATRIX PLOT ----
    plt.figure()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(values_format="d")
    plt.title(f"{name} Confusion Matrix")

    filename = f"{name}{main.settingsText}_confusion_matrix.png"
    plt.savefig(os.path.join(plot_save_path, filename), dpi=300)
    plt.close()


    # ---- PER-CLASS ACCURACY ----
    class_totals = cm.sum(axis=1)
    class_correct = cm.diagonal()

    # avoid divide-by-zero + ensure float division
    class_accuracy = class_correct / np.maximum(class_totals, 1).astype(float)


    # ---- BAR PLOT ----
    plt.figure()
    classes = np.arange(len(class_accuracy))

    plt.bar(classes, class_accuracy)
    plt.xlabel("Class")
    plt.ylabel("Accuracy")
    plt.title(f"{name} Accuracy Per Class")

    plt.xticks(classes)
    plt.ylim(0, 1)

    plt.tight_layout()

    filename = f"{name}{main.settingsText}_per_class_accuracy.png"
    plt.savefig(os.path.join(plot_save_path, filename), dpi=300)
    plt.close()
        