import umap.umap_ as umap
import matplotlib.pyplot as plt
import numpy as np
import main
import os


def plot_subset(X_umap, y, subset_idx, title, save_path):
    plt.figure(figsize=(8, 6))

    mask = np.zeros(len(y), dtype=bool)
    mask[subset_idx] = True  # subset indices directly map

    # Background (full dataset)
    plt.scatter(X_umap[~mask, 0], X_umap[~mask, 1],
                c='red', s=5, alpha=1)

    # Highlight subset
    scatter = plt.scatter(X_umap[mask, 0], X_umap[mask, 1],
                          c=y[mask], s=8)

    plt.title(title)
    plt.legend(*scatter.legend_elements(), title="Classes")

    filename = title.replace(" ", "_").replace("(", "").replace(")", "") + ".png"
    plt.savefig(os.path.join(save_path, filename), dpi=300)
    plt.close()


subset_save_path = r"C:\Users\user\Desktop\DSCI_Final_Project\subsets"
image_save_path = r"C:\Users\user\Desktop\DSCI_Final_Project\TestImages"
total_subset_size = 500
samples_per_class = 500

# Load training set
print("Loading training data...")
train_data = np.load(f"{subset_save_path}/{main.train_subset_path}.npz")
X_train, y_train, idx_train = train_data["X"], train_data["y"], train_data["indices"]

print("Fitting UMAP...")
# Fit UMAP on training set
umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', random_state=0)
X_train_umap = umap_model.fit_transform(X_train)

def load_and_transform(path, umap_model):
    d = np.load(path)
    X_umap = umap_model.transform(d["X"])
    return X_umap, d["y"], d["indices"]


print("Loading test data...")
# Load test set
test_data = np.load(f"{subset_save_path}\{main.test_subset_path}.npz")
X_test, y_test, idx_test = test_data["X"], test_data["y"], test_data["indices"]

# List of subset files to plot
subset_files = [
    ("random_subset", "Random Subset"),
    ("deg_high", "High Degree"),
    ("deg_low", "Low Degree"),
    ("clust_high", "High Clustering"),
    ("clust_low", "Low Clustering"),
    ("pathlen_high", "High Path Length"),
    ("pathlen_low", "Low Path Length"),
    ("diam_high", "High Diameter"),
    ("diam_low", "Low Diameter"),
    ("density_high", "High Density"),
    ("density_low", "Low Density"),
    ("class_net_filter", "Class Net Filter")
]

print("Plotting Subsets...")
# Loop through and plot all subsets
for file_prefix, title in subset_files:
    path = f"{subset_save_path}\{file_prefix}{main.settingsText}.npz"

    if not os.path.exists(path):
        print(f"File not found: {path}")
        continue

    _, _, subset_idx = load_and_transform(path, umap_model)

    plot_subset(
        X_train_umap,     # FULL dataset
        y_train,          # FULL labels
        subset_idx,       # subset indices
        title=title,
        save_path=image_save_path
    )


# Plot full training set
plot_subset(
    X_train_umap,
    y_train,
    idx_train,
    title="Train Full",
    save_path=image_save_path
)
