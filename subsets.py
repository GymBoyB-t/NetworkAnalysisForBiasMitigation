import os
import numpy as np
import networkx as nx


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def save_subset(name, idx, X_sub, y_sub, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    filepath = os.path.join(save_dir, f"{name}.npz")

    np.savez_compressed(
        filepath,
        indices=idx,
        X=X_sub,
        y=y_sub
    )

    print(f"Saved: {filepath}")


def class_network_filter_subset(
    X, y, G, size, samples_per_class, save_dir, settings, remove_frac=0.2
):
    import numpy as np

    keep_mask = np.ones(len(y), dtype=bool)

    classes = np.unique(y)

    # Precompute class centers
    class_centers = {}
    for c in classes:
        idx = np.where(y == c)[0]
        if len(idx) > 0:
            class_centers[c] = np.mean(X[idx], axis=0)

    for c in classes:
        class_idx = np.where((y == c) & keep_mask)[0]
        class_size = len(class_idx)

        if class_size == 0:
            continue

        # Number to remove (fixed fraction)
        n_remove = int(remove_frac * class_size)

        if n_remove == 0:
            continue

        # Collect centers of other classes
        other_centers = np.array(
            [class_centers[oc] for oc in classes if oc != c]
        )

        if len(other_centers) == 0:
            continue

        # Compute distance to nearest other-class center
        distances = []
        for idx_i in class_idx:
            dists = np.linalg.norm(other_centers - X[idx_i], axis=1)
            min_dist = np.min(dists)
            distances.append(min_dist)

        distances = np.array(distances)

        # Get indices of closest points (to remove)
        remove_local_idx = np.argsort(distances)[:n_remove]
        remove_idx = class_idx[remove_local_idx]

        keep_mask[remove_idx] = False

    # Final subset
    idx = np.where(keep_mask)[0]

    # Enforce global size constraint
    if len(idx) > size:
        idx = idx[:size]

    save_subset(f"class_net_filter{settings}", idx, X[idx], y[idx], save_dir)

    return idx, X[idx], y[idx]

def random_subset(X, y, size, samples_per_class, save_dir, settings):
    idx = np.random.choice(len(X), size, replace=False)
    assert len(idx) == size
    save_subset(f"random_subset{settings}", idx, X[idx], y[idx], save_dir)
    return idx, X[idx], y[idx]

def high_degree_subset(X, y, G, size, samples_per_class, save_dir, settings):
    scores = nx.degree_centrality(G)
    idx = np.array(sorted(scores, key=scores.get, reverse=True)[:size])
    assert len(idx) == size
    save_subset(f"deg_high{settings}", idx, X[idx], y[idx], save_dir)
    return idx, X[idx], y[idx]

def low_degree_subset(X, y, G, size, samples_per_class, save_dir, settings):
    scores = nx.degree_centrality(G)
    idx = np.array(sorted(scores, key=scores.get)[:size])
    assert len(idx) == size
    save_subset(f"deg_low{settings}", idx, X[idx], y[idx], save_dir)
    return idx, X[idx], y[idx]

def high_clustering_subset(X, y, G, size, samples_per_class, save_dir, settings):
    scores = nx.clustering(G)
    idx = np.array(sorted(scores, key=scores.get, reverse=True)[:size])
    assert len(idx) == size
    save_subset(f"clust_high{settings}", idx, X[idx], y[idx], save_dir)
    return idx, X[idx], y[idx]

def low_clustering_subset(X, y, G, size, samples_per_class, save_dir, settings):
    scores = nx.clustering(G)
    idx = np.array(sorted(scores, key=scores.get)[:size])
    assert len(idx) == size
    save_subset(f"clust_low{settings}", idx, X[idx], y[idx], save_dir)
    return idx, X[idx], y[idx]

def high_pathlength_subset(X, y, G, size, samples_per_class, save_dir, settings):
    scores = nx.closeness_centrality(G)
    idx = np.array(sorted(scores, key=scores.get, reverse=True)[:size])
    assert len(idx) == size
    save_subset(f"pathlen_high{settings}", idx, X[idx], y[idx], save_dir)
    return idx, X[idx], y[idx]

def low_pathlength_subset(X, y, G, size, samples_per_class, save_dir, settings):
    scores = nx.closeness_centrality(G)
    idx = np.array(sorted(scores, key=scores.get)[:size])
    assert len(idx) == size
    save_subset(f"pathlen_low{settings}", idx, X[idx], y[idx], save_dir)
    return idx, X[idx], y[idx]

def high_diameter_subset(X, y, G, size, samples_per_class, save_dir, settings):
    ecc = nx.eccentricity(G)
    idx = np.array(sorted(ecc, key=ecc.get, reverse=True)[:size])
    assert len(idx) == size
    save_subset(f"diam_high{settings}", idx, X[idx], y[idx], save_dir)
    return idx, X[idx], y[idx]

def low_diameter_subset(X, y, G, size, samples_per_class, save_dir, settings):
    ecc = nx.eccentricity(G)
    idx = np.array(sorted(ecc, key=ecc.get)[:size])
    assert len(idx) == size
    save_subset(f"diam_low{settings}", idx, X[idx], y[idx], save_dir)
    return idx, X[idx], y[idx]

def high_density_subset(X, y, G, size, samples_per_class, save_dir, settings):
    deg = dict(G.degree())
    idx = np.array(sorted(deg, key=deg.get, reverse=True)[:size])
    assert len(idx) == size
    save_subset(f"density_high{settings}", idx, X[idx], y[idx], save_dir)
    return idx, X[idx], y[idx]

def low_density_subset(X, y, G, size, samples_per_class, save_dir, settings):
    deg = dict(G.degree())
    idx = np.array(sorted(deg, key=deg.get)[:size])
    assert len(idx) == size
    save_subset(f"density_low{settings}", idx, X[idx], y[idx], save_dir)
    return idx, X[idx], y[idx]

