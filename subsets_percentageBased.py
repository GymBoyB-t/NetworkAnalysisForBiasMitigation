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
    X, y, G, size, min_threshold, save_dir, settings, remove_frac=0.2
):
    import numpy as np

    keep_mask = np.ones(len(y), dtype=bool)
    classes = np.unique(y)

    # Precompute class centers
    class_centers = {
        c: np.mean(X[y == c], axis=0)
        for c in classes if np.any(y == c)
    }

    for c in classes:
        class_idx = np.where((y == c) & keep_mask)[0]
        class_size = len(class_idx)

        if class_size <= min_threshold:
            continue

        target_size = max(min_threshold, int((1 - remove_frac) * class_size))
        n_remove = class_size - target_size

        other_centers = np.array(
            [class_centers[oc] for oc in classes if oc != c]
        )

        if len(other_centers) == 0:
            continue

        distances = []
        for i in class_idx:
            d = np.min(np.linalg.norm(other_centers - X[i], axis=1))
            distances.append(d)

        distances = np.array(distances)

        remove_idx = class_idx[np.argsort(distances)[:n_remove]]
        keep_mask[remove_idx] = False

    idx = np.where(keep_mask)[0]


    save_subset(f"class_net_filter{settings}", idx, X[idx], y[idx], save_dir)
    return idx, X[idx], y[idx]



def _per_class_score_subset(y, scores, min_threshold, remove_frac):

    keep_idx = []
    classes = np.unique(y)

    for c in classes:
        class_idx = np.where(y == c)[0]
        class_scores = np.array([scores[i] for i in class_idx])
        class_size = len(class_idx)

        if class_size <= min_threshold:
            keep_idx.extend(class_idx)
            continue

        target_size = max(min_threshold, int((1 - remove_frac) * class_size))

        order = np.argsort(-class_scores)  # high → low
        selected = class_idx[order[:target_size]]

        keep_idx.extend(selected)

    return np.array(keep_idx)


def random_subset(X, y, size, min_threshold, save_dir, settings, remove_frac=0.2):
    # assign random score to each node
    scores = {i: np.random.rand() for i in range(len(X))}

    idx = _per_class_score_subset(y, scores, min_threshold, remove_frac)

    save_subset(f"random_subset{settings}", idx, X[idx], y[idx], save_dir)
    return idx, X[idx], y[idx]

def high_degree_subset(X, y, G, size, min_threshold, save_dir, settings, remove_frac=0.2):
    scores = nx.degree_centrality(G)
    idx = _per_class_score_subset(y, scores, min_threshold, remove_frac)

    save_subset(f"deg_high{settings}", idx, X[idx], y[idx], save_dir)
    return idx, X[idx], y[idx]


def low_degree_subset(X, y, G, size, min_threshold, save_dir, settings, remove_frac=0.2):
    scores = nx.degree_centrality(G)
    scores = {k: -v for k, v in scores.items()}

    idx = _per_class_score_subset(y, scores, min_threshold, remove_frac)

    save_subset(f"deg_low{settings}", idx, X[idx], y[idx], save_dir)
    return idx, X[idx], y[idx]


def high_clustering_subset(X, y, G, size, min_threshold, save_dir, settings, remove_frac=0.2):
    scores = nx.clustering(G)
    idx = _per_class_score_subset(y, scores, min_threshold, remove_frac)

    save_subset(f"clust_high{settings}", idx, X[idx], y[idx], save_dir)
    return idx, X[idx], y[idx]


def low_clustering_subset(X, y, G, size, min_threshold, save_dir, settings, remove_frac=0.2):
    scores = nx.clustering(G)
    scores = {k: -v for k, v in scores.items()}

    idx = _per_class_score_subset(y, scores, min_threshold, remove_frac)

    save_subset(f"clust_low{settings}", idx, X[idx], y[idx], save_dir)
    return idx, X[idx], y[idx]


def high_pathlength_subset(X, y, G, size, min_threshold, save_dir, settings, remove_frac=0.2):
    scores = nx.closeness_centrality(G)
    idx = _per_class_score_subset(y, scores, min_threshold, remove_frac)

    save_subset(f"pathlen_high{settings}", idx, X[idx], y[idx], save_dir)
    return idx, X[idx], y[idx]


def low_pathlength_subset(X, y, G, size, min_threshold, save_dir, settings, remove_frac=0.2):
    scores = nx.closeness_centrality(G)
    scores = {k: -v for k, v in scores.items()}

    idx = _per_class_score_subset(y, scores, min_threshold, remove_frac)

    save_subset(f"pathlen_low{settings}", idx, X[idx], y[idx], save_dir)
    return idx, X[idx], y[idx]


def high_diameter_subset(X, y, G, size, min_threshold, save_dir, settings, remove_frac=0.2):
    scores = nx.eccentricity(G)
    idx = _per_class_score_subset(y, scores, min_threshold, remove_frac)

    save_subset(f"diam_high{settings}", idx, X[idx], y[idx], save_dir)
    return idx, X[idx], y[idx]


def low_diameter_subset(X, y, G, size, min_threshold, save_dir, settings, remove_frac=0.2):
    scores = nx.eccentricity(G)
    scores = {k: -v for k, v in scores.items()}

    idx = _per_class_score_subset(y, scores, min_threshold, remove_frac)

    save_subset(f"diam_low{settings}", idx, X[idx], y[idx], save_dir)
    return idx, X[idx], y[idx]


def high_density_subset(X, y, G, size, min_threshold, save_dir, settings, remove_frac=0.2):
    scores = dict(G.degree())
    idx = _per_class_score_subset(y, scores, min_threshold, remove_frac)

    save_subset(f"density_high{settings}", idx, X[idx], y[idx], save_dir)
    return idx, X[idx], y[idx]


def low_density_subset(X, y, G, size, min_threshold, save_dir, settings, remove_frac=0.2):
    scores = dict(G.degree())
    scores = {k: -v for k, v in scores.items()}

    idx = _per_class_score_subset(y, scores, min_threshold, remove_frac)

    save_subset(f"density_low{settings}", idx, X[idx], y[idx], save_dir)
    return idx, X[idx], y[idx]