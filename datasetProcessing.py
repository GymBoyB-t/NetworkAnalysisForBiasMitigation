import numpy as np
from sklearn.model_selection import train_test_split


def get_classSubset(images, labels, samples_per_class):
    selected_images = []
    selected_labels = []

    for digit in range(10):
        idx = np.where(labels == digit)[0][:samples_per_class]
        selected_images.append(images[idx])
        selected_labels.append(labels[idx])

    selected_images = np.concatenate(selected_images)
    selected_labels = np.concatenate(selected_labels)

    ############################################
    # 3. FLATTEN
    ############################################

    x = selected_images.reshape(len(selected_images), -1).astype("float32")
    y = selected_labels

    return x, y



def get_unbalancedClassSubset(images, labels, class_percentages):
    ############################################
    # FLATTEN
    ############################################
    x = images.reshape(len(images), -1).astype("float32")
    y = labels

    ############################################
    # STEP 1: FIXED STRATIFIED 80/20 SPLIT
    ############################################
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    ############################################
    # STEP 2: APPLY UNBALANCED SAMPLING ONLY ON TRAINING SET
    ############################################
    selected_images = []
    selected_labels = []

    total_train_samples = len(y_train_full)

    for digit in range(10):
        # Get indices for this class in training set
        idx = np.where(y_train_full == digit)[0]

        # Shuffle to avoid bias
        np.random.shuffle(idx)

        # Number of samples for this class
        n_samples = int(class_percentages[digit] * total_train_samples)

        # Avoid exceeding available samples
        n_samples = min(n_samples, len(idx))

        chosen_idx = idx[:n_samples]

        selected_images.append(X_train_full[chosen_idx])
        selected_labels.append(y_train_full[chosen_idx])

    X_train = np.concatenate(selected_images)
    y_train = np.concatenate(selected_labels)

    return X_train, X_test, y_train, y_test


