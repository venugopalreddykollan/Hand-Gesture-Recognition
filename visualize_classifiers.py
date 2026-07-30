"""Visualization of Random Forest classifier on hand gesture recognition data."""

from pathlib import Path
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data():
    """Load landmarks and labels from saved numpy files."""
    script_dir = Path(__file__).resolve().parent
    workspace_root = script_dir.parent.parent

    candidate_dirs = [
        script_dir / "landmarks",
        script_dir.parent / "landmarks",
        workspace_root / "OPENCV" / "hand_gesture_recognition" / "landmarks",
        workspace_root / "hand_gesture_recognition" / "landmarks",
    ]

    landmarks_dir = next(
        (candidate for candidate in candidate_dirs if candidate.exists()),
        None,
    )

    if landmarks_dir is None:
        searched_paths = "\n".join(f"  - {path}" for path in candidate_dirs)
        raise FileNotFoundError(
            "Could not find the landmarks directory. Searched:\n" + searched_paths
        )

    landmarks_file = landmarks_dir / "landmarks.npy"
    labels_file = landmarks_dir / "labels.npy"
    class_labels_file = landmarks_dir / "class_labels.pkl"

    missing_files = [
        path for path in (landmarks_file, labels_file, class_labels_file)
        if not path.exists()
    ]
    if missing_files:
        missing_paths = "\n".join(f"  - {path}" for path in missing_files)
        raise FileNotFoundError(
            "Missing required data files:\n" + missing_paths
        )

    print("Loading extracted landmarks...\n")

    X = np.load(landmarks_file)
    y = np.load(labels_file)

    with open(class_labels_file, "rb") as f:
        class_labels = pickle.load(f)

    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Classes: {class_labels}\n")

    return X, y, class_labels


def prepare_data(X, y):
    """Split, scale, and apply PCA to data."""
    print("Splitting data into training (80%) and testing (20%)...\n")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Normalizing features...\n")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Applying PCA to reduce dimensions for visualization...\n")
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.2%}\n")

    return X_train_pca, X_test_pca, y_train, y_test


def visualize_classifier(X_train_pca, X_test_pca, y_train, y_test, class_labels):
    """Train and visualize Random Forest classifier decision boundaries."""
    print("Training Random Forest Classifier...")

    clf = RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=42, n_jobs=-1
    )
    clf.fit(X_train_pca, y_train)

    score = clf.score(X_test_pca, y_test)
    print(f"Accuracy: {score:.2%}\n")

    # Define colors for each gesture class
    colors = plt.cm.Set3(np.linspace(0, 1, len(class_labels)))

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create mesh for decision boundary
    h = 0.02  # step size in mesh
    x_min = X_train_pca[:, 0].min() - 1
    x_max = X_train_pca[:, 0].max() + 1
    y_min = X_train_pca[:, 1].min() - 1
    y_max = X_train_pca[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )

    # Predict on mesh
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Set3)
    ax.contour(
        xx, yy, Z, colors='black', linewidths=0.5,
        levels=len(class_labels) - 1
    )

    # Plot training points
    for class_idx, (class_label, color) in enumerate(
        zip(class_labels, colors)
    ):
        mask = y_train == class_idx
        ax.scatter(
            X_train_pca[mask, 0],
            X_train_pca[mask, 1],
            c=[color],
            label=class_label,
            edgecolors='k',
            s=50,
            alpha=0.7,
        )

    # Plot test points with different marker
    for class_idx, (class_label, color) in enumerate(
        zip(class_labels, colors)
    ):
        mask = y_test == class_idx
        ax.scatter(
            X_test_pca[mask, 0],
            X_test_pca[mask, 1],
            c=[color],
            marker='^',
            edgecolors='k',
            s=80,
            alpha=0.8,
        )

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_title(
        f"Random Forest Classifier\nAccuracy: {score:.2%}",
        fontsize=14,
        fontweight='bold'
    )
    ax.set_xlabel("PCA Component 1", fontsize=12)
    ax.set_ylabel("PCA Component 2", fontsize=12)
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.08),
        ncol=len(class_labels),
        fontsize=10
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_dir = Path(__file__).resolve().parent / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "classifier_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to {output_file}")

    plt.show()


def main():
    """Main function to load data and create visualization."""
    X, y, class_labels = load_data()
    X_train_pca, X_test_pca, y_train, y_test = prepare_data(X, y)
    visualize_classifier(X_train_pca, X_test_pca, y_train, y_test, class_labels)


if __name__ == "__main__":
    main()
