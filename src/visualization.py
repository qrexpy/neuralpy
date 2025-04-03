import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional
import networkx as nx

def plot_training_progress(losses: List[float], title: str = "Training Progress") -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                         title: str = "Confusion Matrix") -> None:
    cm = np.zeros((len(np.unique(y_true)), len(np.unique(y_true))))
    for i in range(len(y_true)):
        cm[int(y_true[i]), int(y_pred[i])] += 1

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def plot_network_architecture(layer_sizes: List[int],
                            title: str = "Neural Network Architecture") -> None:
    G = nx.Graph()

    pos = {}
    for layer_idx, size in enumerate(layer_sizes):
        for node_idx in range(size):
            node_id = f"L{layer_idx}N{node_idx}"
            G.add_node(node_id)
            pos[node_id] = (layer_idx, -node_idx)

    for layer_idx in range(len(layer_sizes) - 1):
        for node_idx in range(layer_sizes[layer_idx]):
            for next_node_idx in range(layer_sizes[layer_idx + 1]):
                G.add_edge(f"L{layer_idx}N{node_idx}",
                          f"L{layer_idx + 1}N{next_node_idx}")

    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=False, node_color='lightblue',
            node_size=500, width=0.1, alpha=0.6)
    plt.title(title)
    plt.show()

def plot_decision_boundary(model, X: np.ndarray, y: np.ndarray,
                          title: str = "Decision Boundary") -> None:
    if X.shape[1] != 2:
        raise ValueError("Decision boundary plotting only works for 2D input data")

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

def plot_weight_distribution(model, title: str = "Weight Distribution") -> None:
    n_layers = len(model.weights)
    fig, axes = plt.subplots(1, n_layers, figsize=(15, 5))

    for i, (weights, ax) in enumerate(zip(model.weights, axes)):
        sns.histplot(weights.flatten(), ax=ax, bins=30)
        ax.set_title(f'Layer {i+1}')
        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Count')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show() 