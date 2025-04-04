import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional
import networkx as nx
import os
import sys

# Use a simpler style for wider compatibility
plt.style.use('classic')

# Define a safe print function
def safe_print(message):
    """Print in a way that's safe for all terminals"""
    try:
        # Try to strip ANSI color codes if terminal doesn't support them
        if os.name == 'nt':  # Windows
            # Simple regex-free ANSI code stripper
            msg = ""
            i = 0
            while i < len(message):
                if message[i] == '\033' and i+1 < len(message) and message[i+1] == '[':
                    # Skip until 'm'
                    while i < len(message) and message[i] != 'm':
                        i += 1
                    i += 1  # Skip the 'm'
                else:
                    if i < len(message):
                        msg += message[i]
                    i += 1
            message = msg
        
        # Use plain print
        print(message)
    except Exception:
        # Super safe fallback
        print(str(message).encode('ascii', 'replace').decode('ascii'))
    sys.stdout.flush()

def plot_training_progress(losses: List[float], title: str = "Training Progress") -> None:
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100, facecolor='white')
    
    # Plot losses
    ax.plot(losses, 'b-', linewidth=2, label='Loss')
    
    # Add labels and title
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    if title:
        ax.set_title(title, fontsize=14)
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Simple box around the plot
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    
    fig.tight_layout()
    
    # Save the figure with a timestamp to avoid conflicts
    import time
    timestamp = int(time.time())
    filename = f"training_progress_{timestamp}.png"
    plt.savefig(filename)
    plt.close()
    
    safe_print(f"Training progress visualization saved as {filename}")
    return filename

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
    """Plot a neural network architecture"""
    # Create a figure with a simple style
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100, facecolor='white')
    
    # Number of layers
    n_layers = len(layer_sizes)
    
    # Maximum layer size for scaling
    max_neurons = max(layer_sizes)
    
    # Positions of layers
    layer_positions = np.linspace(0, n_layers - 1, n_layers)
    
    # Draw the layers
    for i, (n_neurons, pos) in enumerate(zip(layer_sizes, layer_positions)):
        # Calculate vertical positions of neurons in this layer
        neuron_positions = np.linspace(0, n_neurons - 1, n_neurons) - (n_neurons - 1) / 2
        
        # Scale neuron positions based on max layer size
        neuron_positions = neuron_positions * (max_neurons / (n_neurons + 1))
        
        # Draw neurons as simple circles
        for neuron_pos in neuron_positions:
            circle = plt.Circle((pos, neuron_pos), 0.2, 
                                fill=True, color='lightblue', 
                                edgecolor='blue', alpha=0.8)
            ax.add_patch(circle)
        
        # Draw connections to next layer
        if i < n_layers - 1:
            next_pos = layer_positions[i + 1]
            next_n_neurons = layer_sizes[i + 1]
            next_neuron_positions = np.linspace(0, next_n_neurons - 1, next_n_neurons) - (next_n_neurons - 1) / 2
            next_neuron_positions = next_neuron_positions * (max_neurons / (next_n_neurons + 1))
            
            # Draw connections
            for neuron_pos in neuron_positions:
                for next_neuron_pos in next_neuron_positions:
                    ax.plot([pos, next_pos], [neuron_pos, next_neuron_pos], 
                            'k-', alpha=0.1, linewidth=0.5)
        
        # Add layer labels
        ax.text(pos, max_neurons / 2 + 1, f"Layer {i+1}\n({n_neurons} neurons)",
                ha='center', va='center', color='black', fontsize=10)
    
    # Set axis limits
    ax.set_xlim(-0.5, n_layers - 0.5)
    margin = max_neurons / 2 + 2
    ax.set_ylim(-margin, margin)
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Simple box around the plot
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    
    # Add title
    if title:
        ax.set_title(title, fontsize=12, pad=20)
    
    fig.tight_layout()
    
    # Save the figure with a timestamp to avoid conflicts
    import time
    timestamp = int(time.time())
    filename = f"network_architecture_{timestamp}.png"
    plt.savefig(filename)
    plt.close()
    
    safe_print(f"Network architecture visualization saved as {filename}")
    return filename

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