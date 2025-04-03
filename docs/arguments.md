# neural.py Arguments Documentation

This document provides detailed information about the arguments used in the neural.py implementation.

## NeuralNetwork Class Arguments

### Initialization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `layer_sizes` | List[int] | Required | List of integers representing the number of neurons in each layer |
| `learning_rate` | float | 0.01 | Learning rate for gradient descent |
| `use_relu` | bool | False | Whether to use ReLU activation function instead of sigmoid |
| `use_leaky_relu` | bool | False | Whether to use Leaky ReLU activation function instead of sigmoid |
| `leaky_relu_alpha` | float | 0.01 | Slope for negative values in Leaky ReLU |

### Method Parameters

#### train Method

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X` | np.ndarray | Required | Input data of shape (n_samples, n_features) |
| `y` | np.ndarray | Required | Target data of shape (n_samples, n_outputs) |
| `epochs` | int | Required | Number of training epochs |
| `batch_size` | Optional[int] | None | Size of mini-batches (if None, uses full batch) |

#### predict Method

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X` | np.ndarray | Required | Input data of shape (n_samples, n_features) |

## Visualization Functions

### plot_training_progress

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `losses` | List[float] | Required | List of loss values for each epoch |
| `title` | str | "Training Progress" | Title for the plot |

### plot_confusion_matrix

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `y_true` | np.ndarray | Required | True labels |
| `y_pred` | np.ndarray | Required | Predicted labels |
| `title` | str | "Confusion Matrix" | Title for the plot |

### plot_network_architecture

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `layer_sizes` | List[int] | Required | List of integers representing the number of neurons in each layer |
| `title` | str | "Neural Network Architecture" | Title for the plot |

### plot_decision_boundary

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | NeuralNetwork | Required | Trained neural network model |
| `X` | np.ndarray | Required | Input data |
| `y` | np.ndarray | Required | Target data |
| `title` | str | "Decision Boundary" | Title for the plot |

### plot_weight_distribution

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | NeuralNetwork | Required | Trained neural network model |
| `title` | str | "Weight Distribution" | Title for the plot | 