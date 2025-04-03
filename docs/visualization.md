# Visualization Tools in neural.py

This document provides detailed information about the visualization tools available in neural.py.

## Available Visualization Functions

### plot_training_progress

Plots the training loss over epochs to visualize the learning process.

**Function:**
```python
plot_training_progress(losses, title="Training Progress")
```

**Parameters:**
- `losses`: List of loss values for each epoch
- `title`: Title for the plot (default: "Training Progress")

**Example:**
```python
losses = nn.train(X, y, epochs=1000)
plot_training_progress(losses, "XOR Training Progress")
```

### plot_confusion_matrix

Plots a confusion matrix for classification tasks.

**Function:**
```python
plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix")
```

**Parameters:**
- `y_true`: True labels
- `y_pred`: Predicted labels
- `title`: Title for the plot (default: "Confusion Matrix")

**Example:**
```python
predictions = nn.predict(X)
y_pred = np.argmax(predictions, axis=1)
y_true = np.argmax(y, axis=1)
plot_confusion_matrix(y_true, y_pred, "Digit Recognition Confusion Matrix")
```

### plot_network_architecture

Visualizes the architecture of the neural network.

**Function:**
```python
plot_network_architecture(layer_sizes, title="Neural Network Architecture")
```

**Parameters:**
- `layer_sizes`: List of integers representing the number of neurons in each layer
- `title`: Title for the plot (default: "Neural Network Architecture")

**Example:**
```python
plot_network_architecture([2, 4, 1], "XOR Network Architecture")
```

### plot_decision_boundary

Plots the decision boundary for 2D classification problems.

**Function:**
```python
plot_decision_boundary(model, X, y, title="Decision Boundary")
```

**Parameters:**
- `model`: Trained neural network model
- `X`: Input data
- `y`: Target data
- `title`: Title for the plot (default: "Decision Boundary")

**Example:**
```python
plot_decision_boundary(nn, X, y.ravel(), "XOR Decision Boundary")
```

### plot_weight_distribution

Plots the distribution of weights in the neural network.

**Function:**
```python
plot_weight_distribution(model, title="Weight Distribution")
```

**Parameters:**
- `model`: Trained neural network model
- `title`: Title for the plot (default: "Weight Distribution")

**Example:**
```python
plot_weight_distribution(nn, "XOR Network Weight Distribution")
```

## Dependencies

The visualization tools use the following libraries:
- matplotlib
- seaborn
- networkx (for network architecture visualization)

These dependencies are included in the requirements.txt file. 