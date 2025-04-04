# Control Panel: Your Neural Network's Parameters

This guide provides a friendly overview of all the knobs and dials you can adjust in neural.py to make your neural network perform exactly how you want it.

## Creating Your Neural Network

When you're first setting up your neural network, you have several options to customize its behavior:

| Parameter | Type | Default | What it does |
|-----------|------|---------|-------------|
| `layer_sizes` | List[int] | Required | The architecture of your network: how many neurons in each layer. For example, `[2, 4, 1]` creates a network with 2 input neurons, 4 hidden neurons, and 1 output neuron. |
| `learning_rate` | float | 0.01 | How quickly your network adapts to new information. Higher values mean faster learning but potentially less stability. Think of it like the size of steps your network takes during learning. |
| `use_relu` | bool | False | Whether to use ReLU activation function ($$f(x) = \max(0, x)$$) instead of sigmoid. |
| `use_leaky_relu` | bool | False | Whether to use Leaky ReLU activation ($$f(x) = x$$ if $$x > 0$$, else $$\alpha x$$) instead of sigmoid. |
| `leaky_relu_alpha` | float | 0.01 | The "leakiness" of Leaky ReLU - how much of negative values should pass through ($$\alpha$$ in the equation). |

## Training Your Network

Once your network is created, you'll use the `train` method to teach it:

| Parameter | Type | Default | What it does |
|-----------|------|---------|-------------|
| `X` | np.ndarray | Required | Your input data with shape (number_of_samples, number_of_features) |
| `y` | np.ndarray | Required | Your target data with shape (number_of_samples, number_of_outputs) |
| `epochs` | int | Required | How many complete passes through your dataset the network should make. More epochs generally means better learning, but with diminishing returns. |
| `batch_size` | Optional[int] | None | The number of samples to process before updating the network weights. If None, it uses the entire dataset at once (full batch learning). Smaller batches can help with large datasets and often find better solutions. |

## Making Predictions

After training, you'll want to see what your network has learned:

| Parameter | Type | Default | What it does |
|-----------|------|---------|-------------|
| `X` | np.ndarray | Required | New data you want predictions for, with the same number of features as your training data |

## Visualizing Your Network

neural.py comes with several visualization tools to help you understand what's happening inside your network:

### Training Progress

```python
plot_training_progress(losses, title="Training Progress")
```

Shows you how the error decreases over time, helping you identify if your network is learning properly.

### Confusion Matrix

```python
plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix")
```

For classification tasks, shows you what kinds of mistakes your network is making.

### Network Architecture

```python
plot_network_architecture(layer_sizes, title="Neural Network Architecture")
```

Visualizes your network's structure so you can see how information flows through the layers.

### Decision Boundary

```python
plot_decision_boundary(model, X, y, title="Decision Boundary")
```

For 2D classification problems, shows you how your network divides the input space into different categories.

### Weight Distribution

```python
plot_weight_distribution(model, title="Weight Distribution")
```

Shows the distribution of weights in your network, which can help identify potential issues like exploding gradients. 