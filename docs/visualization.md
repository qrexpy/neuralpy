# Seeing is Believing: Visualization Tools

Ever wondered what's happening inside your neural network? neural.py comes with powerful visualization tools that let you peek under the hood and understand how your network learns and makes decisions.

## Your Visualization Toolkit

### Training Progress: Watch Your Network Learn

```python
plot_training_progress(losses, title="Training Progress")
```

This creates a beautiful graph showing how your network's error decreases over time. It's like watching your network get smarter before your eyes!

**Parameters:**
- `losses`: A list of error values from each training epoch
- `title`: What to call your graph (default: "Training Progress")

**What You'll See:**
A line chart with epochs on the x-axis and loss value on the y-axis, described by the equation:

$$Loss(epoch) = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

Where $$\hat{y}_i$$ is your network's prediction.

**Example:**
```python
losses = nn.train(X, y, epochs=1000)
plot_training_progress(losses, "My Network Learning XOR")
```

### Confusion Matrix: Understand Your Network's Mistakes

```python
plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix")
```

A color-coded grid showing which classes your network confuses with each other. Perfect for classification tasks!

**Parameters:**
- `y_true`: The correct labels
- `y_pred`: Your network's predictions
- `title`: The title for your visualization

**What You'll See:**
A heat map where each cell $$(i,j)$$ shows how many samples of true class $$i$$ were predicted as class $$j$$.

**Example:**
```python
predictions = nn.predict(X)
y_pred = np.argmax(predictions, axis=1)
y_true = np.argmax(y, axis=1)
plot_confusion_matrix(y_true, y_pred, "Digit Recognition Results")
```

### Network Architecture: Visualize Your Network's Structure

```python
plot_network_architecture(layer_sizes, title="Neural Network Architecture")
```

Creates a beautiful diagram of your network's layers and connections.

**Parameters:**
- `layer_sizes`: List of neurons in each layer (e.g., [2, 4, 1])
- `title`: What to name your visualization

**What You'll See:**
A network diagram with nodes representing neurons and connections showing how information flows between layers.

**Example:**
```python
plot_network_architecture([2, 4, 1], "XOR Network Structure")
```

### Decision Boundary: See How Your Network Classifies

```python
plot_decision_boundary(model, X, y, title="Decision Boundary")
```

This creates a colorful visualization of how your network divides the input space into different categories.

**Parameters:**
- `model`: Your trained neural network
- `X`: Input data points
- `y`: True labels
- `title`: Title for your visualization

**What You'll See:**
A 2D plot where different colors represent different decision regions, separated by the boundary line defined by:

$$f(x_1, x_2) = W_2 \cdot \sigma(W_1 \cdot [x_1, x_2] + b_1) + b_2 = 0.5$$

Where $$\sigma$$ is your activation function.

**Example:**
```python
plot_decision_boundary(nn, X, y, "XOR Decision Regions")
```

### Weight Distribution: Analyze Your Network's Parameters

```python
plot_weight_distribution(model, title="Weight Distribution")
```

This shows the distribution of weights in your network as a histogram.

**Parameters:**
- `model`: Your trained neural network
- `title`: What to title your visualization

**What You'll See:**
A histogram showing the distribution of weight values in your network, letting you check for potential issues like vanishing or exploding gradients.

**Example:**
```python
plot_weight_distribution(nn, "Weight Distribution After Training")
```

## Making Customized Visualizations

All visualization functions return matplotlib objects, so you can further customize them:

```python
plt = plot_training_progress(losses)
plt.ylim(0, 0.5)  # Set y-axis limits
plt.savefig("training_progress.png", dpi=300)  # Save high-resolution image
```

## Dependencies

These visualizations use:
- matplotlib
- seaborn
- networkx (for the network architecture plots)

All of these are included in the requirements.txt file, so you're ready to go! 