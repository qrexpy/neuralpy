import numpy as np
from typing import List, Tuple, Optional, Union, Callable

class NeuralNetwork:
    def __init__(self, layer_sizes: List[int], learning_rate: float = 0.01,
                 use_relu: bool = False, use_leaky_relu: bool = False, 
                 leaky_relu_alpha: float = 0.01, momentum: float = 0.9):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weights = []
        self.biases = []
        self.velocity_w = []  # Momentum velocity for weights
        self.velocity_b = []  # Momentum velocity for biases
        
        if use_relu and use_leaky_relu:
            raise ValueError("Cannot use both ReLU and Leaky ReLU simultaneously")
        
        if use_relu:
            self.activation = self.relu
            self.activation_derivative = self.relu_derivative
        elif use_leaky_relu:
            self.leaky_relu_alpha = leaky_relu_alpha
            self.activation = self.leaky_relu
            self.activation_derivative = self.leaky_relu_derivative
        else:
            self.activation = self.sigmoid
            self.activation_derivative = self.sigmoid_derivative
        
        for i in range(len(layer_sizes) - 1):
            if use_relu or use_leaky_relu:
                scale = np.sqrt(2.0 / layer_sizes[i])  # He initialization
            else:
                scale = np.sqrt(1.0 / layer_sizes[i])  # Xavier initialization
                
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * scale)
            self.biases.append(np.zeros((1, layer_sizes[i + 1])))
            
            # Initialize momentum velocities
            self.velocity_w.append(np.zeros_like(self.weights[-1]))
            self.velocity_b.append(np.zeros_like(self.biases[-1]))
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        return x * (1 - x)
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def relu_derivative(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, 0)
    
    def leaky_relu(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, self.leaky_relu_alpha * x)
    
    def leaky_relu_derivative(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, self.leaky_relu_alpha)
    
    def forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        activations = [X]
        weighted_sums = []
        
        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            weighted_sums.append(z)
            activations.append(self.activation(z))
            
        return activations, weighted_sums
    
    def backward(self, X: np.ndarray, y: np.ndarray, activations: List[np.ndarray], 
                weighted_sums: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        m = X.shape[0]
        delta = activations[-1] - y
        weight_gradients = []
        bias_gradients = []
        
        for i in range(len(self.weights) - 1, -1, -1):
            dW = np.dot(activations[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            
            weight_gradients.insert(0, dW)
            bias_gradients.insert(0, db)
            
            if i > 0:
                if self.activation == self.relu or self.activation == self.leaky_relu:
                    delta = np.dot(delta, self.weights[i].T) * self.activation_derivative(weighted_sums[i-1])
                else:
                    delta = np.dot(delta, self.weights[i].T) * self.activation_derivative(activations[i])
        
        return weight_gradients, bias_gradients
    
    def update_parameters(self, weight_gradients: List[np.ndarray], bias_gradients: List[np.ndarray]) -> None:
        """Update network parameters using momentum"""
        for i in range(len(self.weights)):
            # Update velocity with momentum
            self.velocity_w[i] = self.momentum * self.velocity_w[i] + weight_gradients[i]
            self.velocity_b[i] = self.momentum * self.velocity_b[i] + bias_gradients[i]
            
            # Update parameters using velocity
            self.weights[i] -= self.learning_rate * self.velocity_w[i]
            self.biases[i] -= self.learning_rate * self.velocity_b[i]
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int, 
              batch_size: Optional[int] = None) -> List[float]:
        losses = []
        best_loss = float('inf')
        patience = 50  # Increased patience
        patience_counter = 0
        min_delta = 1e-6  # Smaller improvement threshold
        best_weights = None
        best_biases = None
        
        for epoch in range(epochs):
            if batch_size is None:
                activations, weighted_sums = self.forward(X)
                weight_gradients, bias_gradients = self.backward(X, y, activations, weighted_sums)
                self.update_parameters(weight_gradients, bias_gradients)
                loss = np.mean(np.square(activations[-1] - y))
            else:
                indices = np.random.permutation(len(X))
                X_shuffled = X[indices]
                y_shuffled = y[indices]
                
                for i in range(0, len(X), batch_size):
                    batch_X = X_shuffled[i:i + batch_size]
                    batch_y = y_shuffled[i:i + batch_size]
                    
                    activations, weighted_sums = self.forward(batch_X)
                    weight_gradients, bias_gradients = self.backward(batch_X, batch_y, activations, weighted_sums)
                    self.update_parameters(weight_gradients, bias_gradients)
                
                activations, _ = self.forward(X)
                loss = np.mean(np.square(activations[-1] - y))
            
            losses.append(loss)
            
            # Early stopping with minimum improvement threshold
            if loss < best_loss - min_delta:
                best_loss = loss
                patience_counter = 0
                # Save best model
                best_weights = [w.copy() for w in self.weights]
                best_biases = [b.copy() for b in self.biases]
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    # Restore best model
                    self.weights = best_weights
                    self.biases = best_biases
                    break
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        return losses
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        activations, _ = self.forward(X)
        return activations[-1] 