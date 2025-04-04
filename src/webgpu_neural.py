import random
import numpy as np
import multiprocessing as mp
from functools import partial
from typing import List, Tuple, Optional, Union
from .webgpu_ops import (
    WebGPUMatrix, webgpu_matrix_multiply, webgpu_matrix_add,
    webgpu_matrix_subtract, webgpu_matrix_hadamard,
    webgpu_matrix_scalar_multiply, webgpu_matrix_sum_axis,
    webgpu_matrix_transpose, webgpu_matrix_reshape,
    webgpu_matrix_random, webgpu_matrix_zeros,
    webgpu_matrix_ones, webgpu_matrix_eye, WEBGPU_AVAILABLE
)
import pickle

# Custom exceptions for WebGPU availability
class WebGPUNotAvailableError(ImportError):
    """Exception raised when WebGPU is not available."""
    pass

class WebGPUNeuralNetwork:
    def __init__(
        self,
        layer_sizes: List[int],
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        weight_init: str = 'he',
        num_workers: int = None,
        use_mixed_precision: bool = False
    ):
        # Check if WebGPU is available
        if not WEBGPU_AVAILABLE:
            raise WebGPUNotAvailableError(
                "WebGPU is not available. Please install the 'wgpu' package."
            )
            
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_init = weight_init
        self.num_workers = num_workers or mp.cpu_count()
        self.use_mixed_precision = use_mixed_precision
        
        self.weights = []
        self.biases = []
        self.activations = []
        self.velocity_w = []
        self.velocity_b = []
        self.initialize_weights()
    
    def initialize_weights(self):
        """Initialize weights using different strategies"""
        self.weights = []
        self.biases = []
        self.velocity_w = []
        self.velocity_b = []
        
        for i in range(len(self.layer_sizes) - 1):
            input_size = self.layer_sizes[i]
            output_size = self.layer_sizes[i + 1]
            
            if self.weight_init == 'he':
                # He initialization: weights ~ N(0, sqrt(2/n))
                scale = np.sqrt(2.0 / input_size)
                weights = webgpu_matrix_random(output_size, input_size, -scale, scale)
            elif self.weight_init == 'xavier':
                # Xavier initialization: weights ~ N(0, sqrt(1/n))
                scale = np.sqrt(1.0 / input_size)
                weights = webgpu_matrix_random(output_size, input_size, -scale, scale)
            else:  # random initialization
                weights = webgpu_matrix_random(output_size, input_size, -1, 1)
            
            # Initialize biases to small random values
            biases = webgpu_matrix_random(output_size, 1, -0.1, 0.1)
            
            # Initialize momentum velocities
            velocity_w = webgpu_matrix_zeros(output_size, input_size)
            velocity_b = webgpu_matrix_zeros(output_size, 1)
            
            self.weights.append(weights)
            self.biases.append(biases)
            self.velocity_w.append(velocity_w)
            self.velocity_b.append(velocity_b)
    
    # Helper functions for activation functions
    def _relu(self, x: WebGPUMatrix) -> WebGPUMatrix:
        """ReLU activation function: max(0, x)"""
        result = WebGPUMatrix(x.rows, x.cols)
        result.data = np.maximum(0, x.data)
        result.device.queue.write_buffer(result.buffer, 0, result.data.tobytes())
        return result
    
    def _sigmoid(self, x: WebGPUMatrix) -> WebGPUMatrix:
        """Sigmoid activation function: 1 / (1 + exp(-x))"""
        result = WebGPUMatrix(x.rows, x.cols)
        result.data = 1 / (1 + np.exp(-x.data))
        result.device.queue.write_buffer(result.buffer, 0, result.data.tobytes())
        return result
    
    def _relu_derivative(self, x: WebGPUMatrix) -> WebGPUMatrix:
        """Derivative of ReLU: 1 if x > 0 else 0"""
        result = WebGPUMatrix(x.rows, x.cols)
        result.data = (x.data > 0).astype(np.float32)
        result.device.queue.write_buffer(result.buffer, 0, result.data.tobytes())
        return result
    
    def forward(self, X: WebGPUMatrix) -> WebGPUMatrix:
        """Forward propagation through the network"""
        self.activations = [X]
        current_activation = X
        
        for i in range(len(self.weights)):
            # Linear transformation
            z = webgpu_matrix_add(
                webgpu_matrix_multiply(self.weights[i], current_activation),
                self.biases[i]
            )
            
            # Apply ReLU activation for hidden layers, sigmoid for output
            if i < len(self.weights) - 1:
                current_activation = self._relu(z)
            else:
                current_activation = self._sigmoid(z)
            
            self.activations.append(current_activation)
        
        return current_activation
    
    def backward(
        self,
        X: WebGPUMatrix,
        y: WebGPUMatrix,
        output: WebGPUMatrix
    ) -> Tuple[List[WebGPUMatrix], List[WebGPUMatrix]]:
        """Backward propagation to compute gradients"""
        m = X.cols  # number of training examples
        weight_gradients = []
        bias_gradients = []
        
        # Compute output layer error
        error = webgpu_matrix_subtract(output, y)
        
        # Backpropagate through layers
        for i in range(len(self.weights) - 1, -1, -1):
            # Compute gradients for current layer
            if i == len(self.weights) - 1:  # output layer
                delta = error
            else:  # hidden layers
                # Compute derivative of ReLU
                activation_derivative = self._relu_derivative(self.activations[i + 1])
                delta = webgpu_matrix_hadamard(
                    webgpu_matrix_multiply(webgpu_matrix_transpose(self.weights[i + 1]), error),
                    activation_derivative
                )
            
            # Compute gradients
            weight_grad = webgpu_matrix_multiply(
                delta,
                webgpu_matrix_transpose(self.activations[i])
            )
            weight_grad = webgpu_matrix_scalar_multiply(weight_grad, 1.0 / m)
            
            bias_grad = webgpu_matrix_sum_axis(delta, axis=1)
            bias_grad = webgpu_matrix_scalar_multiply(bias_grad, 1.0 / m)
            
            weight_gradients.insert(0, weight_grad)
            bias_gradients.insert(0, bias_grad)
            
            error = delta
        
        return weight_gradients, bias_gradients
    
    def update_parameters(
        self,
        weight_gradients: List[WebGPUMatrix],
        bias_gradients: List[WebGPUMatrix]
    ):
        """Update weights and biases using momentum"""
        for i in range(len(self.weights)):
            # Update velocity
            self.velocity_w[i] = webgpu_matrix_add(
                webgpu_matrix_scalar_multiply(self.velocity_w[i], self.momentum),
                webgpu_matrix_scalar_multiply(weight_gradients[i], -self.learning_rate)
            )
            self.velocity_b[i] = webgpu_matrix_add(
                webgpu_matrix_scalar_multiply(self.velocity_b[i], self.momentum),
                webgpu_matrix_scalar_multiply(bias_gradients[i], -self.learning_rate)
            )
            
            # Update parameters
            self.weights[i] = webgpu_matrix_add(self.weights[i], self.velocity_w[i])
            self.biases[i] = webgpu_matrix_add(self.biases[i], self.velocity_b[i])
    
    def _process_batch(
        self,
        batch_X: WebGPUMatrix,
        batch_y: WebGPUMatrix
    ) -> Tuple[float, List[WebGPUMatrix], List[WebGPUMatrix]]:
        """Process a single batch"""
        # Forward pass
        output = self.forward(batch_X)
        
        # Compute loss
        loss = webgpu_matrix_subtract(output, batch_y)
        loss = webgpu_matrix_hadamard(loss, loss)
        loss = webgpu_matrix_scalar_multiply(loss, 0.5)
        loss = webgpu_matrix_sum_axis(loss).data[0, 0] / batch_X.cols
        
        # Backward pass
        weight_gradients, bias_gradients = self.backward(batch_X, batch_y, output)
        
        return loss, weight_gradients, bias_gradients
    
    def _prepare_batches(
        self,
        X: WebGPUMatrix,
        y: WebGPUMatrix,
        batch_size: int
    ) -> List[Tuple[WebGPUMatrix, WebGPUMatrix]]:
        """Prepare batches for training"""
        m = X.cols
        indices = list(range(m))
        random.shuffle(indices)
        
        batches = []
        for i in range(0, m, batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_X = WebGPUMatrix(X.rows, len(batch_indices))
            batch_y = WebGPUMatrix(y.rows, len(batch_indices))
            
            for j, idx in enumerate(batch_indices):
                for r in range(X.rows):
                    batch_X[r, j] = X[r, idx]
                for r in range(y.rows):
                    batch_y[r, j] = y[r, idx]
            
            batches.append((batch_X, batch_y))
        
        return batches
    
    def train(
        self,
        X: WebGPUMatrix,
        y: WebGPUMatrix,
        epochs: int = 1000,
        batch_size: int = 32,
        verbose: bool = True
    ) -> List[float]:
        """Train the network"""
        losses = []
        best_loss = float('inf')
        patience = 50
        patience_counter = 0
        min_delta = 1e-6
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            batches = self._prepare_batches(X, y, batch_size)
            
            for batch_X, batch_y in batches:
                loss, weight_grads, bias_grads = self._process_batch(batch_X, batch_y)
                epoch_loss += loss
                self.update_parameters(weight_grads, bias_grads)
            
            # Compute average loss
            avg_loss = epoch_loss / len(batches)
            losses.append(avg_loss)
            
            # Early stopping check
            if avg_loss < best_loss - min_delta:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
        
        return losses
    
    def predict(self, X: WebGPUMatrix) -> WebGPUMatrix:
        """Make predictions for input data"""
        return self.forward(X)
    
    def evaluate(self, X: WebGPUMatrix, y: WebGPUMatrix) -> Tuple[float, float]:
        """Evaluate the network's performance"""
        output = self.predict(X)
        
        # Compute loss
        loss_matrix = webgpu_matrix_subtract(output, y)
        loss_matrix = webgpu_matrix_hadamard(loss_matrix, loss_matrix)
        loss = webgpu_matrix_sum_axis(loss_matrix).data[0, 0] / (2 * X.cols)
        
        # Compute accuracy (for binary classification)
        if output.rows == 1:
            predictions = (output.data > 0.5).astype(np.float32)
            correct = np.sum(np.abs(predictions - y.data) < 0.5)
            accuracy = correct / X.cols
        else:
            # For multi-class, calculate accuracy based on max probability
            predictions = np.zeros_like(output.data)
            for i in range(output.cols):
                max_idx = np.argmax(output.data[:, i])
                predictions[max_idx, i] = 1.0
            
            # For each example, check if the predicted class matches the true class
            correct = 0
            for i in range(y.cols):
                if np.array_equal(predictions[:, i], y.data[:, i]):
                    correct += 1
            
            accuracy = correct / y.cols
        
        return accuracy, loss
    
    def save(self, filename: str):
        """Save the model to a file"""
        # Convert WebGPU matrices to CPU matrices for saving
        weights_cpu = [w.data.tolist() for w in self.weights]
        biases_cpu = [b.data.tolist() for b in self.biases]
        
        model_data = {
            'layer_sizes': self.layer_sizes,
            'learning_rate': self.learning_rate,
            'momentum': self.momentum,
            'weight_init': self.weight_init,
            'use_mixed_precision': self.use_mixed_precision,
            'weights': weights_cpu,
            'biases': biases_cpu
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load(cls, filename: str) -> 'WebGPUNeuralNetwork':
        """Load a model from a file"""
        # Check if WebGPU is available
        if not WEBGPU_AVAILABLE:
            raise WebGPUNotAvailableError(
                "WebGPU is not available. Please install the 'wgpu' package."
            )
            
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create a new model with the saved parameters
        model = cls(
            model_data['layer_sizes'],
            model_data['learning_rate'],
            model_data['momentum'],
            model_data['weight_init'],
            use_mixed_precision=model_data.get('use_mixed_precision', False)
        )
        
        # Load weights and biases
        for i in range(len(model_data['weights'])):
            model.weights[i].data = np.array(model_data['weights'][i], dtype=np.float32)
            model.biases[i].data = np.array(model_data['biases'][i], dtype=np.float32)
            model.weights[i].device.queue.write_buffer(model.weights[i].buffer, 0, model.weights[i].data.tobytes())
            model.biases[i].device.queue.write_buffer(model.biases[i].buffer, 0, model.biases[i].data.tobytes())
        
        return model 