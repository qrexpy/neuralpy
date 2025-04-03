import os
import sys
import time
import numpy as np

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.pure_neural import PureNeuralNetwork
from src.neural import NeuralNetwork
from src.math_ops import Matrix

def generate_xor_data(n_samples=1000):
    """Generate XOR dataset"""
    X = []
    y = []
    for _ in range(n_samples):
        x1 = np.random.choice([0, 1])
        x2 = np.random.choice([0, 1])
        X.append([x1, x2])
        y.append([x1 ^ x2])
    return X, y

def generate_digit_data(n_samples=1000):
    """Generate simple digit recognition dataset (0 and 1)"""
    X = []
    y = []
    for _ in range(n_samples):
        digit = np.random.choice([0, 1])
        if digit == 0:
            # Create a simple '0' pattern
            x = [
                1, 1, 1, 1,
                1, 0, 0, 1,
                1, 0, 0, 1,
                1, 0, 0, 1,
                1, 1, 1, 1
            ]
        else:
            # Create a simple '1' pattern
            x = [
                0, 0, 0, 1,
                0, 0, 1, 1,
                0, 1, 0, 1,
                0, 0, 0, 1,
                0, 0, 0, 1
            ]
        X.append(x)
        y.append([digit])
    return X, y

def convert_to_matrix(X, y):
    """Convert data to Matrix format"""
    # Transpose data to match Matrix format (features as rows, samples as columns)
    X_transposed = [[X[j][i] for j in range(len(X))] for i in range(len(X[0]))]
    y_transposed = [[y[j][i] for j in range(len(y))] for i in range(len(y[0]))]
    
    X_matrix = Matrix(len(X[0]), len(X), X_transposed)
    y_matrix = Matrix(len(y[0]), len(y), y_transposed)
    
    return X_matrix, y_matrix

def compare_implementations(task='xor', n_samples=1000, epochs=1000, batch_size=32):
    """Compare pure Python and NumPy implementations"""
    print(f"\nComparing implementations on {task.upper()} task:")
    print(f"Samples: {n_samples}, Epochs: {epochs}, Batch size: {batch_size}")
    
    # Generate data
    if task == 'xor':
        X, y = generate_xor_data(n_samples)
        layer_sizes = [2, 4, 1]
    else:  # digit recognition
        X, y = generate_digit_data(n_samples)
        layer_sizes = [20, 10, 1]
    
    # Convert data to appropriate formats
    X_matrix, y_matrix = convert_to_matrix(X, y)
    X_numpy = np.array(X)
    y_numpy = np.array(y)
    
    # Test pure Python implementation
    print("\nTesting Pure Python implementation:")
    start_time = time.time()
    
    pure_network = PureNeuralNetwork(
        layer_sizes=layer_sizes,
        learning_rate=0.01,
        use_relu=True
    )
    
    pure_losses = pure_network.train(X_matrix, y_matrix, epochs=epochs, batch_size=batch_size)
    pure_time = time.time() - start_time
    
    # Test predictions
    pure_predictions = pure_network.predict(X_matrix)
    pure_accuracy, pure_final_loss = pure_network.evaluate(X_matrix, y_matrix)
    
    print(f"Training time: {pure_time:.2f} seconds")
    print(f"Final loss: {pure_final_loss:.6f}")
    print(f"Accuracy: {pure_accuracy:.2%}")
    print("First 10 predictions:", [[pure_predictions[i, j] for i in range(pure_predictions.rows)] for j in range(10)])
    
    # Test NumPy implementation
    print("\nTesting NumPy implementation:")
    start_time = time.time()
    
    numpy_network = NeuralNetwork(
        layer_sizes=layer_sizes,
        learning_rate=0.01,
        use_relu=True
    )
    
    numpy_losses = numpy_network.train(X_numpy, y_numpy, epochs=epochs, batch_size=batch_size)
    numpy_time = time.time() - start_time
    
    # Test predictions
    numpy_predictions = numpy_network.predict(X_numpy[:10])
    numpy_accuracy = np.mean(np.round(numpy_predictions) == y_numpy[:10])
    numpy_final_loss = np.mean(np.square(numpy_predictions - y_numpy[:10])) / 2
    
    print(f"Training time: {numpy_time:.2f} seconds")
    print(f"Final loss: {numpy_final_loss:.6f}")
    print(f"Accuracy: {numpy_accuracy:.2%}")
    print("First 10 predictions:", numpy_predictions.tolist())
    
    # Compare results
    print("\nComparison:")
    print(f"Time difference: {numpy_time - pure_time:.2f} seconds")
    print(f"Loss difference: {numpy_final_loss - pure_final_loss:.6f}")
    print(f"Accuracy difference: {numpy_accuracy - pure_accuracy:.2%}")
    
    return {
        'pure': {
            'time': pure_time,
            'loss': pure_final_loss,
            'accuracy': pure_accuracy,
            'predictions': [[pure_predictions[i, j] for i in range(pure_predictions.rows)] for j in range(10)]
        },
        'numpy': {
            'time': numpy_time,
            'loss': numpy_final_loss,
            'accuracy': numpy_accuracy,
            'predictions': numpy_predictions.tolist()
        }
    }

if __name__ == '__main__':
    # Compare on XOR task
    xor_results = compare_implementations(
        task='xor',
        n_samples=1000,
        epochs=1000,
        batch_size=32
    )
    
    # Compare on digit recognition task
    digit_results = compare_implementations(
        task='digit',
        n_samples=1000,
        epochs=1000,
        batch_size=32
    ) 