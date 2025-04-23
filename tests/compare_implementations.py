import os
import sys
import time
import numpy as np
import unittest
import random
import matplotlib.pyplot as plt
from rich.console import Console

# Create a plain console that works with Windows CMD
console = Console(safe_box=True, highlight=False, emoji=False, legacy_windows=True)

# Custom simple progress tracker that works in all terminals
class SimpleProgress:
    def __init__(self, console=None):
        self.console = console or Console(safe_box=True, highlight=False, emoji=False, legacy_windows=True)
        self.tasks = {}
        self.task_id = 0
    
    @staticmethod
    def safe_print(message, end="\n"):
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
            print(message, end=end)
        except Exception:
            # Super safe fallback
            print(str(message).encode('ascii', 'replace').decode('ascii'), end=end)
        sys.stdout.flush()
    
    def add_task(self, description, total=100):
        task_id = self.task_id
        self.task_id += 1
        self.tasks[task_id] = {
            "description": description,
            "total": total,
            "completed": 0,
            "start_time": time.time()
        }
        SimpleProgress.safe_print(f"{description} [0%]")
        return task_id
    
    def update(self, task_id, advance=None, completed=None):
        if task_id not in self.tasks:
            return
            
        task = self.tasks[task_id]
        
        if advance is not None:
            task["completed"] += advance
        if completed is not None:
            task["completed"] = completed
            
        # Ensure we don't exceed 100%
        task["completed"] = min(task["completed"], task["total"])
        
        # Calculate percentage
        percentage = int(100 * task["completed"] / task["total"])
        
        # Create a simple ASCII progress bar
        bar_length = 30
        filled_length = int(bar_length * task["completed"] / task["total"])
        bar = "=" * filled_length + "-" * (bar_length - filled_length)
        
        # Calculate time elapsed
        elapsed = time.time() - task["start_time"]
        
        # Print progress
        SimpleProgress.safe_print(f"\r{task['description']} [{percentage}%] [{bar}] {elapsed:.1f}s", end="")
        
        # Print newline when complete
        if task["completed"] >= task["total"]:
            SimpleProgress.safe_print("")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        SimpleProgress.safe_print("")  # Ensure we end with a newline

# Use SimpleProgress instead of Rich's Progress
Progress = SimpleProgress

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.pure_neural import PureNeuralNetwork
from src.neural import NeuralNetwork
from src.math_ops import Matrix
try:
    from src.webgpu_neural import WebGPUNeuralNetwork
    from src.webgpu_ops import WebGPUMatrix, WEBGPU_AVAILABLE
except ImportError:
    WEBGPU_AVAILABLE = False
from src.visualization import plot_training_progress, plot_network_architecture

class NeuralNetworkTests(unittest.TestCase):
    def setUp(self):
        """Set up test cases"""
        self.console = console  # Use the same console instance
        self.xor_data = {
            'X': [[0, 0], [0, 1], [1, 0], [1, 1]],
            'y': [[0], [1], [1], [0]]
        }
        
        # Simple digit patterns (0 and 1)
        self.digit_data = {
            'X': [
                # Digit 0 pattern
                [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1],
                # Digit 1 pattern
                [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]
            ],
            'y': [[0], [1]]
        }
        
        # Generate larger dataset for performance testing
        self.large_data = self._generate_large_dataset(1000, 10)

    def _generate_large_dataset(self, n_samples, n_features):
        """Generate a larger dataset for performance testing"""
        X = []
        y = []
        
        for _ in range(n_samples):
            # Generate random features
            sample = [random.random() for _ in range(n_features)]
            X.append(sample)
            
            # Generate target (simple function of features)
            target = 1 if sum(sample) > n_features/2 else 0
            y.append([target])
        
        return {'X': X, 'y': y}

    def test_xor_pure_implementation(self):
        """Test XOR problem with pure Python implementation"""
        X_matrix = Matrix(2, 4, [[self.xor_data['X'][j][i] for j in range(4)] for i in range(2)])
        y_matrix = Matrix(1, 4, [[self.xor_data['y'][j][i] for j in range(4)] for i in range(1)])
        
        network = PureNeuralNetwork(
            [2, 32, 16, 1],  # Even wider network
            learning_rate=0.1,  # Higher learning rate
            use_relu=True,
            momentum=0.9
        )
        
        # Plot network architecture before training
        plot_network_architecture(network.layer_sizes, title="Pure Python Network Architecture")
        
        losses = network.train(X_matrix, y_matrix, epochs=10000, batch_size=4)  # More epochs
        
        # Plot training progress
        plot_training_progress(losses, title="Pure Python Training Progress")
        
        predictions = network.predict(X_matrix)
        accuracy, final_loss = network.evaluate(X_matrix, y_matrix)
        
        self.assertGreater(accuracy, 0.95, "XOR accuracy should be above 95%")
        self.assertLess(final_loss, 0.1, "XOR loss should be below 0.1")

    def test_xor_numpy_implementation(self):
        """Test XOR problem with NumPy implementation"""
        X = np.array(self.xor_data['X'])
        y = np.array(self.xor_data['y'])

        network = NeuralNetwork(
            [2, 32, 16, 1],  # Even wider network
            learning_rate=0.05,  # Adjusted learning rate for better convergence
            use_relu=False,  # Use Sigmoid instead of ReLU
            momentum=0.9
        )

        # Plot network architecture before training
        plot_network_architecture(network.layer_sizes, title="NumPy Network Architecture")

        losses = network.train(X, y, epochs=5000, batch_size=4)  # Increased epochs for better training

        # Plot training progress
        plot_training_progress(losses, title="NumPy Training Progress")

        predictions = network.predict(X)
        accuracy = np.mean(np.round(predictions) == y)
        final_loss = np.mean(np.square(predictions - y)) / 2

        self.assertGreater(accuracy, 0.95, "XOR accuracy should be above 95%")
        self.assertLess(final_loss, 0.1, "XOR loss should be below 0.1")

    def test_xor_webgpu_implementation(self):
        """Test XOR problem with WebGPU implementation"""
        if not WEBGPU_AVAILABLE:
            self.console.print("WebGPU implementation not available: wgpu module not installed")
            self.skipTest("WebGPU not available")
            return
            
        X_matrix = Matrix(2, 4, [[self.xor_data['X'][j][i] for j in range(4)] for i in range(2)])
        y_matrix = Matrix(1, 4, [[self.xor_data['y'][j][i] for j in range(4)] for i in range(1)])
        
        # Convert data to WebGPU format
        X_gpu = WebGPUMatrix(2, 4, X_matrix.data)
        y_gpu = WebGPUMatrix(1, 4, y_matrix.data)
        
        # Test both mixed precision and FP32
        for use_mixed_precision in [True, False]:
            network = WebGPUNeuralNetwork(
                [2, 32, 16, 1], 
                learning_rate=0.1, 
                momentum=0.9,
                use_mixed_precision=use_mixed_precision
            )
            
            # Plot network architecture before training
            plot_network_architecture(
                network.layer_sizes, 
                title=f"WebGPU Network Architecture ({'Mixed Precision' if use_mixed_precision else 'FP32'})"
            )
            
            losses = network.train(X_gpu, y_gpu, epochs=10000, batch_size=4)
            
            # Plot training progress
            plot_training_progress(
                losses, 
                title=f"WebGPU Training Progress ({'Mixed Precision' if use_mixed_precision else 'FP32'})"
            )
            
            predictions = network.predict(X_gpu)
            accuracy = np.mean((np.round(predictions.data) > 0.5) == (y_matrix.data > 0.5))
            final_loss = np.mean(np.square(predictions.data - y_matrix.data)) / 2
            
            self.assertGreater(accuracy, 0.95, "XOR accuracy should be above 95%")
            self.assertLess(final_loss, 0.1, "XOR loss should be below 0.1")

    def test_digit_recognition_pure_implementation(self):
        """Test digit recognition with pure Python implementation"""
        X_matrix = Matrix(20, 2, [[self.digit_data['X'][j][i] for j in range(2)] for i in range(20)])
        y_matrix = Matrix(1, 2, [[self.digit_data['y'][j][i] for j in range(2)] for i in range(1)])
        
        network = PureNeuralNetwork(
            [20, 16, 8, 1],  # Deeper network
            learning_rate=0.01,
            use_relu=True,
            momentum=0.9
        )
        
        # Plot network architecture before training
        plot_network_architecture(network.layer_sizes, title="Pure Python Digit Recognition Network")
        
        losses = network.train(X_matrix, y_matrix, epochs=2000)  # More epochs
        
        # Plot training progress
        plot_training_progress(losses, title="Pure Python Digit Recognition Training")
        
        predictions = network.predict(X_matrix)
        accuracy, final_loss = network.evaluate(X_matrix, y_matrix)
        
        self.assertGreater(accuracy, 0.9, "Digit recognition accuracy should be above 90%")
        self.assertLess(final_loss, 0.2, "Digit recognition loss should be below 0.2")

    def test_digit_recognition_numpy_implementation(self):
        """Test digit recognition with NumPy implementation"""
        X = np.array(self.digit_data['X'])
        y = np.array(self.digit_data['y'])
        
        network = NeuralNetwork(
            [20, 16, 8, 1],  # Deeper network
            learning_rate=0.01,
            use_relu=True,
            momentum=0.9
        )
        
        # Plot network architecture before training
        plot_network_architecture(network.layer_sizes, title="NumPy Digit Recognition Network")
        
        losses = network.train(X, y, epochs=2000)  # More epochs
        
        # Plot training progress
        plot_training_progress(losses, title="NumPy Digit Recognition Training")
        
        predictions = network.predict(X)
        accuracy = np.mean(np.round(predictions) == y)
        final_loss = np.mean(np.square(predictions - y)) / 2
        
        self.assertGreater(accuracy, 0.9, "Digit recognition accuracy should be above 90%")
        self.assertLess(final_loss, 0.2, "Digit recognition loss should be below 0.2")

    def test_digit_recognition_webgpu_implementation(self):
        """Test digit recognition with WebGPU implementation"""
        if not WEBGPU_AVAILABLE:
            self.console.print("WebGPU implementation not available: wgpu module not installed")
            self.skipTest("WebGPU not available")
            return
            
        X_matrix = Matrix(20, 2, [[self.digit_data['X'][j][i] for j in range(2)] for i in range(20)])
        y_matrix = Matrix(1, 2, [[self.digit_data['y'][j][i] for j in range(2)] for i in range(1)])
        
        # Convert data to WebGPU format
        X_gpu = WebGPUMatrix(20, 2, X_matrix.data)
        y_gpu = WebGPUMatrix(1, 2, y_matrix.data)
        
        # Test both mixed precision and FP32
        for use_mixed_precision in [True, False]:
            network = WebGPUNeuralNetwork(
                [20, 16, 8, 1], 
                learning_rate=0.01, 
                momentum=0.9,
                use_mixed_precision=use_mixed_precision
            )
            
            # Plot network architecture before training
            plot_network_architecture(
                network.layer_sizes, 
                title=f"WebGPU Digit Recognition Network ({'Mixed Precision' if use_mixed_precision else 'FP32'})"
            )
            
            losses = network.train(X_gpu, y_gpu, epochs=2000, batch_size=32)
            
            # Plot training progress
            plot_training_progress(
                losses, 
                title=f"WebGPU Digit Recognition Training ({'Mixed Precision' if use_mixed_precision else 'FP32'})"
            )
            
            predictions = network.predict(X_gpu)
            accuracy = np.mean((np.round(predictions.data) > 0.5) == (y_matrix.data > 0.5))
            final_loss = np.mean(np.square(predictions.data - y_matrix.data)) / 2
            
            self.assertGreater(accuracy, 0.9, "Digit recognition accuracy should be above 90%")
            self.assertLess(final_loss, 0.2, "Digit recognition loss should be below 0.2")

    def test_performance_comparison(self):
        """Compare performance of different implementations"""
        self._safe_print("Running performance tests...")
        
        # Convert data to Matrix format
        X_matrix = Matrix(len(self.large_data['X'][0]), len(self.large_data['X']), 
                         [[self.large_data['X'][j][i] for j in range(len(self.large_data['X']))] 
                          for i in range(len(self.large_data['X'][0]))])
        y_matrix = Matrix(1, len(self.large_data['y']), 
                         [[self.large_data['y'][j][i] for j in range(len(self.large_data['y']))] 
                          for i in range(1)])
        
        # Convert to NumPy arrays
        X_np = np.array(self.large_data['X'])
        y_np = np.array(self.large_data['y'])
        
        # Set up common parameters
        layer_sizes = [len(self.large_data['X'][0]), 32, 16, 1]
        epochs = 100
        batch_size = 32
        
        # Dictionary to store performance metrics
        configs = []
        
        # Use a single progress instance for all tests to prevent flickering
        with Progress(console=self.console) as progress:
            # Test Pure Python implementation (single-threaded)
            self._safe_print("Testing Pure Python (Single-threaded)...")
            task = progress.add_task("Testing Pure Python (Single-threaded)...", total=100)
            
            pure_net = PureNeuralNetwork(
                layer_sizes, 
                learning_rate=0.01, 
                use_relu=True, 
                momentum=0.9,
                num_workers=1
            )
            
            start_time = time.time()
            losses = pure_net.train(X_matrix, y_matrix, epochs=epochs, batch_size=batch_size, verbose=False)
            train_time = time.time() - start_time
            progress.update(task, advance=50)
            
            start_time = time.time()
            _ = pure_net.predict(X_matrix)
            infer_time = time.time() - start_time
            progress.update(task, advance=50)
            
            configs.append({
                'name': 'Pure Python (Single-Threaded)',
                'model': pure_net,
                'train_time': train_time,
                'infer_time': infer_time,
                'final_loss': losses[-1],
                'color': 'blue'
            })
            
            # Test Pure Python implementation (parallel)
            self._safe_print("Testing Pure Python (Parallel)...")
            task = progress.add_task("Testing Pure Python (Parallel)...", total=100)
            
            pure_net_parallel = PureNeuralNetwork(
                layer_sizes, 
                learning_rate=0.01, 
                use_relu=True, 
                momentum=0.9,
                num_workers=os.cpu_count()
            )
            
            start_time = time.time()
            losses = pure_net_parallel.train(X_matrix, y_matrix, epochs=epochs, batch_size=batch_size, verbose=False)
            train_time = time.time() - start_time
            progress.update(task, advance=50)
            
            start_time = time.time()
            _ = pure_net_parallel.predict(X_matrix)
            infer_time = time.time() - start_time
            progress.update(task, advance=50)
            
            configs.append({
                'name': 'Pure Python (Parallel)',
                'model': pure_net_parallel,
                'train_time': train_time,
                'infer_time': infer_time,
                'final_loss': losses[-1],
                'color': 'green'
            })
            
            # Test NumPy implementation
            self._safe_print("Testing NumPy implementation...")
            task = progress.add_task("Testing NumPy implementation...", total=100)
            
            numpy_net = NeuralNetwork(
                layer_sizes, 
                learning_rate=0.01, 
                use_relu=True, 
                momentum=0.9
            )
            
            start_time = time.time()
            losses = numpy_net.train(X_np, y_np, epochs=epochs, batch_size=batch_size)
            train_time = time.time() - start_time
            progress.update(task, advance=50)
            
            start_time = time.time()
            _ = numpy_net.predict(X_np)
            infer_time = time.time() - start_time
            progress.update(task, advance=50)
            
            configs.append({
                'name': 'NumPy',
                'model': numpy_net,
                'train_time': train_time,
                'infer_time': infer_time,
                'final_loss': losses[-1],
                'color': 'red'
            })
            
            # Test WebGPU implementation if available
            if WEBGPU_AVAILABLE:
                try:
                    self._safe_print("Testing WebGPU implementation...")
                    
                    # Convert to WebGPU format
                    X_gpu = WebGPUMatrix(X_matrix.rows, X_matrix.cols, X_matrix.data)
                    y_gpu = WebGPUMatrix(y_matrix.rows, y_matrix.cols, y_matrix.data)
                    
                    # Test WebGPU with mixed precision
                    task = progress.add_task("Testing WebGPU (Mixed Precision)...", total=100)
                    webgpu_net_mixed = WebGPUNeuralNetwork(
                        layer_sizes, 
                        learning_rate=0.01, 
                        momentum=0.9,
                        use_mixed_precision=True
                    )
                    
                    start_time = time.time()
                    losses = webgpu_net_mixed.train(X_gpu, y_gpu, epochs=epochs, batch_size=batch_size, verbose=False)
                    train_time = time.time() - start_time
                    progress.update(task, advance=50)
                    
                    start_time = time.time()
                    _ = webgpu_net_mixed.predict(X_gpu)
                    infer_time = time.time() - start_time
                    progress.update(task, advance=50)
                    
                    configs.append({
                        'name': 'WebGPU (Mixed Precision)',
                        'model': webgpu_net_mixed,
                        'train_time': train_time,
                        'infer_time': infer_time,
                        'final_loss': losses[-1],
                        'color': 'purple'
                    })
                    
                    # Test WebGPU with FP32
                    task = progress.add_task("Testing WebGPU (FP32)...", total=100)
                    webgpu_net_fp32 = WebGPUNeuralNetwork(
                        layer_sizes, 
                        learning_rate=0.01, 
                        momentum=0.9,
                        use_mixed_precision=False
                    )
                    
                    start_time = time.time()
                    losses = webgpu_net_fp32.train(X_gpu, y_gpu, epochs=epochs, batch_size=batch_size, verbose=False)
                    train_time = time.time() - start_time
                    progress.update(task, advance=50)
                    
                    start_time = time.time()
                    _ = webgpu_net_fp32.predict(X_gpu)
                    infer_time = time.time() - start_time
                    progress.update(task, advance=50)
                    
                    configs.append({
                        'name': 'WebGPU (FP32)',
                        'model': webgpu_net_fp32,
                        'train_time': train_time,
                        'infer_time': infer_time,
                        'final_loss': losses[-1],
                        'color': 'orange'
                    })
                except Exception as e:
                    self._safe_print(f"WebGPU implementation not available: {str(e)}")
            else:
                self._safe_print("WebGPU implementation not available: wgpu module not installed")
        
        # Display performance comparison
        self._safe_print("\n[bold]Performance Comparison:[/bold]")
        for config in configs:
            self._safe_print(
                f"[{config['color']}]{config['name']}[/{config['color']}]: "
                f"Training: {config['train_time']:.4f}s, "
                f"Inference: {config['infer_time']:.6f}s, "
                f"Final Loss: {config['final_loss']:.6f}"
            )
        
        # Plot performance comparison
        plt.figure(figsize=(12, 8))
        names = [config['name'] for config in configs]
        train_times = [config['train_time'] for config in configs]
        infer_times = [config['infer_time'] for config in configs]
        colors = [config['color'] for config in configs]
        
        x = np.arange(len(names))
        width = 0.35
        
        plt.bar(x - width/2, train_times, width, label='Training Time (s)', alpha=0.8)
        plt.bar(x + width/2, infer_times, width, label='Inference Time (s)', alpha=0.8)
        
        plt.xlabel('Implementation')
        plt.ylabel('Time (s)')
        plt.title('Performance Comparison')
        plt.xticks(x, names, rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('performance_comparison.png')
        plt.close()
        
        # Check that implementations achieve similar loss
        if len(configs) > 1:
            base_loss = configs[0]['final_loss']
            for config in configs[1:]:
                # Allow for some variation in loss values
                self.assertLess(
                    abs(config['final_loss'] - base_loss), 
                    0.5, 
                    f"{config['name']} loss differs significantly from {configs[0]['name']}"
                )
        
            # Assert that specialized implementations are faster
            pure_time = configs[0]['train_time']
            for config in configs[1:]:
                if "NumPy" in config['name'] or "WebGPU" in config['name']:
                    # Specialized implementations should be faster (at least 10% speedup)
                    self.assertLess(
                        config['train_time'], 
                        pure_time * 0.9, 
                        f"{config['name']} should be significantly faster than {configs[0]['name']}"
                    )

    def _safe_print(self, message):
        """Print a message in a way that's safe for all terminals"""
        try:
            self.console.print(message)
        except Exception:
            # Fallback to basic print if rich console fails
            print(message)
        sys.stdout.flush()

if __name__ == '__main__':
    unittest.main()