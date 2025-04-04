import os
import sys
import datetime
import logging
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint
from rich.logging import RichHandler
from rich.traceback import install

# Install rich's traceback handler for more readable tracebacks
install()

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from tests.compare_implementations import NeuralNetworkTests, SimpleProgress
    from src.webgpu_ops import WEBGPU_AVAILABLE
except ImportError as e:
    print(f"Error importing test modules: {e}")
    WEBGPU_AVAILABLE = False

import unittest
import warnings

# Configure logging
LOG_DIR = os.path.join(os.path.dirname(__file__), 'log')
os.makedirs(LOG_DIR, exist_ok=True)

# Fallback print function for terminal compatibility
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

# Initialize console with ASCII-only settings
try:
    console = Console(
        safe_box=True,         # Use ASCII for boxes
        highlight=False,       # Disable syntax highlighting
        emoji=False,           # Disable emojis
        legacy_windows=True    # Force legacy Windows console mode
    )
except Exception:
    # If Rich console creation fails, define a simple fallback
    class SimpleConsole:
        def print(self, message, end="\n"):
            safe_print(message, end)
        
        def status(self, message):
            class DummyContext:
                def __enter__(self):
                    safe_print(message)
                    return self
                def __exit__(self, *args):
                    pass
            return DummyContext()
    
    console = SimpleConsole()

# Setup logging
logger = logging.getLogger("neuralpy")
logger.setLevel(logging.INFO)

def setup_logging(log_option):
    """Set up logging based on user preference"""
    # Remove all handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Set up based on option
    timestamp = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    log_file = os.path.join(LOG_DIR, f"{timestamp}.txt")
    
    if log_option == "file":
        # Log to file only
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        # Suppress warnings but show errors
        warnings.filterwarnings("ignore")
        
    elif log_option == "both":
        # Log to file and console
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        try:
            console_handler = RichHandler(console=console, show_time=False)
            console_handler.setFormatter(logging.Formatter('%(message)s'))
            logger.addHandler(console_handler)
        except Exception:
            # Fallback to standard stream handler
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
            logger.addHandler(stream_handler)
            
        # Show all warnings
        warnings.filterwarnings("default")
        
    else:  # "none"
        # No logging
        logger.addHandler(logging.NullHandler())
        # Suppress warnings and non-error outputs
        warnings.filterwarnings("ignore")
    
    return log_file if log_option in ["file", "both"] else None

def get_log_preference():
    """Get user preference for logging"""
    safe_print("\nLogging Options:")
    safe_print("1. Save logs to file only")
    safe_print("2. Show logs in console and save to file")
    safe_print("3. Don't log (show errors only)")
    
    while True:
        try:
            choice = input("Select logging option (default: 3): ")
            if choice == "":
                choice = "3"
                
            choice = int(choice)
            if choice == 1:
                return "file"
            elif choice == 2:
                return "both"
            elif choice == 3:
                return "none"
            else:
                safe_print("Invalid choice. Please enter 1, 2, or 3.")
        except ValueError:
            safe_print("Please enter a valid number.")

def show_welcome():
    """Display welcome message"""
    safe_print("\n========================================================")
    safe_print("        Neural Network Implementation Tests")
    safe_print("========================================================")
    safe_print("This test suite compares different neural network implementations:")
    safe_print("- Pure Python (Single-threaded and Parallel)")
    safe_print("- NumPy")
    
    if WEBGPU_AVAILABLE:
        safe_print("- WebGPU (with Mixed Precision support)")
    else:
        safe_print("- WebGPU (not available)")
    
    safe_print("========================================================\n")

def get_model_selection():
    """Get user selection for model type"""
    safe_print("\nSelect Model Type:")
    safe_print("1. Pure Python (Single-threaded)")
    safe_print("2. Pure Python (Parallel)")
    safe_print("3. NumPy")
    
    if WEBGPU_AVAILABLE:
        safe_print("4. WebGPU (Mixed Precision)")
        safe_print("5. WebGPU (FP32)")
    else:
        safe_print("4. WebGPU (Mixed Precision) (not available)")
        safe_print("5. WebGPU (FP32) (not available)")
    
    safe_print("6. All Implementations")
    
    max_option = 6
    while True:
        try:
            choice = input("Enter your choice (default: 6): ")
            if choice == "":
                choice = "6"
                
            choice = int(choice)
            if 1 <= choice <= max_option:
                # Skip WebGPU options if not available
                if not WEBGPU_AVAILABLE and choice in [4, 5]:
                    safe_print("WebGPU is not available on this system. Please choose another option.")
                    continue
                return choice
            safe_print(f"Invalid choice. Please enter a number between 1 and {max_option}.")
        except ValueError:
            safe_print("Please enter a valid number.")

def get_test_selection():
    """Get user selection for test type"""
    safe_print("\nSelect Test Type:")
    safe_print("1. XOR Problem")
    safe_print("2. Digit Recognition")
    safe_print("3. Performance Comparison")
    safe_print("4. All Tests")
    
    while True:
        try:
            choice = input("Enter your choice (default: 4): ")
            if choice == "":
                choice = "4"
                
            choice = int(choice)
            if 1 <= choice <= 4:
                return choice
            safe_print("Invalid choice. Please enter a number between 1 and 4.")
        except ValueError:
            safe_print("Please enter a valid number.")

def get_test_parameters():
    """Get test parameters from user"""
    safe_print("\nTest Parameters:")
    
    try:
        n_samples = input("Number of samples for performance test (default: 1000): ")
        n_samples = 1000 if n_samples == "" else int(n_samples)
        
        epochs = input("Number of epochs (default: 100): ")
        epochs = 100 if epochs == "" else int(epochs)
        
        batch_size = input("Batch size (default: 32): ")
        batch_size = 32 if batch_size == "" else int(batch_size)
        
        return n_samples, epochs, batch_size
    except ValueError:
        safe_print("Invalid input. Using default values.")
        return 1000, 100, 32

class LoggingTestRunner(unittest.TextTestRunner):
    """Custom test runner that supports logging control"""
    def __init__(self, log_option="none", **kwargs):
        super().__init__(**kwargs)
        self.log_option = log_option

    def run(self, test):
        # Redirect stdout/stderr if not logging to console
        if self.log_option != "both":
            # We only want to show errors, not all test output
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = open(os.devnull, 'w')
            # Keep stderr for errors
            
        result = super().run(test)
        
        # Restore stdout/stderr
        if self.log_option != "both":
            sys.stdout.close()
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
        return result

def run_selected_tests(model_choice, test_choice, n_samples, epochs, batch_size, log_option):
    """Run selected tests"""
    suite = unittest.TestSuite()
    test_case = NeuralNetworkTests()
    
    try:
        # Add tests based on user selections
        if test_choice in [1, 4]:  # XOR or All Tests
            if model_choice in [1, 6]:  # Pure Python or All
                suite.addTest(NeuralNetworkTests('test_xor_pure_implementation'))
            if model_choice in [2, 6]:  # Parallel or All
                suite.addTest(NeuralNetworkTests('test_xor_pure_implementation'))
            if model_choice in [3, 6]:  # NumPy or All
                suite.addTest(NeuralNetworkTests('test_xor_numpy_implementation'))
            if WEBGPU_AVAILABLE and model_choice in [4, 5, 6]:  # WebGPU or All
                suite.addTest(NeuralNetworkTests('test_xor_webgpu_implementation'))
        
        if test_choice in [2, 4]:  # Digit Recognition or All Tests
            if model_choice in [1, 6]:  # Pure Python or All
                suite.addTest(NeuralNetworkTests('test_digit_recognition_pure_implementation'))
            if model_choice in [2, 6]:  # Parallel or All
                suite.addTest(NeuralNetworkTests('test_digit_recognition_pure_implementation'))
            if model_choice in [3, 6]:  # NumPy or All
                suite.addTest(NeuralNetworkTests('test_digit_recognition_numpy_implementation'))
            if WEBGPU_AVAILABLE and model_choice in [4, 5, 6]:  # WebGPU or All
                suite.addTest(NeuralNetworkTests('test_digit_recognition_webgpu_implementation'))
        
        if test_choice in [3, 4]:  # Performance or All Tests
            suite.addTest(NeuralNetworkTests('test_performance_comparison'))
        
        # Run the tests
        safe_print("\nRunning tests...\n")
        runner = LoggingTestRunner(log_option=log_option, verbosity=2)
        result = runner.run(suite)
        
        # Display summary
        safe_print("\nTest Summary:")
        safe_print(f"Ran {result.testsRun} tests")
        if result.wasSuccessful():
            safe_print("All tests passed!")
        else:
            safe_print(f"Tests failed: {len(result.failures)} failures, {len(result.errors)} errors")
            
            # Log the errors
            if log_option != "none":
                logger.error("Test failures:")
                for failure in result.failures:
                    logger.error(f"{failure[0]}: {failure[1]}")
                
                logger.error("Test errors:")
                for error in result.errors:
                    logger.error(f"{error[0]}: {error[1]}")
    
    except Exception as e:
        safe_print(f"Error running tests: {str(e)}")
        logger.error(f"Error running tests: {str(e)}")

def main():
    """Main function"""
    show_welcome()
    
    # Get logging preference first
    log_option = get_log_preference()
    log_file = setup_logging(log_option)
    
    if log_file:
        safe_print(f"Logs will be saved to: {log_file}")
    
    customize = input("Do you want to customize test parameters? (y/n, default: n): ")
    if customize.lower() in ["y", "yes"]:
        model_choice = get_model_selection()
        test_choice = get_test_selection()
        n_samples, epochs, batch_size = get_test_parameters()
    else:
        model_choice = 6  # All implementations
        test_choice = 4   # All tests
        n_samples, epochs, batch_size = 1000, 100, 32
    
    # Make sure SimpleProgress uses safe_print
    import tests.compare_implementations
    tests.compare_implementations.SimpleProgress.safe_print = safe_print
    
    run_selected_tests(model_choice, test_choice, n_samples, epochs, batch_size, log_option)
    
    run_more = input("\nRun more tests? (y/n, default: n): ")
    if run_more.lower() in ["y", "yes"]:
        main()
    else:
        safe_print("Thank you for using the Neural Network test suite!")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        safe_print("\nTest execution interrupted by user.")
    except Exception as e:
        safe_print(f"\nAn unexpected error occurred: {str(e)}") 