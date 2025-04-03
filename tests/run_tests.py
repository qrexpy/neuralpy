import os
import sys
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tests.compare_implementations import (
    compare_implementations,
    NeuralNetworkTests,
    unittest
)

console = Console()

def show_welcome():
    """Display welcome message and project info"""
    console.print(Panel.fit(
        "[bold blue]Neural.py Test Suite[/bold blue]\n\n"
        "Select models and tests to run",
        title="Welcome",
        border_style="blue"
    ))

def get_model_selection():
    """Get user selection for which models to test"""
    console.print("\n[bold]Available Models:[/bold]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Option", style="cyan")
    table.add_column("Model", style="green")
    table.add_column("Description", style="yellow")
    
    table.add_row("1", "Pure Python", "Educational implementation from scratch")
    table.add_row("2", "NumPy", "Optimized implementation using NumPy")
    table.add_row("3", "Both", "Compare both implementations")
    
    console.print(table)
    
    while True:
        choice = Prompt.ask(
            "\nSelect models to test",
            choices=["1", "2", "3"],
            default="3"
        )
        if choice == "1":
            return "pure"
        elif choice == "2":
            return "numpy"
        else:
            return "both"

def get_test_selection():
    """Get user selection for which tests to run"""
    console.print("\n[bold]Available Tests:[/bold]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Option", style="cyan")
    table.add_column("Test", style="green")
    table.add_column("Description", style="yellow")
    
    table.add_row("1", "XOR", "Test XOR problem solving")
    table.add_row("2", "Digit Recognition", "Test digit recognition (0 and 1)")
    table.add_row("3", "Both", "Run all tests")
    
    console.print(table)
    
    while True:
        choice = Prompt.ask(
            "\nSelect tests to run",
            choices=["1", "2", "3"],
            default="3"
        )
        if choice == "1":
            return "xor"
        elif choice == "2":
            return "digit"
        else:
            return "both"

def get_test_parameters():
    """Get test parameters from user"""
    console.print("\n[bold]Test Parameters:[/bold]")
    
    n_samples = int(Prompt.ask(
        "Number of samples",
        default="1000"
    ))
    
    epochs = int(Prompt.ask(
        "Number of epochs",
        default="1000"
    ))
    
    batch_size = int(Prompt.ask(
        "Batch size",
        default="32"
    ))
    
    return n_samples, epochs, batch_size

def run_selected_tests(model_choice, test_choice, n_samples, epochs, batch_size):
    """Run the selected tests"""
    console.print("\n[bold]Running Tests...[/bold]")
    
    # Run unit tests if selected
    if model_choice in ["pure", "both"]:
        if test_choice in ["xor", "both"]:
            console.print("\n[cyan]Running XOR tests with Pure Python implementation...[/cyan]")
            test_suite = unittest.TestSuite()
            test_suite.addTest(NeuralNetworkTests('test_xor_pure_implementation'))
            unittest.TextTestRunner().run(test_suite)
            
        if test_choice in ["digit", "both"]:
            console.print("\n[cyan]Running Digit Recognition tests with Pure Python implementation...[/cyan]")
            test_suite = unittest.TestSuite()
            test_suite.addTest(NeuralNetworkTests('test_digit_recognition_pure_implementation'))
            unittest.TextTestRunner().run(test_suite)
    
    if model_choice in ["numpy", "both"]:
        if test_choice in ["xor", "both"]:
            console.print("\n[cyan]Running XOR tests with NumPy implementation...[/cyan]")
            test_suite = unittest.TestSuite()
            test_suite.addTest(NeuralNetworkTests('test_xor_numpy_implementation'))
            unittest.TextTestRunner().run(test_suite)
            
        if test_choice in ["digit", "both"]:
            console.print("\n[cyan]Running Digit Recognition tests with NumPy implementation...[/cyan]")
            test_suite = unittest.TestSuite()
            test_suite.addTest(NeuralNetworkTests('test_digit_recognition_numpy_implementation'))
            unittest.TextTestRunner().run(test_suite)
    
    # Run performance comparison if both models are selected
    if model_choice == "both":
        console.print("\n[bold]Running Performance Comparison...[/bold]")
        
        if test_choice in ["xor", "both"]:
            console.print("\n[green]Comparing XOR implementations...[/green]")
            compare_implementations(
                task='xor',
                n_samples=n_samples,
                epochs=epochs,
                batch_size=batch_size
            )
        
        if test_choice in ["digit", "both"]:
            console.print("\n[green]Comparing Digit Recognition implementations...[/green]")
            compare_implementations(
                task='digit',
                n_samples=n_samples,
                epochs=epochs,
                batch_size=batch_size
            )

def main():
    """Main function to run the test suite"""
    show_welcome()
    
    model_choice = get_model_selection()
    test_choice = get_test_selection()
    n_samples, epochs, batch_size = get_test_parameters()
    
    run_selected_tests(model_choice, test_choice, n_samples, epochs, batch_size)
    
    console.print("\n[bold green]Tests completed![/bold green]")

if __name__ == "__main__":
    main() 