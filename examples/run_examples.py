#!/usr/bin/env python
"""
neuralpy example cli
"""

import os
import sys
import argparse

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def list_examples():
    """List all available examples"""
    examples = []
    
    # Look for subdirectories in the examples folder
    for item in os.listdir(os.path.dirname(__file__)):
        if os.path.isdir(os.path.join(os.path.dirname(__file__), item)) and not item.startswith('__'):
            # Check if there's a main file or game.py in the directory
            if os.path.exists(os.path.join(os.path.dirname(__file__), item, 'game.py')):
                examples.append((item, 'game.py'))
            elif os.path.exists(os.path.join(os.path.dirname(__file__), item, 'main.py')):
                examples.append((item, 'main.py'))
    
    return examples

def run_example(example_name):
    """Run the specified example"""
    examples = list_examples()
    
    # Find the example
    for name, script in examples:
        if name.lower() == example_name.lower():
            example_path = os.path.join(os.path.dirname(__file__), name, script)
            print(f"Running example: {name}")
            
            # Execute the example
            try:
                # Change to the directory of the example
                original_dir = os.getcwd()
                example_dir = os.path.join(os.path.dirname(__file__), name)
                os.chdir(example_dir)
                
                # Run the example script
                if sys.version_info >= (3, 0):
                    # Python 3+
                    # Define __file__ in the execution environment
                    script_path = os.path.join(example_dir, script)
                    exec(open(script).read(), {"__name__": "__main__", "__file__": script_path})
                else:
                    # Python 2
                    execfile(script)
                    
                # Change back to original directory
                os.chdir(original_dir)
                return True
            except Exception as e:
                print(f"Error running example: {e}")
                # Make sure to change back to original directory even if there's an error
                if 'original_dir' in locals():
                    os.chdir(original_dir)
                return False
    
    print(f"Example '{example_name}' not found")
    return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Run examples from the NeuralPy project')
    
    # Add arguments
    parser.add_argument('example', nargs='?', help='Name of the example to run')
    parser.add_argument('--list', '-l', action='store_true', help='List all available examples')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.list:
        examples = list_examples()
        print("\nAvailable examples:")
        for name, script in examples:
            print(f"  - {name}")
        print("\nRun an example with: python examples/run_examples.py <example_name>")
        return 0
    
    if not args.example:
        examples = list_examples()
        if not examples:
            print("No examples found")
            return 1
        
        print("\nAvailable examples:")
        for i, (name, _) in enumerate(examples, 1):
            print(f"{i}. {name}")
        
        try:
            choice = input("\nSelect an example to run (or 'q' to quit): ")
            if choice.lower() == 'q':
                return 0
            
            choice = int(choice)
            if 1 <= choice <= len(examples):
                example_name = examples[choice-1][0]
                run_example(example_name)
            else:
                print("Invalid choice")
                return 1
        except (ValueError, IndexError):
            print("Invalid choice")
            return 1
    else:
        if run_example(args.example):
            return 0
        else:
            return 1

if __name__ == "__main__":
    sys.exit(main()) 