# Putting Neural Networks to Work: Practical Examples

Ready to see what your neural networks can actually do? This guide walks you through real-world examples that demonstrate how to apply neural.py to solve practical problems.

## How to Run the Examples

Getting started with the examples is easy:

```bash
# Run a specific example
python examples/run_examples.py tic_tac_toe

# See what examples are available
python examples/run_examples.py --list

# Interactive mode - choose from a menu
python examples/run_examples.py
```

Or if you prefer to go straight to a particular example:

```bash
python examples/tic_tac_toe/game.py
```

## Tic Tac Toe with AI: Learn by Playing

Ever wanted to create an AI that learns from experience? The Tic Tac Toe example shows you how to build an agent that gets smarter the more you play against it.

### What Makes This Example Special

- **Interactive Gameplay**: A clean, intuitive GUI built with tkinter
- **Learning in Real-Time**: Watch as the AI improves its strategy with each game
- **Reinforcement Learning**: The AI learns which moves lead to victory using the reward function:
  $$R(s,a) = \begin{cases}
  1.0 & \text{if move leads to winning} \\
  0.5 & \text{if move leads to a draw} \\
  0.0 & \text{if move leads to losing}
  \end{cases}$$
- **Smart Strategy**: Combines neural network evaluation with game-specific rules

### Jump Right In

1. Start the game:
   ```
   python examples/tic_tac_toe/game.py
   ```
2. Click any empty square to place your X
3. Watch the AI respond with O
4. Try to outsmart it - but beware, it learns from its mistakes!

### Training Your AI Champion

Want a tougher opponent? You've got options:

- **Quick Training**: Click the "Train AI (100 games)" button in the game
- **Deep Training**: Run the dedicated training script:
  ```
  python examples/tic_tac_toe/train_ai.py
  ```
- **Advanced Strategy**: Create a pre-configured smart model:
  ```
  python examples/tic_tac_toe/create_smart_model.py
  ```

### Behind the Scenes

The AI uses a neural network architecture of 9-27-9 (input, hidden, output). For those interested in the math:

1. **State Representation**: The game board is flattened into a vector $$s \in \{-1,0,1\}^9$$ where:
   - 1 represents X (player)
   - -1 represents O (AI)
   - 0 represents empty squares

2. **Decision Making**: The AI combines:
   - Neural network evaluation: $$Q(s,a) = f(W_2 \cdot \text{ReLU}(W_1 \cdot s + b_1) + b_2)$$
   - Strategic rules for critical positions
   - Exploration vs. exploitation with probability $$\epsilon$$ (exploration rate)

3. **Learning Process**: The AI updates its neural network weights using:
   - Temporal difference learning
   - Mini-batch gradient descent
   - Experience replay for better generalization

## Create Your Own Example

Inspired to build your own neural network application? Follow these steps:

1. Create a new directory: `examples/your_example_name/`
2. Add an `__init__.py` file to make it a proper package
3. Create your main script (e.g., `main.py`)
4. Add a README.md explaining what your example does

The best examples demonstrate a practical application of neural networks while being accessible to newcomers.

## Requirements

Each example may have different requirements, but all examples use the neural network implementations from the main project. Make sure you have the necessary dependencies installed. 