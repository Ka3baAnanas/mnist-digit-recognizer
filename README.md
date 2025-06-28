# MNIST Digit Recognizer

This project is a from-scratch implementation of a neural network trained on the MNIST dataset, based on the principles presented in Michael Nielsen's book *Neural Networks and Deep Learning*. It also complements the mathematical paper I've written on the backpropagation algorithm.

## About

This repository is part of my ongoing deep research in machine learning and mathematical AI. As a student with a solid background in mathematics and a strong passion for understanding how learning systems work, I’ve been exploring backpropagation not just through implementation, but by reconstructing it mathematically and visually — step by step.

The code here directly reflects the formulas derived in my paper, including:

- δᴸ = (aᴸ − y) ⊙ σ′(zᴸ)
- δˡ = (Wˡ⁺¹)ᵗ · δˡ⁺¹ ⊙ σ′(zˡ)
- ∂C/∂wˡ = δˡ · (aˡ⁻¹)ᵗ


These are implemented clearly and directly using NumPy.

If you're reading the paper and want to see the math brought to life in code — this repo is for you.

## Project Structure

- `train.py` — trains the neural network on the MNIST dataset.
- `test_mnist.py` — evaluates the trained model on the test set.
- `models/` — stores trained network weights.
- `mnist_loader.py` — loads and preprocesses MNIST data.
- `network.py` — core neural network and backpropagation implementation.

## Setup and Usage

```bash
pip install -r requirements.txt
python train.py         # Train the model
python test_mnist.py    # Test the model
Notes
While this started as a learning project, the code has matured alongside my deeper mathematical study of backpropagation.

📄 **[Read the full paper (PDF)](backpropagation.pdf)**  
*A visual, step-by-step derivation of the backpropagation algorithm, from first principles to code.*
I welcome others who are curious about the math behind learning — feel free to explore and build upon it!
