# MNIST Digit Recognizer

This project is my personal implementation of a neural network for handwritten digit recognition, based on the book *Neural Networks and Deep Learning* by Michael Nielsen.

## About

I am currently learning deep learning by working through this book and implementing its concepts from scratch. This project is part of my deep research and hands-on practice with neural networks, aiming to better understand how they work and learn.

## Project Structure

- `train.py`: Script to train the neural network on the MNIST dataset.
- `test_mnist.py`: Script to evaluate the trained model on MNIST test data.
- `models/`: Directory where the trained network is saved.
- `mnist_loader.py`: Helper module to load and preprocess MNIST data.
- `network.py`: Neural network implementation following Nielsen's methodology.

## Setup and Usage

1. Create and activate a Python virtual environment.
2. Install required packages:

pip install -r requirements.txt

3. Train the model:

python train.py

4. Test the trained model:

python test_mnist.py

## Notes

- This is a learning project, so the code is designed to follow the book's explanations closely.
- Feel free to explore and improve!