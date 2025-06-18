import mnist_loader
import network

# Load the MNIST data
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# Create the network and train it
net = network.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
net.save("../models/trained_network.pkl")