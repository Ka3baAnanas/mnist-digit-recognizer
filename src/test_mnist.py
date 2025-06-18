import numpy as np
import network  
from mnist_loader import load_data_wrapper

def main():
    training_data, validation_data, test_data = load_data_wrapper()
    net = network.Network.load("C:\\Users\\msi1\\Desktop\\Github\\mnist-digit-recognizer\\models\\trained_network.pkl")

    print("Showing 3 test images:")
    for i in range(20):
        x, y = test_data[i]
        print(f"the label is {y} ")
        prediction = np.argmax(net.feedforward(x))
        print(f"Predicted digit: {prediction}")

if __name__ == "__main__":
    main()
