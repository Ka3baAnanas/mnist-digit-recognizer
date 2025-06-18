import numpy as np
from PIL import Image
import network  
import sys
from mnist_loader import load_data_wrapper

def load_image(filename):
    # Open, convert to grayscale, resize and invert
    img = Image.open(filename).convert('L').resize((28, 28))
    img = 255 - np.array(img)  # Invert: black digit on white background
    img = img / 255.0          # Normalize to [0,1]
    img = img.reshape((784, 1))
    return img

def main():
    '''if len(sys.argv) != 2:
        print("Usage: python predict_digit.py path_to_image.png")
        return

    image_path = sys.argv[1]
    x = load_image(image_path)'''
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
