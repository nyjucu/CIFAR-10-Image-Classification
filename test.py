from tensorflow.keras import datasets
import numpy as np

(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

class_names = ["Plane", "Car", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]
num_per_classes = np.zeros(10)

for i in range(50_000):
    num_per_classes[y_train[i]] += 1

for i in range(10):
    print(class_names[i], ": ", num_per_classes[i])
