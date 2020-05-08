import numpy as np


# sigmoid function to normalize inputs
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# sigmoid derivatives to adjust synaptic weights
def sigmoid_derivative(x):
    return x * (1 - x)


training_inputs = np.array([[0.2, 0.7, 0.4, 0.8546, 0.1, 0.7, 0.2, 0.67],
                            [0.2, 0.6, 0.8, 0.7229, 0.2, 0.1, 0.4, 0.57],
                            [0.2, 0.6, 0.5, 1.0147, 0.0, 0.5, 0.1, 0.57],
                            [0.9, 0.6, 0.3, 1.4306, 0.0, 0, 0.1, 0.7],
                            [0.3, 0.5, 1.0, 0.6784, 0.3, 1.5, 0.1, 0.14],
                            [0.3, 0.2, 1.0, 0.5175, 0.1, 0.8, 0.1, 0.58],
                            [0.9, 0.3, 0.9, 1.9556, 0.0, 0.1, 0, 0.67],
                            [0.7, 0.4, 0.4, 1.4821, 0.3, 0.9, 0.1, 0.76],
                            [1.1, 0.4, 0.5, 2.0904, 0, 0.2, 0.1, 0.86],
                            [0, 0.5, 1.4, 0.8221, 0.5, 2.0, 0.2, 0.08],
                            [0.2, 0.7, 0.4, 1.5747, 0.1, 1.0, 0.1, 0.68],
                            [0.9, 0.1, 1.2, 2.2201, 0.2, 0.2, 0.5, 0.68],
                            [1.3, 0.3, 0.3, 1.8488, 0.1, 0.9, 0, 0.66],
                            [0.8, 0.7, 1.3, 2.5573, 0.2, 1.2, 0.3, 0.77],
                            [0.5, 0.8, 1.4, 1.1932, 0.2, 2.3, 0.5, 0.06],
                            [0.1, 0.6, 0.4, 1.4068, 0, 0.9, 0, 0.59],
                            [1.1, 0.8, 0.5, 1.9464, 0, 0.3, 0, 0.44],
                            [0.2, 0.9, 0.5, 1.2716, 0, 0.8, 0.1, 0.74],
                            [0.5, 0.7, 1.1, 2.4793, 0.2, 1.1, 0.3, 0.75],
                            [0.7, 0.7, 0.5, 1.1014, 0.6, 2.3, 1.4, 0.15],
                            [2.3, 0.4, 1.4, 4.2012, 0.2, 0.2, 0.4, 0.66],
                            [0.8, 0.2, 0.5, 2.1200, 0.1, 0.7, 0.2, 0.91],
                            [1.0, 0.7, 0.8, 3.6737, 0.8, 1.4, 0.2, 0.64]])

training_outputs = np.array([[2, 2, 1, 2, 2, 2, 3, 3, 3, 2, 2, 4, 2, 3, 2, 1, 1, 1,
                              2, 3, 4, 4, 4]]).T

# seed random numbers to make calculation
np.random.seed(1)

# initialize weights randomly with mean 0 to create weight matrix, synaptic weights
synaptic_weights = 2 * np.random.random((8,1)) - 1

print('Random starting synaptic weights: ')
print(synaptic_weights)

# Iterate 10,000 times
for iteration in range(10000):

    # Define input layer
    input_layer = training_inputs
    # Normalize the product of the input layer with the synaptic weights
    outputs = sigmoid(np.dot(input_layer, synaptic_weights))

    # how much did we miss?
    error = training_outputs - outputs

    # multiply how much we missed by the
    # slope of the sigmoid at the values in outputs
    adjustments = error * sigmoid_derivative(outputs)

    # update weights
    synaptic_weights += np.dot(input_layer.T, adjustments)

print('Synaptic weights after training: ')
print(synaptic_weights)

print("Output After Training:")
print(outputs)
