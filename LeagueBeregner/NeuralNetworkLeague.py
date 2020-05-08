import numpy as np


class NeuralNetwork():

    def __init__(self):
        # Seed the random number generator
        np.random.seed(1)

        # Set synaptic weights to a 3x1 matrix,
        # with values from -1 to 1 and mean 0
        self.synaptic_weights = 2 * np.random.random((8, 1)) - 1

    def sigmoid(self, x):
        """
        Takes in weighted sum of the inputs and normalizes
        them through between 0 and 1 through a sigmoid function
        """
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """
        The derivative of the sigmoid function used to
        calculate necessary weight adjustments
        """
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):
        """
        We train the model through trial and error, adjusting the
        synaptic weights each time to get a better result
        """
        for iteration in range(training_iterations):
            # Pass training set through the neural network
            output = self.think(training_inputs)

            # Calculate the error rate
            error = training_outputs - output

            # Multiply error by input and gradient of the sigmoid function
            # Less confident weights are adjusted more through the nature of the function
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))

            # Adjust synaptic weights
            self.synaptic_weights += adjustments

    def think(self, inputs):


        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output


if __name__ == "__main__":
    # Initialize the single neuron neural network
    neural_network = NeuralNetwork()

    print("Random starting synaptic weights: ")
    print(neural_network.synaptic_weights)

    # The training set, with 4 examples consisting of 3
    # input values and 1 output value
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

    # Train the neural network
    neural_network.train(training_inputs, training_outputs, 10000)

    print("Synaptic weights after training: ")
    print(neural_network.synaptic_weights)

    A = str(input("Input 1 Antal kills: "))
    B = str(input("Input 2 Antal Deaths: "))
    C = str(input("Input 3 Antal Assist: "))
    D = str(input("Input 4 Antal Damage: "))
    E = str(input("Input 5 Antal Pink wards placeret: "))
    F = str(input("Input 6 Antal Normal wards placeret: "))
    G = str(input("Input 7 Antal Wards destroyed: "))
    H = str(input("Input 8 Antal Creep score: "))

    print("New situation: input data = ", A, B, C, D, E, F, G, H)
    print("Output data: ")
    print(neural_network.think(np.array([A, B, C, D, E, F, G, H])))