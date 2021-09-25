import numpy
import component.utils as utils
from scipy.special import expit  # more efficient calculation of sigmoid


class StandardLayer:
    def __init__(self, intput_size, output_size):
        self.weights = numpy.random.randn(intput_size + 1, output_size)

    def evaluate(self, inputs):
        """evaluation of multilayer (continuous) perceptron network"""

        # add one more input (column) dimension with value 1 for bias
        ones = numpy.ones((len(inputs), 1))
        currentActivation = numpy.concatenate((inputs, ones), axis=1)

        # dot product
        currentActivation = currentActivation.dot(self.weights)

        # sigmoid calculation
        return expit(currentActivation)

    def backpropagation(self, inputs, delta):
        # row vector of 1s
        ones = numpy.ones((1, 1))

        # nabla = [previous layer outputs]^T * delta
        nabla = numpy.concatenate((inputs.transpose(), ones), axis=0).dot(delta)

        # [previous layer delta] =
        # ([current layer delta] * [current layer weights excluding biases]^T)
        #      \odot [previous layer sigmoid_prime]
        delta = delta.dot(self.weights[:-1].transpose()) * utils.sigmoid_prime(inputs)

        return nabla, delta

    def update_weights(self, nabla, learning_rate):
        self.weights -= learning_rate * nabla
