import numpy
from scipy.special import expit # more efficient calculation of sigmoid

# helpers
def sigmoid_prime(z):
	"""derivative of sigmoid function, where z = sigmoid(x)"""
	# doing this instead of using scipy.stats.logistic._pdf reduces the accuracy, for greater speed
	return z * (1 - z)

# Network
class Network:
	def __init__(self, sizes, cost): #, smallWeightInit=True):
		"""initialise weights and biases for the feedforward neural network, along with specified cost"""
		# assert sizes[0] == 784 # assert that the first layer has 784 = 28x28 nodes, one for each pixel
		# assert sizes[-1] == 10 # assert that the last layer has 10 nodes, one for each digit

		# record number of layers
		self.layerNum = len(sizes)

		# initialise weights and biases
		self.weights = [numpy.random.randn(input_dim+1, output_dim) for output_dim,input_dim in zip(sizes[1:], sizes[:-1])]
		# if using small weights, modify weights
		# if smallWeightInit: 
		# 	for W in self.weights: W[:-1,:] *= 1 / numpy.sqrt(len(W)-1) 

		# record cost function
		self.cost = cost

	def evaluate(self, inputs):
		"""evaluation of multilayer (continuous) perceptron network"""
		# output of each layer
		outputs = [inputs]
		# column vector of 1s
		ones = numpy.ones((len(inputs), 1))

		for M in self.weights:
			# add one more input (column) dimension with value 1 for bias
			currentActivation = numpy.concatenate((outputs[-1], ones), axis=1)

			# dot product
			currentActivation = currentActivation.dot(M)

			# sigmoid calculation
			outputs += [expit(currentActivation)]

		return outputs

	def backpropagation(self, inputs, labels):
		"""backpropogation algorithm, returning the gradient of the cost function"""
		outputs = self.evaluate(inputs)

		# row vector of 1s
		ones = numpy.ones((1,len(inputs)))

		# initialise variables
		# # delta, initialised with (output - label) \odot modifier (see costfunctions.Cost class)
		delta = self.cost.deltaInit(outputs[-1], labels)
		# nabla
		nabla = []

		# iterating backwards
		for i in range(1, self.layerNum):
			# nabla = [previous layer outputs]^T * delta
			nabla = [numpy.concatenate((outputs[-i-1].transpose(), ones), axis=0).dot(delta)] + nabla

			# [previous layer delta] = 
			# ([current layer delta] * [current layer weights excluding biases]^T) \odot [previous layer sigmoid_prime]
			delta = delta.dot(self.weights[-i][:-1].transpose()) * sigmoid_prime(outputs[-i-1])

		return nabla

	def updateWeights(self, inputs, labels, total, learningRate=1, regularisation=0):
		"""updates the weights of the network by gradient descent"""
		assert len(inputs) == len(labels)
		n = len(inputs)

		# backpropagate
		nabla = self.backpropagation(inputs, labels)

		# calculated new weight
		for W,V in zip(self.weights, nabla):
			# biases are ignored when tweaking according to regularisation
			if regularisation: W[:-1,:] *= 1 - ((learningRate * regularisation) / total)
			# new weight = regularised old weight - \frac{learningRate}{n} nabla
			W -= (learningRate  * V) / n