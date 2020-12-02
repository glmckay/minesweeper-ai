class Network:
    def __init__(self, layers):
        self.layers = layers

    def evaluate(self, inputs):
        """evaluation of multilayer (continuous) perceptron network"""
        # output of each layer
        outputs = [inputs]

        for layer in self.layers:
            outputs.append(layer.evaluate(outputs[-1]))

        return outputs

    def backpropagation(self, outputs, delta):
        # initialise variables
        # delta, initialised with (output - label) \odot modifier
        #   (see costfunctions.Cost class)
        # delta = self.cost.deltaInit(outputs[-1], labels)

        nabla = []
        for layer, inputs in reversed(zip(self.layers, outputs)):
            nab, delta = layer.backpropagation(inputs, delta)
            nabla.append(nab)

        nabla.reverse()
        return nabla

    def update_weights(self, outputs, delta, learning_rate=1):
        # backpropagate
        nabla = self.backpropagation(outputs, delta)

        # calculated new weight
        for layer, nab in zip(self.layers, nabla):
            layer.update_weights(nab, learning_rate)
