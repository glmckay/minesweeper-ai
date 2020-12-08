import numpy
from scipy.special import expit  # more efficient calculation of sigmoid


class ConvolutionLayer:
    """
    Parameters:
        input_size: size of input_vector
        grid_width: we consider the input as a grid, this is the width of that grid
        feature_size: the feature maps are looking at n x n subgrids, this is that n
            i.e. the width and height of these square subgrids
        num_features: number of feature maps
        pool_size: the pooling layers summarise n x n subgrids of the feature maps,
            this is that n
    """

    def __init__(
        self,
        input_size,
        grid_width,
        feature_size,
        num_features,
        pool_size=2,
    ):

        self.grid_width = grid_width
        self.pool_size = pool_size
        self.feature_size = feature_size
        self.num_features = num_features

        self.grid_height, remainder = divmod(input_size, grid_width)
        assert remainder == 0

        self.pools_width, remainder = divmod(grid_width - (feature_size - 1), pool_size)
        assert remainder == 0
        self.pools_height, remainder = divmod(
            self.grid_height - (feature_size - 1), self.pool_size
        )
        assert remainder == 0

        self.feature_weights = [
            numpy.random.randn(feature_size, feature_size) for i in range(num_features)
        ]
        self.feature_biases = numpy.random.randn(feature_size)

        self.output_size = num_features * self.pools_width * self.pools_height

    def _evaluate_pool(self, input_matrix, feature, r, c):
        m = self.feature_size
        W = self.feature_weights[feature]
        b = self.feature_biases[feature]

        return expit(
            b + max(
                (input_matrix[i : i + m, j : j + m] * W).sum()
                for i in range(r, r + self.pool_size)
                for j in range(c, c + self.pool_size)
            )
        )

    def evaluate(self, inputs):
        input_matrix = numpy.array(
            list(
                inputs[i * self.grid_width : (i + 1) * self.grid_width]
                for i in range(self.grid_height)
            )
        )

        outputs = numpy.zeros(self.output_size)
        index = 0
        for i in range(0, self.pools_height, self.pool_size):
            for j in range(0, self.pools_width, self.pool_size):
                for f in range(self.num_features):
                    outputs[index] = self._evaluate_pool(input_matrix, f, i, j)
                    index += 1
        return outputs

    def backpropagation(self):
        pass

    def update_weights(self):
        pass