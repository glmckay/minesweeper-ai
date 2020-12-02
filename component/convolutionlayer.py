import numpy


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

        self.grid_height, remainder = divmod(input_size, grid_width)
        assert remainder == 0

        self.pool_width, remainder = divmod(grid_width - (feature_size - 1), pool_size)
        assert remainder == 0
        self.pool_height, remainder = divmod(
            self.grid_height - (feature_size - 1), self.pool_size
        )
        assert remainder == 0

        self.feature_weights = [
            numpy.random.randn(feature_size + 1, feature_size)
            for i in range(num_features)
        ]

        self.output_size = num_features * self.pool_width * self.pool_height

    def evaluate(self):
        pass

    def backpropagation(self):
        pass

    def update_weights(self):
        pass