# helpers


def sigmoid_prime(z):
    """derivative of sigmoid function, where z = sigmoid(x)

    Doing this instead of using scipy.stats.logistic._pdf reduces the accuracy, but
        increases the speed
    """
    return z * (1 - z)