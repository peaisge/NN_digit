import numpy as np

def sigmoid(z):

    # SIGMOID returns sigmoid function evaluated at z
    g = 1 / (1 + np.exp(-z))
    return g
