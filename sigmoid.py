import numpy as np

def sigmoid(z):

    # SIGMOID returns sigmoid function evaluated at z
    g = 1 / (1 + np.exp(-z))

    # ============================= TODO ================================
    # Instructions: Compute sigmoid function evaluated at each value of z.

    return g
