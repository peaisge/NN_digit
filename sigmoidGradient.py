from sigmoid import sigmoid

def sigmoidGradient(z):

    # SIGMOIDGRADIENT returns the gradient of the sigmoid function evaluated at z

    sigmo = sigmoid(z)
    # Compute the gradient of the sigmoid function evaluated at
    # each value of z.
    g = sigmo * (1 - sigmo)
    
    return g




