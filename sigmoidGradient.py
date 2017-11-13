from sigmoid import sigmoid

def sigmoidGradient(z):

    # SIGMOIDGRADIENT returns the gradient of the sigmoid function evaluated at z

    sigmo = sigmoid(z)
    return sigmo * (1 - sigmo)



