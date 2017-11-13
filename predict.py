import numpy as np
from sigmoid import sigmoid

def predict(Theta, X):
    # Takes as input a number of instances and the network learned variables
    # Returns a vector of the predicted labels for each one of the instances
    
    # Useful values
    m = X.shape[0]
    num_layers = len(Theta) + 1

    p = np.zeros((1, m))
    
    for i in range(m):
        a = X[i, :]
        for h in range(num_layers - 1):
            a = np.append([1], a)
            a = sigmoid(np.dot(a, Theta[h].T))
        p[0][i] = np.argmax(a)

    return p

