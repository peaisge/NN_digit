import numpy as np

def randInitializeWeights(layers):

    num_of_layers = len(layers)
    epsilon = 0.12
        
    Theta = []
    for i in range(num_of_layers-1):
        
        # Initialize W randomly so that we break the symmetry while training the neural network.
        
        W = np.random.uniform(-epsilon, epsilon, (layers[i+1], layers[i] + 1))
        Theta.append(W)
                
    return Theta
            
