import numpy as np

def debugInitializeWeights(fan_out, fan_in):
    W = np.zeros((fan_out, 1 + fan_in))

    W = np.reshape(np.sin(np.arange(np.size(W))+1.0), W.shape ) / 10. 
    return W
