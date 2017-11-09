import numpy as np

def unroll_params(Theta):
    
    nn_params = np.reshape(Theta[0],(1,-1)).transpose()
    for i in range(1,len(Theta)):
        nn_params = np.concatenate((nn_params,np.reshape(Theta[i], (1, -1)).transpose()))
        
    nn_params = np.ndarray.flatten(nn_params)
    
    return nn_params
        
