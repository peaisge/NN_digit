import numpy as np
from sigmoid import sigmoid
from roll_params import roll_params


def costFunction(nn_weights, layers, X, y, num_labels, lambd):
    # Computes the cost function of the neural network.
    # nn_weights: Neural network parameters (vector)
    # layers: a list with the number of units per layer.
    # X: a matrix where every row is a training example for a handwritten digit image
    # y: a vector with the labels of each instance
    # num_labels: the number of units in the output layer
    # lambd: regularization factor
    
    # Setup some useful variables
    m = X.shape[0]
    num_layers = len(layers)


    # Unroll Params
    Theta = roll_params(nn_weights, layers)
    
    J = 0;
    
    yv = np.zeros((num_labels, m))
    for i in range(m):
        yv[y[i]][i] = 1
  

    # Cost of the neural network (feedforward)
    
    # Activation of the k-th layer for the i-th example
    def scores_layer(i, k):
        # k = 0 for the input layer
        # k = l for the l-th hidden layer
        
        x_vect = np.append([1], X[i, :]) # insert 1 at the beginning of the input image 
        
        if k == 0:
            return sigmoid(np.dot(Theta[0], x_vect))
         
        # Insert 1 at the beginning of the activation of the previous layer
        res_with_bias = np.append([1], scores_layer(i, k-1))
        return sigmoid(np.dot(Theta[k], res_with_bias))
      
    # Cost function for the i-th example          
    def cost_i(i):
        activation_layer = scores_layer(i, num_layers - 2) # output: activation of the outer layer
        y_i = yv[:, i]
        return (-y_i * np.log(activation_layer) - (1 - y_i) * np.log(1 - activation_layer)).sum()
    
    # Total cost J
    for i in range(m):
        J += cost_i(i)
    J /= m
    
    # Regularization
    coeff_reg = lambd / (2 * m)
    # Loop on the weight matrixes
    for h in range(num_layers - 1):
        sub_weights = Theta[h][:, 1:] # the terms corresponding to the bias factors are not regularized
        J += coeff_reg * (sub_weights * sub_weights).sum()
        
        
    return J

    

