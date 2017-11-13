import numpy as np
from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient
from roll_params import roll_params
from unroll_params import unroll_params

def backwards(nn_weights, layers, X, y, num_labels, lambd):
    # Computes the gradient fo the neural network.
    # nn_weights: Neural network parameters (vector)
    # layers: a list with the number of units per layer.
    # X: a matrix where every row is a training example for a handwritten digit image
    # y: a vector with the labels of each instance
    # num_labels: the number of units in the output layer
    # lambd: regularization factor
    
    # Setup some useful variables
    m = X.shape[0]
    num_layers = len(layers)

    # Roll Params
    # The parameters for the neural network are "unrolled" into the vector
    # nn_params and need to be converted back into the weight matrices.
    Theta = roll_params(nn_weights, layers)
  
    Theta_grad = [np.zeros(w.shape) for w in Theta]

    yv = np.zeros((num_labels, m))
    for i in range(m):
        yv[y[i]][i] = 1

    # Implementation of the backpropagation algorithm
    
    for i in range(m):
               
        a_values, z_values = [], [] # arrays where the values of the activations are to be stored
        
        a = np.append([1], X[i, :])
        a_values.append(a)
        
        # Loop of the feedforward algorithm
        for k in range(num_layers - 1):
            z = np.dot(Theta[k], a)
            z_values.append(z)
            a = np.append([1], sigmoid(z))
            a_values.append(a)
            
        delta_layer = a[1:] - yv[:, i] # error array of the outer layer
        # np.outer to calculate the matrix product of delta_layer.T and a_values[-2]
        Theta_grad[-1] += np.outer(delta_layer, a_values[-2])/m
        
        # Descending loop
        for h in range(num_layers - 2):
            # Error of the (num_layers - 2 - h)-th hidden layer
            # The error that corresponds to the bias factors is not taken into account
            delta_layer = np.dot(Theta[-1 - h].T, delta_layer)[1:] * sigmoidGradient(z_values[-2 - h])
            # Calculation of the gradient
            Theta_grad[-2 - h] += np.outer(delta_layer, a_values[-3 - h])/m

    #Regularization
    for h in range(num_layers - 1):
        # The terms corresponding to the bias factors are not regularized
        Theta_grad[h][:, 1:] += lambd * Theta[h][:, 1:]/m
        
    # Unroll Params
    Theta_grad = unroll_params(Theta_grad)

    return Theta_grad

    
