import numpy as np
from read_dataset import read_dataset
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from displayData import displayData
from randInitializeWeights import randInitializeWeights
from costFunction import costFunction
from unroll_params import unroll_params
from roll_params import roll_params
import scipy.optimize as opt
from predict import predict
from backwards import backwards
from checkNNCost import checkNNCost
from checkNNGradients import checkNNGradients
from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient
from time import time
import pandas as pd



# ================================ Step 1: Loading and Visualizing Data ================================
print("\nLoading and visualizing Data ...\n")

#Reading of the dataset
# You are free to reduce the number of samples retained for training, in order to reduce the computational cost
size_training = 60000 # number of samples retained for training - to be reduced if complexity is too high
size_test     = 10000 # number of samples retained for testing

#size_validation = 1000 # number of samples retained for cross-validation (lambd & maxfun)
images_training, labels_training, images_test, labels_test = read_dataset(size_training, size_test)

# Separation of images_training:
#       images_validation for the cross-validation of the hyper-parameters (i.e. lambd and maxfun)
#       images_training for the training    
#images_validation, images_training = images_training[:size_validation], images_training[size_validation:]

# Randomly select 100 data points to display
#size_training -= size_validation
random_instances = list(range(size_training))
np.random.shuffle(random_instances)
displayData(images_training[random_instances[0:100],:])

input('Program paused. Press enter to continue!!!')

# ================================ Step 2: Setting up Neural Network Structure &  Initialize NN Parameters ================================
print("\nSetting up Neural Network Structure ...\n")

# Setup the parameters you will use for this exercise
input_layer_size   = 784 # 28x28 Input Images of Digits
num_labels         = 10 # 10 labels, from 0 to 9 (one label for each digit) 

num_of_hidden_layers = int(input('Please select the number of hidden layers: '))
print("\n")

layers = [input_layer_size]
for i in range(num_of_hidden_layers):
    layers.append(int(input('Please select the number nodes for the ' + str(i+1) + ' hidden layers: ')))
layers.append(num_labels)

input('\nProgram paused. Press enter to continue!!!')

print("\nInitializing Neural Network Parameters ...\n")

# ================================ TODO ================================
# Fill the randInitializeWeights.py in order to initialize the neural network weights. 
Theta = randInitializeWeights(layers)

# Unroll parameters
nn_weights = unroll_params(Theta)

input('\nProgram paused. Press enter to continue!!!')

# ================================ Step 3: Sigmoid  ================================================
#  Before you start implementing the neural network, you will first
#  implement the gradient for the sigmoid function. You should complete the
#  code in the sigmoidGradient.m file.
#

print("\nEvaluating sigmoid function ...\n")

g = sigmoid(np.array([1, -0.5, 0,  0.5, 1]))
print("Sigmoid evaluated at [1 -0.5 0 0.5 1]:  ")
print(g)

input('\nProgram paused. Press enter to continue!!!')

# ================================ Step 4: Sigmoid Gradient ================================
#  Before you start implementing the neural network, you will first
#  implement the gradient for the sigmoid function. You should complete the
#  code in the sigmoidGradient.m file.
#

print("\nEvaluating Sigmoid Gradient function ...\n")

g = sigmoidGradient(np.array([1, -0.5, 0,  0.5, 1]))
print("Sigmoid Gradient evaluated at [1 -0.5 0 0.5 1]:  ")
print(g)

input('\nProgram paused. Press enter to continue!!!')

# ================================ Step 5: Implement Feedforward (Cost Function) ================================

print("\nChecking Cost Function without Regularization (Feedforward) ...\n")

lambd = 0.0
checkNNCost(lambd)

print('This value should be about 2.09680198349')

input('\nProgram paused. Press enter to continue!!!')

# ================================ Step 6: Implement Feedforward with Regularization  ================================

print("\nChecking Cost Function with Reguralization ... \n")

lambd = 3.0
checkNNCost(lambd)

print('This value should be about 2.1433733821')


input('\nProgram paused. Press enter to continue!!!')

# ================================ Step 7: Implement Backpropagation  ================================

print("\nChecking Backpropagation without Regularization ...\n")

lambd = 0.0
checkNNGradients(lambd)
input('\nProgram paused. Press enter to continue!!!')


# ================================ Step 8: Implement Backpropagation with Regularization ================================

print("\nChecking Backpropagation with Regularization ...\n")

lambd = 3.0
checkNNGradients(lambd)

input('\nProgram paused. Press enter to continue!!!')

# ================================ Step 9: Training Neural Networks & Prediction ================================
print("\nTraining Neural Network... \n")

#  You should also try different values of the regularization factor
lambd = 5
maxfun = 200
start = time() # start of the timer

res = opt.fmin_l_bfgs_b(costFunction, nn_weights, fprime = backwards, args = (layers, images_training, labels_training, num_labels, lambd), maxfun = maxfun, factr = 1., disp = True)
Theta = roll_params(res[0], layers)

# input('\nProgram paused. Press enter to continue!!!')

print("\nTesting Neural Network... \n")

pred  = predict(Theta, images_test)
end = time() # end of the timer
time_complexity = end - start
print('\nSize training =', size_training)
print('Lambda =', lambd)
print('Maxfun =', maxfun)
print('Time:', time_complexity, 'seconds')
print('Accuracy =', np.mean(labels_test == pred) * 100, '%')


# The function I used for cross-validation
def cross_validation(lambd_values = [0.1], maxfun_values = [200]):

    """Function that trains the neural network and then tests its accuracy
    Parameters : lambd, which measures the coefficient of the regularization
                 maxfun, which counts the number of iterations of the backpropagation
    """
    
    n_lambd, n_maxfun = len(lambd_values), len(maxfun_values)
    
    # Creation of the DataFrame where the results are to be stored
    df_results = pd.DataFrame(index = range(n_lambd * n_maxfun))
    df_results['Maxfun'], df_results['Lambd'] = list(maxfun_values) * n_lambd, list(lambd_values) * n_maxfun
    df_results['Hidden layers'] = num_of_hidden_layers
    nodes_avg = np.mean(layers[1:-1])
    df_results['Nodes per hidden layer (avg)'] = nodes_avg
    accuracy_col = []
    
    for lambd in lambd_values:
        
        for maxfun in maxfun_values:
            
            start = time() # start of the timer
            
            res = opt.fmin_l_bfgs_b(costFunction, nn_weights, fprime = backwards, args = (layers, images_validation, labels_training, num_labels, lambd), maxfun = maxfun, factr = 1., disp = True)
            Theta = roll_params(res[0], layers)
            
            # input('\nProgram paused. Press enter to continue!!!')
            
            # print("\nTesting Neural Network... \n")
            
            pred  = predict(Theta, images_test)
            end = time() # end of the timer
            accuracy = np.mean(labels_test == pred) * 100
            print('\nLambda =', lambd)
            print('Maxfun =', maxfun)
            time_complexity = end - start
            print('Time:', time_complexity, 'seconds')
            print('Accuracy =', accuracy, '%')
            
            # Modification of the 'Accuracy' column
            accuracy_col.append(accuracy)
    
    # Accuracy values stored into the dataframe
    df_results['Accuracy'] = accuracy_col
    
    return df_results

# To do a cross-validation, choose the parameters lambd_values and maxfun_values
# then, launch the function test_NN to train and test the neural network
# For example: df1x5 = cross_validation([0.1, 1], [50, 200])


