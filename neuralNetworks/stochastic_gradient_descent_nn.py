## Savanna Wolvin
# Created: Nov. 29th, 2021
# Edited: Dec. 1st, 2021
    
# SUMMARY

# INPUT

# OUTPUT


#%% Global Inputs
import numpy as np
import pandas as pd
import random as rd




#%% Variable Presets
# Data set
data_file_train = 'train'
data_file_test = 'test'

data_folder = 'bank-note/'

max_epoch = 50

lr_0 = 0.1

width = 3
depth = 2




#%% Load Data
    
trainData = pd.read_csv(data_folder + data_file_train + '.csv', sep=',', header=None)
trainData = trainData.to_numpy()
trainData[np.where(trainData[:,4] == 0), 4] = -1
X = trainData[:,range(4)]
Y = trainData[:,4]

testData = pd.read_csv(data_folder + data_file_test + '.csv', sep=',', header=None)
testData = testData.to_numpy()
testData[np.where(testData[:,4] == 0), 4] = -1




#%% create an array of sizes

num_nodes = [np.shape(X)[1]]

for layerx in range(depth):
    num_nodes.append(width-1)

num_nodes.append(1)




#%% Learning Rate

def learnRate(t):
    lr = lr_0 / ( 1 + ( ( lr_0 / depth ) * t ) )
    return lr




#%% Sigmoid Function

def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s




#%%  Initialize the layers

def initialize_layers():
    w_layer = {}
    b_layer = {}

    for dx in range(depth+1):
        w_layer[dx] = np.random.normal(0, 1, (num_nodes[dx+1], num_nodes[dx]))
        b_layer[dx] = np.ones((num_nodes[dx+1], 1))
    
    return w_layer, b_layer




#%% Forward propagation

def forward_prop(X_rand_ex):
    z_layer = {} 
    
    z = w_layer[0].dot(X_rand_ex) + b_layer[0]
    Z1 = sigmoid(z)
        
    for dx in range(1,depth+1):
        z_layer[dx] = np.expand_dims(np.append(1, Z1), axis=1)
        z = w_layer[dx].dot(Z1) + b_layer[dx]
        Z1 = sigmoid(z)
    
    return z, z_layer




#%% Backward Propagation

def backward_prop(y_star, y, z_layer, X_rand_ex):
    gradient_loss = {}
    z_layer[0] = X_rand_ex
    
    cashe_array = (y - y_star) # cashe this value
    
    # output layer
    dL_output = cashe_array * z_layer[depth].T
    gradient_loss[depth] = dL_output
    
    # z = 2
    cashe_array = cashe_array * w_layer[2] # dL/dy * dy/dz
    cashe_array_z_1_z = cashe_array * z_layer[2][1:].T * (1 - z_layer[2][1:].T)
    gradient_loss[2-1] = np.dot(z_layer[2-1], cashe_array_z_1_z).T
    
    # z = 1
    cashe_array = cashe_array.T * w_layer[1] # dL/dy * dy/dz
    cashe_array_z_1_z = cashe_array * z_layer[1][1:] * (1 - z_layer[1][1:])
    gradient_loss[1-1] = np.expand_dims(z_layer[1-1], axis=1).dot(cashe_array_z_1_z).T
    
    # # initial variables
    # gradient_loss_1 = {}
    # cashe_array = np.array(y - y_star) # cashe this value
    
    # # Calculate output layer
    # gradient_loss_1[depth] = cashe_array * z_layer[depth]

    # # loop through hidden layers
    # for lx in range(depth, 0, -1):
    #     print(lx)
    #     cashe_array = cashe_array.T * w_layer[lx] # dL/dy * dy/dz
    #     cashe_array_z_1_z = cashe_array * z_layer[lx][1:] * (z_layer[lx][1:] - 1)
    #     z_layer_x = np.tile(np.expand_dims(z_layer[lx-1], axis=1), np.shape(cashe_array_z_1_z)[0]) 
        
    #     gradient_loss_1[lx-1] = z_layer_x.dot(cashe_array_z_1_z).T
    
    return gradient_loss




#%% Main of the Neural Network

w_layer, b_layer = initialize_layers() # create data arrays to hold the weight 
                                       # vectors and the bias values

# Number of times to loop through the entire dataset, change to stop at a 
# certain value of loss
for t in range(max_epoch):
    # shuffle dataset
    rand_idx = rd.sample(range(0, np.shape(X)[0]), np.shape(X)[0])
    X_rand = X[rand_idx, :]
    Y_rand = Y[rand_idx]
    
    learning_rate = learnRate(t) # learing rate changes every epoch
    
    # loop through each example of the dataset as if it's its own dataset
    for ex in range(np.shape(X_rand)[0]):
        X_rand_ex = np.expand_dims(X_rand[ex,:], axis=1)
        
        # Forward Propagation
        y, z_layer = forward_prop(X_rand_ex)
        
        # Backward Propagation
        gradientLoss = backward_prop(Y_rand[ex], y, z_layer, X_rand_ex)
        
        # edit wight vector by the back propagation
        w_layer = w_layer - learning_rate * gradientLoss
        b_layer = b_layer - learning_rate * gradientLoss
        
        















































