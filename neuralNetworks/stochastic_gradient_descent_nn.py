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
import matplotlib.pyplot as plt




#%% Variable Presets
# Data set
data_file_train = 'train'
data_file_test = 'test'

data_folder = 'bank-note/'

max_epoch = 100

lr_0 = 0.000001

width = 10
depth = 4




#%% Load Data
    
trainData = pd.read_csv(data_folder + data_file_train + '.csv', sep=',', header=None)
trainData = trainData.to_numpy()
trainData[np.where(trainData[:,4] == 0), 4] = -1
X = trainData[:,range(4)]
Y = trainData[:,4]

testData = pd.read_csv(data_folder + data_file_test + '.csv', sep=',', header=None)
testData = testData.to_numpy()
testData[np.where(testData[:,4] == 0), 4] = -1
X_test = testData[:,range(4)]
Y_test = testData[:,4]




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

def forward_prop(X_rand_ex, z_layer):    
    z = w_layer[0].dot(X_rand_ex) + b_layer[0]
    Z1 = sigmoid(z)
        
    for dx in range(1,depth+1):
        z_layer[dx] = np.expand_dims(np.append(1, Z1), axis=1)
        z = w_layer[dx].dot(Z1) + b_layer[dx]
        Z1 = sigmoid(z)
    
    return z[0], z_layer




#%% Backward Propagation

def backward_prop(y_star, y, z_layer, X_rand_ex):
    
    # initial variables
    gradient_loss = {}
    # z_layer[0] = np.asarray([X_rand_ex]).T
    z_layer[0] =  np.concatenate((np.asarray([[1]]), X_rand_ex))
    
    cashe_array = np.asarray([[float(y - y_star)]]) # cashe this value
    
    # output layer
    dL_output = cashe_array * z_layer[depth].T
    gradient_loss[depth] = dL_output
    
    
    for lx in range(depth, 0, -1):
        # cashe_array = np.multiply(np.tile(cashe_array, (np.shape(w_layer[lx])[0],1)), w_layer[lx])
        cashe_array = cashe_array.T * w_layer[lx]
        
        cashe_array_z_1_z = cashe_array * z_layer[lx][1:].T * (z_layer[lx][1:].T - 1)
        
        # for z_value in z_layer[lx-1]:
        #     np.transpose(np.sum(z_value*cashe_array_z_1_z, axis=0))
        
        multiplied_values = [np.transpose(np.sum(z_value*cashe_array_z_1_z, axis=0)) for z_value in z_layer[lx-1]]
        
        gradient_loss[lx-1] = np.column_stack(multiplied_values) 
        
    
    return gradient_loss




#%% Train Neural Network

w_layer, b_layer = initialize_layers() # create data arrays to hold the weight 
                                        # vectors and the bias values

# Number of times to loop through the entire dataset, change to stop at a 
# certain value of loss
L = np.zeros((max_epoch,1))
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
        z_layer = {} 
        y, z_layer = forward_prop(X_rand_ex, z_layer)
        
        # Backward Propagation
        gradientLoss = backward_prop(Y_rand[ex], y, z_layer, X_rand_ex)
        
        # edit wight vector by the back propagation
        for layer_x in range(depth+1):
            w_layer[layer_x] = w_layer[layer_x] - learning_rate * gradientLoss[layer_x][:,1:]
            b_layer[layer_x] = b_layer[layer_x] - learning_rate * np.expand_dims(gradientLoss[layer_x][:,0], axis=1)
            
    # create prediction
    y_predict = np.zeros(np.shape(Y)[0])
    for ex in range(np.shape(X)[0]):
        X_ex = np.expand_dims(X[ex,:], axis=1)
        
        # Forward Propagation
        z_layer = {}
        y, _ = forward_prop(X_ex, z_layer)
        y_predict[ex] = y
        
    # Calculate Loss
    L[t,0] = np.sum( (1/2) * (y_predict - Y)**2 )
    # print("Prediction Loss = " + str(L[t,0]))
    
    
   

#%% Train Neural Network

# w_layer, b_layer = initialize_layers() # create data arrays to hold the weight 
#                                        # vectors and the bias values

# # Number of times to loop through the entire dataset, change to stop at a 
# # certain value of loss
# diff_loss = 0
# count = 0
# L = []
# for t in range(max_epoch):
#     # shuffle dataset
#     rand_idx = rd.sample(range(0, np.shape(X)[0]), np.shape(X)[0])
#     X_rand = X[rand_idx, :]
#     Y_rand = Y[rand_idx]
    
#     learning_rate = learnRate(count) # learing rate changes every epoch
    
#     # loop through each example of the dataset as if it's its own dataset
#     for ex in range(np.shape(X_rand)[0]):
#         X_rand_ex = np.expand_dims(X_rand[ex,:], axis=1)
        
#         # Forward Propagation
#         z_layer = {} 
#         y, z_layer = forward_prop(X_rand_ex, z_layer)
        
#         # Backward Propagation
#         gradientLoss = backward_prop(Y_rand[ex], y, z_layer, X_rand_ex)
        
#         # edit wight vector by the back propagation
#         for layer_x in range(depth+1):
#             w_layer[layer_x] = w_layer[layer_x] - learning_rate * gradientLoss[layer_x][:,1:]
#             b_layer[layer_x] = b_layer[layer_x] - learning_rate * np.expand_dims(gradientLoss[layer_x][:,0], axis=1)
            
#     # create prediction
#     y_predict = np.zeros(np.shape(Y)[0])
#     for ex in range(np.shape(X)[0]):
#         X_ex = np.expand_dims(X[ex,:], axis=1)
        
#         # Forward Propagation
#         z_layer = {}
#         y, _ = forward_prop(X_ex, z_layer)
#         y_predict[ex] = y
        
#     # Calculate Loss
#     L.append(np.sum( (1/2) * (y_predict - Y)**2 ))
    
#     count += 1
    
    
    
    

#%% Plot Prediction Loss

plt.plot(L)
plt.ylabel('Prediction Loss')
plt.xlabel('Iteration')
plt.title('Prediction Loss as Iteration Increases')
plt.show()




#%% Calculate Training and Test Error

training_error = np.sum(np.sign(y_predict) != Y) / np.shape(Y)[0]
print('Training Error: ' + str(training_error))

# create prediction
test_predict = np.zeros(np.shape(Y_test)[0])
for ex in range(np.shape(X_test)[0]):
    X_test_ex = np.expand_dims(X_test[ex,:], axis=1)
    
    # Forward Propagation
    z_layer = {}
    y, _ = forward_prop(X_test_ex, z_layer)
    test_predict[ex] = y

test_error = np.sum(np.sign(test_predict) != Y_test) / np.shape(Y_test)[0]
print('Test Error: ' + str(test_error))













































