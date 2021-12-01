## Savanna Wolvin
# Created: Nov. 29th, 2021
# Edited: 
    
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

width = 5
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



#%% Sigmoid Function

def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s


#%%  Initialize the layers

def initialize_layers(initial_width):
    w_layer = {}

    for dx in range(depth):
        w_layer[dx] = np.append(np.array(1), np.random.normal(0, 1, (1, width-1)))
    
    return w_layer




#%% Forward propagation

def forward_prop(z_layer, X):
    for dx in range(depth):
        
    
    
    
    
    

weighted_vector = np.ones((np.shape(trainData)[1]-1))

for t in range(max_epoch):
    # shuffle dataset
    rand_idx = rd.sample(range(0, np.shape(trainData)[0]), np.shape(trainData)[0])
    trainDataX = trainData[rand_idx, :]
    
    # for ex in range(np.shape(trainDataX)[0]):
        


















