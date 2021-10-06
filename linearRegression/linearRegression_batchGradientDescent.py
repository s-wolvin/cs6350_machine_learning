## Savanna Wolvin
# Created: Oct. 5th, 2021
# Edited: Sep. 10th, 2021








#%% Global Imports
import pandas as pd 
import numpy as np
import sys



#%% Variable Presets

# Data set
data_file_name = 'test'
data_file = 'concrete/' + data_file_name + '.csv'

# column labels
labels = ['Cement','Slag','Fly Ash','Water','SP','Coarse Aggr','Fine Aggr','Slump']

# learning rate
r = 1

# initial W
w = [0, 0, 0, 0, 0, 0, 0]

# number of iterations
T = 10


#%% Main

def main():
    ### Load Data
    print('Load data and attributes...')
    trainData = pd.read_csv(data_file, sep=',', header=None)
    trainData = trainData.to_numpy()
    data_lngth = np.shape(trainData)[1]
    
    for tx in range(T):  # each iteration
        batchGradient = np.zeros([0])
        
        for wi in range(np.shape(trainData)[1]):  # each w value
            xx = 0
            
            for ii in range(np.shape(trainData)[0]): # each example
                xx += -sum(   (trainData[:,data_lngth] - (w * trainData[:,data_lngth])) * (trainData[ii,data_lngth])   )    
        






#%%







#%% Main
main()


