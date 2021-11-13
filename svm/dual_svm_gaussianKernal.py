## Savanna Wolvin
# Created: Nov. 8th, 2021
# Edited: 
    
# SUMMARY
# Now let us implement SVM in the dual domain. We use the same dataset, 
# “bank-note.zip”. You can utilize existing constrained optimization libraries.
# For Python, we recommend using “scipy.optimize.minimize”, and you can learn 
# how to use this API from the document at 
# https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.optimize.minimize.html. 
# We recommend using SLSQP to incorporate the equality constraints.

# INPUT 


# OUTPUT



#%% Global Imports
import pandas as pd 
import numpy as np
import scipy.optimize as spo
from datetime import datetime




#%% Variable Presets

C = [100, 500, 700]
gamma = [0.1, 0.5, 1, 5, 100]

# Data set
data_file_train = 'train'
data_file_test = 'test'

data_folder = 'bank-note/'




#%% Load Data

print('Load Data...')

trainData = pd.read_csv(data_folder + data_file_train + '.csv', sep=',', header=None)
trainData = trainData.to_numpy()
trainData[np.where(trainData[:,4] == 0), 4] = -1

x = trainData[:,range(4)]
y = trainData[:,4]

testData = pd.read_csv(data_folder + data_file_test + '.csv', sep=',', header=None)
testData = testData.to_numpy()
testData[np.where(testData[:,4] == 0), 4] = -1

x2 = testData[:,range(4)]
y2 = testData[:,4]


#%% Functions

def objective_func(a, kernal):    
    cmp_ay = np.reshape(np.multiply(a, y), (-1, 1))
    cmp_ayay = np.matmul(cmp_ay, cmp_ay.T)
    comp_1 = np.multiply(cmp_ayay, kernal)
    comp_1 = (1/2)*np.sum(comp_1)
    
    comp_2 = np.sum(a)
    
    return comp_1 - comp_2


def equality_constraint(a):
    # iteration = [(a[i]*y[i]) for i in range(np.shape(a)[0])]
    iteration = np.multiply(a, y)
    
    return np.sum(iteration)

constraint1 = {'type': 'eq', 'fun': equality_constraint}




#%% Calculate Minimization, Weight Vector, and Bias

for Cx in C:
    for gammax in gamma:    
        # Bounds
        bnds = [(0, (Cx/873))] * np.shape(trainData)[0]
        
        # Calculate Minimization
        print('Calculate Alpha Values for C = ' + str(Cx) + '/' + str(873) + ' and gamma = ' + str(gammax) + '...')
        
        # Calculate Kernal Array
        kernal = np.zeros([np.shape(x)[0], np.shape(x)[0]])
        for i in range(np.shape(x)[0]):
            for j in range(np.shape(x)[0]):
                kernalx = np.linalg.norm(x[i,:] - x[j,:])**2
                kernalx = -(kernalx / gammax)
                kernal[i, j] = np.exp(kernalx) 
        
        a0 = [0] * np.shape(x)[0]
        
        try:
            start_time = datetime.now()
            result = spo.minimize(objective_func, a0, args=(kernal), method='SLSQP', bounds=bnds, constraints=[constraint1], options={'disp': True})
            end_time = datetime.now()
            
            print(result.message + ': ' + 'Duration: ' + str(end_time - start_time))
             
            # Calculate Weighted Vector and Bias
            a = result.x
            
            train_prediction = []
            ay = np.multiply(a, y)
            for j in range(np.shape(x)[0]):
                sum_ayk = np.sum(np.multiply(ay, kernal[:,j]))
                train_prediction.append(np.sign(sum_ayk))
                
            train_error = [0]
            [train_error.append(1) for i in range(np.shape(y)[0]) if y[i] != train_prediction[i]]
            train_error = np.sum(train_error) / np.shape(y)[0]
            
            print('Training Error: ' + str(train_error))
            
            
            test_prediction = []
            kernal_test = np.zeros([np.shape(x)[0], np.shape(x2)[0]])
            for i in range(np.shape(x)[0]):
                for j in range(np.shape(x2)[0]):
                    kernalx = np.linalg.norm(x[i,:] - x2[j,:])**2
                    kernalx = -(kernalx / gammax)
                    kernal_test[i, j] = np.exp(kernalx) 
            
            for j in range(np.shape(x2)[0]):
                sum_ayk = np.sum(np.multiply(ay, kernal_test[:,j]))
                test_prediction.append(np.sign(sum_ayk))
            
            test_error = [0]
            [test_error.append(1) for i in range(np.shape(y2)[0]) if y2[i] != test_prediction[i]]
            test_error = np.sum(test_error) / np.shape(y2)[0]
            
            print('Test Error: ' + str(test_error))
        except: 
            print('Failed for Alpha Values for C = ' + str(Cx) + '/' + str(873) + ' and gamma = ' + str(gammax) + '...')
        
        

        
































