## Savanna Wolvin
# Created: Oct. 11th, 2021
# Edited: Oct. 11th, 2021

# SUMMARY
# Implement the stochastic gradient descent (SGD) algorithm. You can 
# initialize your weight vector to be 0. Each step, you randomly sample a 
# training example, and then calculate the stochastic gradient to update the 
# weight vector. Tune the learning rate r to ensure your SGD converges. To 
# check convergence, you can calculate the cost function of the training data 
# after each stochastic gradient update, and draw a figure showing how the 
# cost function values vary along with the number of updates. At the beginning, 
# your curve will oscillate a lot. However, with an appropriate r, as more and 
# more updates are finished, you will see the cost function tends to converge. 
# Please report the learned weight vector, and the learning rate you chose, 
# and the cost function value of the test data with your learned weight vector.

# INPUT
# data_file - file location that contains the training data to create the 
#                   decision tree
# labels    - list of column labels used by the data_file
# r         - Learning Rate
# T         - Number of Iterations

# OUTPUT
# 'cost_vs_iterations_4b.png' - PNG File Showing the Relationship Between the 
#           Cost Function Value and the Iteration Number for the Training Data






#%% Global Imports
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import random as rd



#%% Variable Presets

# Data set
data_file_name = 'train'
data_file = 'concrete/' + data_file_name + '.csv'

# column labels
labels = ['Cement','Slag','Fly Ash','Water','SP','Coarse Aggr','Fine Aggr','Slump']

# learning rate
r = 0.000095

# number of iterations
T = 5000

# initial W
w_matrix = np.zeros([7, T+1])

norm        = np.zeros([T])
cost_func   = np.zeros([T])




#%% Main

def main():
    ### Load Data
    print('Load data and attributes...')
    trainData = pd.read_csv(data_file, sep=',', header=None)
    trainData = trainData.to_numpy()
    data_lngth = np.shape(trainData)[1]-1

    ### Loop Through Each Iteration
    for tx in range(T):        
        ### Choose Random Example
        rand_int = rd.randint(0, data_lngth)
        
        ### Loop Through Each Coefficient
        for wi in range(np.shape(w_matrix)[0]):  # each w value
            w_matrix[wi, tx+1] = w_matrix[wi, tx] + ( r * (trainData[rand_int, data_lngth] 
                            - np.dot(w_matrix[:,tx], trainData[rand_int,range(0,data_lngth)])) 
                            * trainData[rand_int,wi] )
        
        ### Report Norm of Weight Vector Difference
        diff_w      = w_matrix[:, tx+1] - w_matrix[:, tx]
        norm[tx]    = np.linalg.norm(diff_w)
        # print('Norm of Weight Vector: ' + str(round(norm[tx], 6)))
        
        ### Calculate the Cost Function
        cost_func[tx] = costFunction(trainData, w_matrix, tx+1)
        # print('Cost Function Value: ' + str(round(cost_func[tx], 6)))
        
        
    test_cost_function = testCostFunction(w_matrix, tx+1)
        
    # ### Plot Cost Function
    plt.plot(range(T), cost_func)
    plt.title('Cost Function Value VS Iterations for Training Data')
    plt.ylabel('Cost Function')
    plt.xlabel('Iterations')
    plt.grid()
    plt.show
    plt.savefig('cost_vs_iterations_4b.png', dpi = 300)
    
    print('Final Cost Value: ' + str(cost_func[tx]))
    print('Test Cost Value: ' + str(test_cost_function))
    print('Mean of Last 10 Norm Value: ' + str(np.mean(norm[tx-10:tx])))
    print(w_matrix[:,T])




#%% Calculate Value of Cost Function with New Values

def costFunction(trainData, w_matrix, tx):
    value = 0
    data_lngth = np.shape(trainData)[1]-1
    
    for ex in range(np.shape(trainData)[0]): # each example
        value += ( trainData[ex, data_lngth] - np.dot(w_matrix[:,tx], trainData[ex,range(0,data_lngth)]) )**2

    value = value * (0.5)
    return value




#%% 

def testCostFunction(w_matrix, tx):
    testData = pd.read_csv('concrete/test.csv', sep=',', header=None)
    testData = testData.to_numpy()
    data_lngth = np.shape(testData)[1]-1
    
    value = 0
    data_lngth = np.shape(testData)[1]-1
    
    for ex in range(np.shape(testData)[0]): # each example
        value += ( testData[ex, data_lngth] - np.dot(w_matrix[:,tx], testData[ex,range(0,data_lngth)]) )**2
    
    value = value * (0.5)
    
    return value




#%% Main
main()























