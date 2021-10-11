## Savanna Wolvin
# Created: Oct. 5th, 2021
# Edited: Oct. 11th, 2021








#%% Global Imports
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt



#%% Variable Presets

# Data set
data_file_name = 'train'
data_file = 'concrete/' + data_file_name + '.csv'

# column labels
labels = ['Cement','Slag','Fly Ash','Water','SP','Coarse Aggr','Fine Aggr','Slump']

# learning rate
r = 0.000197

# number of iterations
T = 1000

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
    
    batchGradient = np.zeros([T, np.shape(w_matrix)[0]])
    
    for tx in range(T):  # each iteration        
        ### Calculate gradient for each component in the W matix
        for wi in range(np.shape(w_matrix)[0]):  # each w value
            xx = 0
            
            for ex in range(np.shape(trainData)[0]): # each example
                xx += (trainData[ex,data_lngth] - np.dot(w_matrix[:,tx], trainData[ex,range(0,data_lngth)])) * (trainData[ex,wi]) 
        
            batchGradient[tx, wi] = -xx
            
        
        ### Update W Matrix
        for wi in range(np.shape(w_matrix)[0]):
            w_matrix[wi, tx+1] = w_matrix[wi, tx] - (r * batchGradient[tx, wi])
            
        
        ### Report Norm of Weight Vector Difference
        diff_w      = w_matrix[:, tx+1] - w_matrix[:, tx]
        norm[tx]    = np.linalg.norm(diff_w)
        # print('Norm of Weight Vector: ' + str(round(norm[tx], 6)))
        
        ### Report the Cost Function
        cost_func[tx] = costFunction(trainData, w_matrix, tx)
        # print('Cost Function Value: ' + str(round(cost_func[tx], 6)))
        
        
    test_cost_function = testCostFunction(w_matrix, tx)
        
    ### Plot Cost Function
    plt.plot(range(T), cost_func)
    plt.title('Cost Function Value VS Iterations for Training Data')
    plt.ylabel('Cost Function')
    plt.xlabel('Iterations')
    plt.grid()
    plt.show
    plt.savefig('cost_vs_iterations.png', dpi = 300)
    
    print('Final Cost Value: ' + str(cost_func[T-1]))
    print('Test Cost Value: ' + str(test_cost_function))
    print('Final Norm Value: ' + str(norm[T-1]))





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























