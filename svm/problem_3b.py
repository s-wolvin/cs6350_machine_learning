## Savanna Wolvin
# Created: Nov. 8th, 2021
# Edited: 
    
# SUMMARY
# if we want to find out which training examples just sit on the margin
# (neither inside nor outside), what shall we do? Note you are not allowed to
# examine if the functional margin (i.e., yi(w>xi + b)) is 1.

# INPUT 


# OUTPUT



#%% Global Imports
import numpy as np
import scipy.optimize as spo
from datetime import datetime
import matplotlib.pyplot as plt



#%% Variable Presets

C = [100, 500, 700]


#%% Create Data

print('Create Data...')

# single value
# pos = [[0,1,1]]
# neg = [[0,-2,-1]]

# border values only
# pos = [[-4,-1,1],[-2,0,1],[0,1,1],[2,2,1]]
# neg = [[-2,-3,-1],[0,-2,-1],[2,-1,-1],[4,0,-1]]

# random values, no cross over
# pos = [[-4,-1,1],[-2,0,1],[0,1,1],[2,2,1],
#             [-3,1,1],[-2,1,1],[-1,3,1],[1,3,1]]
# neg = [[-2,-3,-1],[0,-2,-1],[2,-1,-1],[4,0,-1],
#             [0,-4,-1],[3,-3,-1],[3,-1,-1],[5,-2,-1]]

# random values, one on margin
# pos = [[-4,-1,1],[-2,0,1],[0,1,1],[2,2,1],
#             [-3,1,1],[-2,1,1],[-1,3,1],[1,3,1],[1,-0.5,1]]
# neg = [[-2,-3,-1],[0,-2,-1],[2,-1,-1],[4,0,-1],
#             [-1,-2.1,-1],[0,-4,-1],[3,-3,-1],[3,-1,-1],[5,-2,-1]]

# random values, one cross over
# pos = [[-4,-1,1],[-2,0,1],[0,1,1],[2,2,1],
#             [-3,1,1],[-2,1,1],[-1,3,1],[1,3,1],[1,-1,1]]
# neg = [[-2,-3,-1],[0,-2,-1],[2,-1,-1],[4,0,-1],
#             [0,-4,-1],[3,-3,-1],[3,-1,-1],[5,-2,-1]]

# one inch margin
pos = [[-2,1,1],[-1,1,1],[0,1,1],[1,1,1],[2,1,1]]
neg = [[-2,-1,-1],[-1,-1,-1],[0,-1,-1],[1,-1,-1],[2,-1,-1]]

# chaotic one inch margin
# pos = [[-2,2,1],[-1,1.5,1],[0,1,1],[1,1,1],[2,2,1],[-3,1,1]]
# neg = [[-2,-0.5,-1],[-1,-1,-1],[0,-1,-1],[1,-0.5,-1],[2,-2,-1]]

dataset = pos+neg

trainData = np.array(dataset)

x = trainData[:,range(2)]
y = trainData[:,2]

pos = np.array(pos)
neg = np.array(neg)


#%% Functions

def objective_func(a):
    summation1 = np.sum(np.array([(a[i]*y[i]*x[i,:]) for i in range(np.shape(a)[0])]), axis=0)
    summation2 = np.sum(np.array([(a[j]*y[j]*x[j,:]) for j in range(np.shape(a)[0])]), axis=0)
    component1 = np.dot(summation1, summation2)
    component2 = np.sum(a)
    
    return component1 - component2


def equality_constraint(a):
    iteration = [(a[i]*y[i]) for i in range(np.shape(a)[0])]
    
    return np.sum(iteration)

constraint1 = {'type': 'eq', 'fun': equality_constraint}




#%% Calculate Minimization, Weight Vector, and Bias

for Cx in C:
    # Bounds
    bnds = [(0, (Cx/873))] * np.shape(trainData)[0]
    
    
    # Calculate Minimization
    print('Calculate Alpha Values for C = ' + str(Cx) + '/' + str(873) + '...')
    
    a0 = [0] * np.shape(x)[0]
    
    start_time = datetime.now()
    result = spo.minimize(objective_func, a0, method='SLSQP', bounds=bnds, constraints=[constraint1])
    end_time = datetime.now()
    
    print(result.message + ': ' + 'Duration: ' + str(end_time - start_time))
    
    
    # Calculate Weighted Vector and Bias
    a = result.x
    
    weightedVector = np.array([(a[i]*y[i]*x[i,:]) for i in range(np.shape(x)[0])])
    normV = [np.linalg.norm(weightedVector[i,:]) for i in range(np.shape(x)[0])]
    
    weightedVector = np.sum(weightedVector, axis=0)
    print('Weight Vector: ' + str(weightedVector))
    
    bias = [(y[j] - np.dot(weightedVector, x[j,:])) for j in range(np.shape(x)[0])]
    bias = np.mean(bias)
    print('Bias: ' + str(bias))
    print('')
    
    plot_x = range(-5,6)
    [nx, ny] = np.meshgrid(plot_x, plot_x)
    
    plt.contourf(nx, ny, weightedVector[0] * nx + weightedVector[1] * ny + bias, [-10,0,10], colors=([[255/255, 196/255, 196/255],[181/255, 255/255, 198/255]]))
    plt.colorbar()

    hyperplane = - (plot_x*weightedVector[0] + bias) / weightedVector[1]

    plt.plot(plot_x, hyperplane, 'black')
    plt.plot(plot_x, hyperplane+1, 'black')
    plt.plot(plot_x, hyperplane-1, 'black')


    plt.scatter(pos[:,0], pos[:,1],s=(a[0:np.shape(pos)[0]]*1000), c='green')
    plt.scatter(neg[:,0], neg[:,1],s=(a[np.shape(pos)[0]:np.shape(a)[0]]*1000), c='red')
    plt.scatter(x[:,0], x[:,1],s=10, c='black')
    
    for i, a_txt in enumerate(a):
        a_txt = round(a_txt, 4)
        if a_txt != 0.0:
            plt.annotate(a_txt, (trainData[i,0], trainData[i,1]), fontsize=8)
    
    plt.title('Maximum Alpha Value is ' + str(round(Cx/873, 5)))
    
    
    plt.show()
    
    
    
    
    



