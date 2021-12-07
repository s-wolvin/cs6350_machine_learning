## Savanna Wolvin
# Created: Dec. 6th, 2021
# Edited: Dec. , 2021
    
# SUMMARY

# INPUT

# OUTPUT



#%% Global Inputs
import numpy as np
import random as rd

#%%
def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

#%%
learning_rate = [0.01, 0.005, 0.0025]
X = np.array([[1, 0.5, -1, 0.3],[1, -1, -2, -2],[1, 1.5, 0.2, -2.5]])
Y = np.array([[1],[-1],[1]])
w = np.array([[0,0,0,0]])

for lr in learning_rate:
    print("Gradients for Learning Rate " + str(lr) + ":")
    for tx in range(3):
        rand_idx = rd.sample(range(0, np.shape(X)[0]), np.shape(X)[0]) # shuffle values
        X_rand = X[rand_idx, :]
        Y_rand = Y[rand_idx]
        
        for ex in range(np.shape(X)[0]):
            error = -Y_rand[ex]*np.dot(w,X_rand[ex])
            grad = -Y_rand[ex]*X_rand[ex]*np.exp(error)*sigmoid(error) + (2*w / sigmoid(w)**2)
            print(str(grad))
            w = w - lr*grad
    print("")