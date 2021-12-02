## Savanna Wolvin
# Created: Nov. 29th, 2021
# Edited: Dec. 1st, 2021
    
# SUMMARY

# INPUT

# OUTPUT


#%% Global imports
import numpy as np




#%% Create Data

w_layer = {}
w_layer[0] = np.array([[-2,-3],[2,3]])
w_layer[1] = np.array([[-2,-3],[2,3]])
w_layer[2] = np.array([[2,-1.5]])

b_layer = {}
b_layer[0] = np.array([[-1],[1]])
b_layer[1] = np.array([[-1],[1]])
b_layer[2] = np.array([[-1]])

z_layer = {}
z_layer[1] = np.array([[1,0.00247,0.99753]]).T
z_layer[2] = np.array([[1,0.01803,0.98197]]).T

y = -2.43689
y_star = 1
width = 3
depth = 2
X_rand_ex = np.asarray([1,1,1])




#%% Calculate the loss gradient

def backward_prop(y_star, y, z_layer, X_rand_ex):
    gradient_loss = {}
    z_layer[0] = np.asarray([X_rand_ex]).T
    
    cashe_array = (y - y_star) # cashe this value
    
    # output layer
    dL_output = cashe_array * z_layer[depth].T
    gradient_loss[depth] = dL_output
    
    # z = 2
    cashe_array = np.tile(cashe_array, (np.shape(w_layer[2])[0],1)) * w_layer[2]
    # cashe_array = cashe_array * w_layer[2] # dL/dy * dy/dz
    cashe_array_z_1_z = cashe_array * z_layer[2][1:].T * (1 - z_layer[2][1:].T)
    gradient_loss[2-1] = np.dot(z_layer[2-1], cashe_array_z_1_z).T
    
    # z = 1
    cashe_array = np.multiply(np.tile(cashe_array, (np.shape(w_layer[1])[0],1)), w_layer[1].T)
    
    # cashe_array = cashe_array * w_layer[1] # dL/dy * dy/dz
    cashe_array_z_1_z = cashe_array * z_layer[1][1:] * (1 - z_layer[1][1:])
    gradient_loss[1-1] = np.multiply(z_layer[1-1][1:], cashe_array_z_1_z)
    
    
    
    gradient_loss[1-1] = np.multipy(z_layer[1-1], cashe_array_z_1_z).T
    gradient_loss[1-1] = np.expand_dims(z_layer[1-1], axis=1).dot(cashe_array_z_1_z).T
    
    
    #####################################################################
    
    
    # gradient_loss = {}
    # z_layer[0] = X_rand_ex
    
    # cashe_array = np.asarray(y - y_star) # cashe this value
    
    # # output layer
    # gradient_loss[depth] = np.expand_dims(cashe_array * z_layer[depth], axis=0)
    # cashe_array = cashe_array.T
    
    # # z = 2
    # cashe_array = cashe_array * w_layer[2] # dL/dy * dy/dz
    # cashe_array_z_1_z = cashe_array * z_layer[2][1:] * (z_layer[2][1:] - 1)
    # # cashe_array_z_1_z = np.reshape(cashe_array_z_1_z, (1,-1))
    # z_layer_x = np.tile(np.expand_dims(z_layer[2-1], axis=1), np.shape(cashe_array_z_1_z)[0])  
    
    # gradient_loss[2-1] = z_layer_x.dot(cashe_array_z_1_z).T
    
    # # z = 1
    # cashe_array = cashe_array.T * w_layer[1] # dL/dy * dy/dz
    # cashe_array_z_1_z = cashe_array * z_layer[1][1:] * (z_layer[1][1:] - 1)
    # # cashe_array_z_1_z = np.reshape(cashe_array_z_1_z, (1,-1))
    # z_layer_x = np.tile(np.expand_dims(z_layer[1-1], axis=1), np.shape(cashe_array_z_1_z)[0])   
    
    # gradient_loss[1-1] = z_layer_x.dot(cashe_array_z_1_z).T
    
    
    ###################################################################
    
    
    # initial variables
    gradient_loss_1 = {}
    cashe_array = np.asarray(y - y_star) # cashe this value
    
    # Calculate output layer
    gradient_loss_1[depth] = np.expand_dims(cashe_array * z_layer[depth], axis=0)

    # loop through hidden layers
    for lx in range(depth, 0, -1):
        print(lx)
        cashe_array = cashe_array.T * w_layer[lx] # dL/dy * dy/dz
        cashe_array_z_1_z = cashe_array * z_layer[lx][1:] * (z_layer[lx][1:] - 1)
        z_layer_x = np.tile(np.expand_dims(z_layer[lx-1], axis=1), np.shape(cashe_array_z_1_z)[0]) 
        
        gradient_loss_1[lx-1] = z_layer_x.dot(cashe_array_z_1_z).T
        
    
    return gradient_loss, gradient_loss_1



#%%

gradientLoss, gradient_loss_1 = backward_prop(y_star, y, z_layer, X_rand_ex)




































