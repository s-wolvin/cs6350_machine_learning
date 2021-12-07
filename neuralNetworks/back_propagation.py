## Savanna Wolvin
# Created: Nov. 29th, 2021
# Edited: Dec. 1st, 2021
    
# SUMMARY

# INPUT

# OUTPUT


#%% Global imports
import numpy as np
import unicodedata




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
    
    cashe_array = np.asarray([[(y - y_star)]]) # cashe this value
    
    # output layer
    dL_output = cashe_array * z_layer[depth].T
    gradient_loss[depth] = dL_output
    
    
    for lx in range(depth, 0, -1):
        # cashe_array = np.multiply(np.tile(cashe_array, (np.shape(w_layer[lx])[0],1)), w_layer[lx])
        cashe_array = cashe_array.T * w_layer[lx]
        
        cashe_array_z_1_z = cashe_array * z_layer[lx][1:].T * (z_layer[lx][1:].T - 1)
        
        multiplied_values = [np.transpose(np.sum(z_value*cashe_array_z_1_z, axis=0)) for z_value in z_layer[lx-1]]
        
        gradient_loss[lx-1] = np.column_stack(multiplied_values)  
        
    
    return gradient_loss



#%% Calculate gradients and print values

gradientLoss = backward_prop(y_star, y, z_layer, X_rand_ex)


names = [unicodedata.name(chr(c)) for c in range(0, 0x10FFFF+1) if unicodedata.name(chr(c), None)]

print('Output Layer')
print("Gradient Loss for \N{GREEK SMALL LETTER DELTA}L/\N{GREEK SMALL LETTER DELTA}w\N{SUPERSCRIPT THREE}\N{SUBSCRIPT ZERO}\N{SUBSCRIPT ONE} = " + str(gradientLoss[2][0,0]))
print("Gradient Loss for \N{GREEK SMALL LETTER DELTA}L/\N{GREEK SMALL LETTER DELTA}w\N{SUPERSCRIPT THREE}\N{SUBSCRIPT ONE}\N{SUBSCRIPT ONE} = "  + str(gradientLoss[2][0,1]))
print("Gradient Loss for \N{GREEK SMALL LETTER DELTA}L/\N{GREEK SMALL LETTER DELTA}w\N{SUPERSCRIPT THREE}\N{SUBSCRIPT TWO}\N{SUBSCRIPT ONE} = "  + str(gradientLoss[2][0,2]))
print("")
print("Hidden Layer")
print("Gradient Loss for \N{GREEK SMALL LETTER DELTA}L/\N{GREEK SMALL LETTER DELTA}w\N{SUPERSCRIPT TWO}\N{SUBSCRIPT ZERO}\N{SUBSCRIPT ONE} = " + str(gradientLoss[1][0,0]))
print("Gradient Loss for \N{GREEK SMALL LETTER DELTA}L/\N{GREEK SMALL LETTER DELTA}w\N{SUPERSCRIPT TWO}\N{SUBSCRIPT ONE}\N{SUBSCRIPT ONE} = "  + str(gradientLoss[1][0,1]))
print("Gradient Loss for \N{GREEK SMALL LETTER DELTA}L/\N{GREEK SMALL LETTER DELTA}w\N{SUPERSCRIPT TWO}\N{SUBSCRIPT TWO}\N{SUBSCRIPT ONE} = "  + str(gradientLoss[1][0,2]))
print("Gradient Loss for \N{GREEK SMALL LETTER DELTA}L/\N{GREEK SMALL LETTER DELTA}w\N{SUPERSCRIPT TWO}\N{SUBSCRIPT ZERO}\N{SUBSCRIPT TWO} = " + str(gradientLoss[1][1,0]))
print("Gradient Loss for \N{GREEK SMALL LETTER DELTA}L/\N{GREEK SMALL LETTER DELTA}w\N{SUPERSCRIPT TWO}\N{SUBSCRIPT ONE}\N{SUBSCRIPT TWO} = "  + str(gradientLoss[1][1,1]))
print("Gradient Loss for \N{GREEK SMALL LETTER DELTA}L/\N{GREEK SMALL LETTER DELTA}w\N{SUPERSCRIPT TWO}\N{SUBSCRIPT TWO}\N{SUBSCRIPT TWO} = "  + str(gradientLoss[1][1,2]))
print("")
print("Input Layer")
print("Gradient Loss for \N{GREEK SMALL LETTER DELTA}L/\N{GREEK SMALL LETTER DELTA}w\N{SUPERSCRIPT ONE}\N{SUBSCRIPT ZERO}\N{SUBSCRIPT ONE} = " + str(gradientLoss[0][0,0]))
print("Gradient Loss for \N{GREEK SMALL LETTER DELTA}L/\N{GREEK SMALL LETTER DELTA}w\N{SUPERSCRIPT ONE}\N{SUBSCRIPT ONE}\N{SUBSCRIPT ONE} = "  + str(gradientLoss[0][0,1]))
print("Gradient Loss for \N{GREEK SMALL LETTER DELTA}L/\N{GREEK SMALL LETTER DELTA}w\N{SUPERSCRIPT ONE}\N{SUBSCRIPT TWO}\N{SUBSCRIPT ONE} = "  + str(gradientLoss[0][0,2]))
print("Gradient Loss for \N{GREEK SMALL LETTER DELTA}L/\N{GREEK SMALL LETTER DELTA}w\N{SUPERSCRIPT ONE}\N{SUBSCRIPT ZERO}\N{SUBSCRIPT TWO} = " + str(gradientLoss[0][1,0]))
print("Gradient Loss for \N{GREEK SMALL LETTER DELTA}L/\N{GREEK SMALL LETTER DELTA}w\N{SUPERSCRIPT ONE}\N{SUBSCRIPT ONE}\N{SUBSCRIPT TWO} = "  + str(gradientLoss[0][1,1]))
print("Gradient Loss for \N{GREEK SMALL LETTER DELTA}L/\N{GREEK SMALL LETTER DELTA}w\N{SUPERSCRIPT ONE}\N{SUBSCRIPT TWO}\N{SUBSCRIPT TWO} = "  + str(gradientLoss[0][1,2]))











