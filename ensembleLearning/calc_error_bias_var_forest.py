## Savanna Wolvin
# Created: Oct. 17th, 2021
# Edited: 
    
# SUMMARY
# calculate the training, test errors and the bias/variances


#%% imports
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt




#%% presets

# number of iterations
T = 100

dataSet = 'test'
data_file = 'bank-1/' + dataSet + '.csv'

subset_num = [2, 4, 6]



#%% load 

data = pd.read_csv(data_file, sep=',', header=None)
data = data.to_numpy()
data = data[:,np.shape(data)[1]-1]
data[data == 'no'] = 0
data[data == 'yes'] = 1

subset2 = pd.read_csv('bank_' + dataSet + '_baggedTrees_outcome_' + 
                '_attrSubset_' + str(2) + '_100.csv', sep=',', header=0, index_col=0)
subset2 = subset2.to_numpy()
subset2[subset2 == 'no'] = 0
subset2[subset2 == 'yes'] = 1


subset4 = pd.read_csv('bank_' + dataSet + '_baggedTrees_outcome_' + 
                '_attrSubset_' + str(4) + '_100.csv', sep=',', header=0, index_col=0)
subset4 = subset4.to_numpy()
subset4[subset4 == 'no'] = 0
subset4[subset4 == 'yes'] = 1

subset6 = pd.read_csv('bank_' + dataSet + '_baggedTrees_outcome_' + 
                '_attrSubset_' + str(6) + '_100.csv', sep=',', header=0, index_col=0)
subset6 = subset6.to_numpy()
subset6[subset6 == 'no'] = 0
subset6[subset6 == 'yes'] = 1

subset_var = np.zeros([np.shape(data)[0], T, len(subset_num)])
subset_var[:,:,0] = subset2
subset_var[:,:,1] = subset4
subset_var[:,:,2] = subset6



#%% Calculate prediction error

error = np.zeros([np.shape(subset_num)[0], T])
prdtns_all = np.zeros([np.shape(data)[0], T, len(subset_num)])

for subsetx in range(len(subset_num)): # loop through each subset
    
    for tx in range(T): # loop through each iteration/tree
        errorx = 0
    
        for ex in range(np.shape(data)[0]): # loop through each example
            prd_vle, prd_cnt = np.unique(subset_var[ex, range(tx+1),subsetx], return_counts=1)
            prdtns_all[ex, tx, subsetx] = prd_vle[np.argmax(prd_cnt)]
            
            if data[ex] != prd_vle[np.argmax(prd_cnt)]:
                errorx += 1
                
        error[subsetx, tx] = errorx
                    
error = error / np.shape(data)[0]



#%% plotting

fig1 = plt.gcf()
plt.plot(np.arange(T) , error[0,:], color='blue', label='Two Attributes')
plt.plot(np.arange(T) , error[1,:], color='red', label='Four Attributes')
plt.plot(np.arange(T) , error[2,:], color='green', label='Six Attributes')
plt.xlabel('Iterations')
plt.ylabel('Prediction Error')
plt.title('Random Forests Decision Tree')
plt.legend()
plt.show()
fig1.savefig(dataSet + '_error.png', dpi=300)



#%% Bias and variance for single tree

bias_st = np.zeros([len(subset_num)])
vari_st = np.zeros([len(subset_num)])
gsqe_st = np.zeros([len(subset_num)])

for subsetx in range(len(subset_num)): # loop through each subset
    # bias term
    for idx in range(np.shape(prdtns_all)[0]):
        bias_st[subsetx] += ( (data[idx] - subset_var[idx,0, subsetx])**2 )
            
    bias_st[subsetx] = (bias_st[subsetx] /  np.shape(data)[0])
    
    # variance term    
    avg_prd = np.mean(subset_var[:,:,subsetx], axis=1)
    for idx in range(np.shape(subset_var)[0]):
        vari_st[subsetx] += (1/(T-1)) * np.sum((subset_var[idx,:,subsetx] - avg_prd[idx]) **2)
        
    vari_st[subsetx] = vari_st[subsetx] / np.shape(subset_var)[0]
    
    gsqe_st[subsetx] = vari_st[subsetx] + bias_st[subsetx]
    
print('Single Tree Bias for 2,4,6: ' + str(bias_st))
print('Single Tree Variance for 2,4,6: ' + str(vari_st))
print('Single Tree Squared Error for 2,4,6: ' + str(gsqe_st))



#%% bias and variance for total tree

bias_ft = np.zeros([len(subset_num)])
vari_ft = np.zeros([len(subset_num)])
gsqe_ft = np.zeros([len(subset_num)])

for subsetx in range(len(subset_num)): # loop through each subset
    # bias term
    for idx in range(np.shape(prdtns_all)[0]):
        bias_ft[subsetx] += ( (data[idx] - prdtns_all[idx,0, subsetx])**2 )
            
    bias_ft[subsetx] = (bias_ft[subsetx] /  np.shape(data)[0])
    
    # variance term
    avg_prd = np.mean(prdtns_all[:,:,subsetx], axis=1)
    for idx in range(np.shape(subset_var)[0]):
        vari_ft[subsetx] += (1/(T-1)) * np.sum((prdtns_all[idx,:,subsetx] - avg_prd[idx]) **2)
        
    vari_ft[subsetx] = vari_ft[subsetx] / np.shape(subset_var)[0]
    
    gsqe_ft[subsetx] = vari_ft[subsetx] + bias_ft[subsetx]

print('Single Tree Bias for 2,4,6: ' + str(bias_ft))
print('Single Tree Variance for 2,4,6: ' + str(vari_ft))
print('Single Tree Squared Error for 2,4,6: ' + str(gsqe_ft))



































