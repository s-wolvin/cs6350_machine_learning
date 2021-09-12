## Savanna Wolvin
# Created: Sep. 10th, 2021
# Edited: Sep. 10th, 2021

# SUMMARY
# We will implement a decision tree learning algorithm for car evaluation task.
# The dataset is from UCI repository(https://archive.ics.uci.edu/ml/datasets/
# car+evaluation). Please download the processed dataset (car.zip) from Canvas. 
# In this task, we have 6 car attributes, and the label is the evaluation of 
# the car. The attribute and label values are listed in the file 
# “data-desc.txt”. All the attributes are categorical. The training data are 
# stored in the file “train.csv”, consisting of 1,000 examples. The test data 
# are stored in “test.csv”, and comprise 728 examples. In both training and 
# test datasets, attribute values are separated by commas; the file 
# “datadesc.txt” lists the attribute names in each column.

# INPUT
# maxTreeDepth  - maximum number of levels on the decision tree
# algorithmType - choose between three algorithm types to create the decision 
#                   tree
# data_file     - file location that contains the training data to create the 
#                   decision tree
# labels        - list of column labels used by the data_file

# OUTPUT
# Boolean Function???




#%% Global Imports
import pandas as pd 
import numpy as np
import sys




#%% Variable Presets

# 1 through 6
maxTreeDepth = 2

# 'Entropy', 'GiniIndex', 'MajorityError'
algorithmType = 'Entropy'

# Data set
data_file = 'car/train.csv'

# column labels
labels = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label']




#%% Main Function

def main():
    ### Load Data
    print('Load data and attributes...')
    trainData = pd.read_csv(data_file, sep=',', header=None)
    trainData = trainData.to_numpy()
    
    attr_dict = {}
    for idx in np.arange(0, np.shape(trainData)[1]):
        attr_dict.update({labels[idx]: np.unique(trainData[:,idx]).tolist()})

    ### Preset Variables & Arrays
    avail_attributes    = np.arange(0, len(labels)-1) 
    used_attributes     = np.empty(0)
    decisionTree        = pd.DataFrame()
    dtOutcome           = pd.DataFrame()
    dtOutcome[labels[len(labels)-1]] = []
    
    ### Determine Head Node & Create Data Frame Containing Decision Tree
    print('Determine Head Node...')
    headNode            = pickAttribute(trainData, avail_attributes)
    decisionTree[labels[headNode]] = attr_dict[labels[headNode]]
    used_attributes     = np.append(used_attributes, headNode)
    used_attributes     = used_attributes.astype(int)
    avail_attributes    = np.delete(avail_attributes, headNode)
    
    ### Loop to Create a Greater Than One Level Decision Tree
    while len(decisionTree.columns) < maxTreeDepth:
        print('Determine ' + str(len(decisionTree.columns)+1) + ' Layer')
        data_lngth = np.shape(trainData)[0]
        dt = decisionTree.to_numpy()
        
        ### Loop Through Each Available Attribute Combination
        for branchX in range(0, len(decisionTree)):
            decision_branch_idx = [i for i in range(data_lngth) if 
                              trainData[i, used_attributes] == dt[branchX,:]]
            
            trainDataX = trainData[:, np.append(avail_attributes,6).tolist()]
            
            branch_attr = pickAttribute(trainDataX[decision_branch_idx,:], avail_attributes)
            
            ### Add Attribute to Branch
            decisionTree[labels[branch_attr]] = attr_dict[labels[branch_attr]]
            used_attributes     = np.append(used_attributes, branch_attr)
            avail_attributes    = np.delete(avail_attributes, branch_attr)
        
    
    
    
    ### Finish Off The End of the Decision Tree By Deciding Most Likely Ending
    dtOutcome = mostLikelyOutcome(decisionTree, dtOutcome, used_attributes, trainData)
    
    
    ### Save Decision Tree
    # decisionTree & dtOutcome
        
        
         
    

#%% Pick Attribute that Best Splits Data

def pickAttribute(trainingData, avail_attributes):
    ### Local Variables
    data_lngth = np.shape(trainingData)[0]
    total_attributes = len(avail_attributes)
    attributes_infoGain = np.zeros((total_attributes,1))
        
    ### Calculate Total Entropy/GiniIndex/MajorityError
    label_ctgrs, label_cnt = np.unique(trainingData[:,total_attributes],     \
                                       return_counts=1)
    total_info = calcInformationGain(label_cnt, sum(label_cnt))
    
    ### Calculate Entropy/GiniIndex/MajorityError for Each Attribute
    for attrX in np.arange(0, total_attributes):
        # print(labels[avail_attributes[attrX]])
        attr_ctgrs, attr_cnt = np.unique(trainingData[:,attrX], return_counts=1)
        
        ### Create Array for Info Loss For Each Attribute's Category
        attr_ctgrs_infoLoss = np.zeros((len(attr_ctgrs), 1))
        
        ### Loop Through Each Attribute's Category
        for attr_ctgrsX in np.arange(0, len(attr_ctgrs)):
            # print(attr_ctgrs[attr_ctgrsX])
            attr_ctgrs_idx = [i for i in range(data_lngth) if 
                              trainingData[i, attrX] == attr_ctgrs[attr_ctgrsX]]
            label_ctgrs, label_cnt = np.unique(trainingData[attr_ctgrs_idx, 
                                                            total_attributes], return_counts=1)
            
            attr_ctgrs_infoLoss[attr_ctgrsX] = calcInformationGain(
                label_cnt, attr_cnt[attr_ctgrsX]) * (attr_cnt[attr_ctgrsX]/data_lngth)
            
        ### Calculate Expected Value 
        attributes_infoGain[attrX] = total_info - sum(attr_ctgrs_infoLoss)
        
    ### Information Loss
    print(labels[int(np.argmax(attributes_infoGain, axis = 0))])
    
    return int(np.argmax(attributes_infoGain, axis = 0))




#%% Function Containing the Alorithms

def calcInformationGain(counts, total):
    xx = 0
    length = len(counts)
    
    if algorithmType == 'Entropy':
        for idx in np.arange(0, length): 
            if counts[idx] != 0 and total != 0:
                xx = xx - (counts[idx]/total)*np.log(counts[idx]/total)
        
    elif algorithmType == 'GiniIndex':
        for idx in np.arange(0, length): 
            if total != 0:
                xx = xx + (counts[idx]/total)**2
        xx = 1 - xx
        
    elif algorithmType == 'MajorityError':
        for idx in np.arange(0, length): 
            xx
            
    else:
        sys.exit('Incorrect Algorithm Type')
        
    
    return xx




#%% Determine Most Likely Outcome For Decision Tree Branch

def mostLikelyOutcome(decisionTree, dtOutcome, used_attributes, trainData):
    ### Preset Variables
    data_lngth = np.shape(trainData)[0]
    used_attributes = used_attributes.astype(int)
    dt = decisionTree.to_numpy()
    
    for idx in range(0, np.shape(dt)[0]):
        ## Calculate the Most Likely Outcome
        decision_branch_idx = [i for i in range(data_lngth) if 
                              trainData[i, used_attributes] == dt[idx,:]]
        
        outcome_ctgrs, outcome_cnt = np.unique(
            trainData[decision_branch_idx,len(labels)-1], return_counts=1)
        
        dtOutcome.loc[idx, 'label'] = outcome_ctgrs[int(np.argmax(outcome_cnt, axis = 0))]
        
        
    return dtOutcome




#%% MAIN
main()





























