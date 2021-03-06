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
# 'car_dt_' + data_file_name + '_' + algorithmType + '_' + levelNum + '_level.csv' 
#       - CSV File Containing the Attributes, Catagories, and Outcomes of the 
#           Decision Tree




#%% Global Imports
import pandas as pd 
import numpy as np
import sys




#%% Variable Presets

# 1 through 6, it will create arrays for all levels at and below
maxTreeDepth = 6

# 'Entropy', 'GiniIndex', 'MajorityError'
algorithmType = 'MajorityError'

# Data set
data_file_name = 'test'
data_file = 'car/' + data_file_name + '.csv'

# column labels
labels = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label']
# labels = ['Outlook','Temperature','Humidity','Winds','Play?']




#%% Main Function

def main():
    ### Load Data
    print('Load data and attributes...')
    trainData = pd.read_csv(data_file, sep=',', header=None)
    trainData = trainData.to_numpy()
    
    attr_dict = {}
    for idx in np.arange(0, np.shape(trainData)[1]):
        attr_dict.update({labels[idx]: np.unique(trainData[:,idx]).tolist()})
    
    ### Determine Head Node & Create Data Frame Containing Decision Tree
    print('Determine Head Node...')
    headNode            = pickAttribute(trainData, np.arange(0, len(labels)-1) )
    decisionTree_attr   = np.array([labels[headNode]] * len(attr_dict[labels[headNode]]), ndmin=2)
    decisionTree_ctgr   = np.array(attr_dict[labels[headNode]], ndmin=2)
    
    ### Save First Level
    dtOutcome = mostLikelyOutcome(decisionTree_attr, decisionTree_ctgr, trainData)
    avg_PredictionError = avgPredictionError(trainData, decisionTree_attr, decisionTree_ctgr, dtOutcome)
    pd.concat([pd.DataFrame(decisionTree_attr), 
               pd.DataFrame(decisionTree_ctgr), pd.DataFrame(dtOutcome), pd.DataFrame([avg_PredictionError])
               ]).to_csv('car_dt_' + data_file_name + '_' + algorithmType + '_' + 
                   str(1) + '_level.csv', index = True, header = True) 
    
    # decisionTree = {'decisionTree_attr':decisionTree_attr, 'decisionTree_ctgr':decisionTree_ctgr, 'dtOutcome':dtOutcome}
    # np.savetxt('car_dt_' + data_file_name + '_' + algorithmType + '_' + str(1) + '_level.csv', [decisionTree], delimiter=',', fmt='%s')   

    ### Loop to Create a Greater Than One Level Decision Tree
    level = 2
    while np.shape(decisionTree_attr)[0] < (maxTreeDepth) and np.shape(decisionTree_attr)[0] < (len(labels)-1):
        print('Determine ' + str((np.shape(decisionTree_attr)[0])+1) + ' Layer...')
        data_lngth = np.shape(trainData)[0]
        
        ### Create Temporary Arrays
        decisionTree_attrX = np.zeros((np.shape(decisionTree_attr)[0]+1,0))
        decisionTree_ctgrX = np.zeros((np.shape(decisionTree_ctgr)[0]+1,0))
        
        ### Loop Through Each Available Attribute Combination ###
        for branchX in range(0, np.shape(decisionTree_attr)[1]):
            ### Determine Used and Available Attributes
            used_attributes, avail_attributes = whichAttributes(decisionTree_attr, branchX)
            
            ### Determine if Another Row Is Needed
            if needAnotherNode(trainData, used_attributes, decisionTree_ctgr[:,branchX]):
                ### Determine Next Node
                decision_branch_idx = [i for i in range(data_lngth) if 
                                  np.array_equal(trainData[i, used_attributes], decisionTree_ctgr[:,branchX])]
                trainDataX  = trainData[:, np.append(avail_attributes,(len(labels)-1)).tolist()]
                branch_attr = pickAttribute(trainDataX[decision_branch_idx,:], avail_attributes)
                
                ### Add Attribute to Branch
                xx                  = np.column_stack(
                    [[decisionTree_attr[:,branchX]] * len(attr_dict[labels[branch_attr]]), 
                     [labels[branch_attr]]* len(attr_dict[labels[branch_attr]])])
                decisionTree_attrX  = np.column_stack([decisionTree_attrX, xx.T])
                
                xx                  = np.column_stack(
                    [[decisionTree_ctgr[:,branchX]] * len(attr_dict[labels[branch_attr]]),
                     np.array(attr_dict[labels[branch_attr]], ndmin=2).T])
                decisionTree_ctgrX  = np.column_stack([decisionTree_ctgrX, xx.T])
            else:
                # print('End of Branch')
                xx = np.column_stack([[decisionTree_attr[:,branchX]], ['']])
                decisionTree_attrX = np.column_stack([decisionTree_attrX, xx.T])
                
                xx = np.column_stack([[decisionTree_ctgr[:,branchX]],['']])
                decisionTree_ctgrX = np.column_stack([decisionTree_ctgrX, xx.T])
            
        ### Move Temporary Arrays into Permanent Arrays
        decisionTree_attr = decisionTree_attrX
        decisionTree_ctgr = decisionTree_ctgrX
        
        ### Save Current Level
        dtOutcome = mostLikelyOutcome(decisionTree_attr, decisionTree_ctgr, trainData)
        avg_PredictionError = avgPredictionError(trainData, decisionTree_attr, decisionTree_ctgr, dtOutcome)
        
        pd.concat([pd.DataFrame(decisionTree_attr), 
               pd.DataFrame(decisionTree_ctgr), pd.DataFrame(dtOutcome), pd.DataFrame([avg_PredictionError])]).to_csv(
                   'car_dt_' + data_file_name + '_' + algorithmType + '_' + 
                   str(level) + '_level.csv', index = True, header = True) 
        
        # decisionTree = {'decisionTree_attr':decisionTree_attr, 'decisionTree_ctgr':decisionTree_ctgr, 'dtOutcome':dtOutcome}
        # np.savetxt('car_dt_' + data_file_name + '_' + algorithmType + '_' + str(level) + '_level.csv', [decisionTree], delimiter=',', fmt='%s')    

        ### Next Level Label
        level += 1




#%% Pick Attribute that Best Splits Data

def pickAttribute(trainingData, avail_attributes):
    ### Local Variables
    data_lngth          = np.shape(trainingData)[0]
    total_attributes    = len(avail_attributes)
    attributes_infoGain = np.zeros((total_attributes,1))
        
    ### Calculate Total Entropy/GiniIndex/MajorityError
    label_ctgrs, label_cnt = np.unique(trainingData[:,total_attributes],return_counts=1)
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
            attr_ctgrs_idx          = [i for i in range(data_lngth) if 
                              np.array_equal(trainingData[i, attrX], attr_ctgrs[attr_ctgrsX])]
            
            label_ctgrs, label_cnt  = np.unique(trainingData[attr_ctgrs_idx, 
                                                            total_attributes], return_counts=1)            
            
            attr_ctgrs_infoLoss[attr_ctgrsX] = calcInformationGain(
                label_cnt, attr_cnt[attr_ctgrsX]) * (attr_cnt[attr_ctgrsX]/data_lngth)
            
        ### Calculate Expected Value 
        attributes_infoGain[attrX] = total_info - sum(attr_ctgrs_infoLoss)
        
    ### Information Loss
    # print(labels[avail_attributes[int(np.argmax(attributes_infoGain, axis = 0))]])
    
    return avail_attributes[int(np.argmax(attributes_infoGain, axis = 0))]




#%% Function Containing the Algorithms

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
        if len(counts) == 1:
            xx = 0
        else:
            max = int(np.argmax(counts, axis = 0))
            xx = (sum(counts) - max) / sum(counts)
                
    else:
        sys.exit('Incorrect Algorithm Type')
        
    
    return xx




#%% Return the Available and Used Attributes

def whichAttributes(decisionTree_attr, branchX):
    used_attributes = np.empty([0,0])
    
    ### Loop Through Each Column of the Decision Tree
    for row in decisionTree_attr[:,branchX]:
        ### If the Variable is Empty, Skip It. The Branch Has Reached Its End
        if row == '':
            continue

        idx = labels.index(row)
        used_attributes = np.append(used_attributes, idx)
    
    ### Create Available Attributes Array
    used_attributes     = used_attributes.astype(int)
    avail_attributes    = np.arange(0, len(labels)-1)
    avail_attributes    = np.delete(avail_attributes, used_attributes)
        
    return used_attributes, avail_attributes
        



#%% Determine If Another Node Is Needed

def needAnotherNode(trainData, used_attributes, decisionTree_ctgr):
    ### Variable Presets
    data_lngth = np.shape(trainData)[0]
    decisionTree_ctgr = decisionTree_ctgr[decisionTree_ctgr != '']
    
    ### Determine Count of Endings For Each Current Branch
    decision_branch_idx = [i for i in range(data_lngth) if 
                              np.array_equal(trainData[i, used_attributes], decisionTree_ctgr)]
    outcome_ctgrs, outcome_cnt = np.unique(
            trainData[decision_branch_idx,len(labels)-1], return_counts=1)
    
    ### Return True if More Branches are Needed
    if len(outcome_cnt) > 1:
        return True
    else:
        return False




#%% Determine Most Likely Outcome For Decision Tree Branch

def mostLikelyOutcome(decisionTree_attr, decisionTree_ctgr, trainData):
    ### Preset Variables
    data_lngth = np.shape(trainData)[0]
    dtOutcome = np.zeros([0])
    
    for idx in range(0, np.shape(decisionTree_attr)[1]):
        ## Calculate the Most Likely Outcome
        used_attributes, avail_attributes = whichAttributes(decisionTree_attr, idx)
        decisionTree_ctgrX = decisionTree_ctgr[:,idx]
        
        decision_branch_idx = [i for i in range(data_lngth) if 
                              np.array_equal(trainData[i, used_attributes], decisionTree_ctgrX[decisionTree_ctgrX != ''])]
        outcome_ctgrs, outcome_cnt = np.unique(
            trainData[decision_branch_idx,len(labels)-1], return_counts=1)
        
        if len(outcome_cnt) > 0:
            dtOutcome = np.concatenate([dtOutcome, np.array(outcome_ctgrs[int(np.argmax(outcome_cnt, axis = 0))], ndmin=1)])
        else:
            dtOutcome = np.concatenate([dtOutcome, ['']])

        
    return np.array([dtOutcome])




#%% Calculate Average Prediction Error

def avgPredictionError(trainData, decisionTree_attr, decisionTree_ctgr, dtOutcome):
    data_lngth = np.shape(trainData)[0]
    total_attributes = np.shape(trainData)[1]-1
    branches = np.shape(decisionTree_attr)[1]
    errors = 0
    
    for idx in range(0, branches):
        dt_ctgr_branch = decisionTree_ctgr[:,idx]
        
        used_attributes, avail_attributes = whichAttributes(decisionTree_attr, idx)
        
        attr_ctgrs_idx          = [i for i in range(data_lngth) if 
                          np.array_equal(trainData[i, used_attributes], dt_ctgr_branch)]
        
        label_ctgrs, label_cnt  = np.unique(trainData[attr_ctgrs_idx, 
                                                        total_attributes], return_counts=1)
        
        errors += sum(label_cnt[np.where(label_ctgrs != dtOutcome[:,idx])])
        
    
    return errors/data_lngth




#%% MAIN
main()



