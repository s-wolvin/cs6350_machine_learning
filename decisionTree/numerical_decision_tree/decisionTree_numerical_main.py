## Savanna Wolvin
# Created: Sep. 10th, 2021
# Edited: Sep. 10th, 2021

# SUMMARY
# Next, modify your implementation a little bit to support numerical 
# attributes. We will use a simple approach to convert a numerical feature to 
# a binary one. We choose the media (NOT the average) of the attribute values 
# (in the training set) as the threshold, and examine if the feature is bigger 
# (or less) than the threshold. We will use another real dataset from UCI 
# repository(https://archive.ics.uci.edu/ml/datasets/Bank+Marketing). This 
# dataset contains 16 attributes, including both numerical and categorical 
# ones. Please download the processed dataset from Canvas (bank.zip). The 
# attribute and label values are listed in the file “data-desc.txt”. The 
# training set is the file “train.csv”, consisting of 5, 000 examples, and the 
# test “test.csv” with 5, 000 examples as well. In both training and test 
# datasets, attribute values are separated by commas; the file “data-desc.txt” 
# lists the attribute names in each column.

# INPUT
# maxTreeDepth  - maximum number of levels on the decision tree
# algorithmType - choose between three algorithm types to create the decision 
#                   tree
# data_file     - file location that contains the training data to create the 
#                   decision tree
# labels        - list of column labels used by the data_file
# isCategory    - indicate if you want to use 'unknown' as its own attribute 
#                   feature or not

# OUTPUT
# 'car_decision_tree.csv' - CSV File Containing the Attributes, Catagories, 
#                           and Outcomes of the Decision Tree




#%% Global Imports
import pandas as pd 
import numpy as np
import sys




#%% Variable Presets

# 1 through 16
maxTreeDepth = 16

# 'Entropy', 'GiniIndex', 'MajorityError'
algorithmType = 'Entropy'

# Data set
data_file_name = 'train'
data_file = 'bank/' + data_file_name + '.csv'

# column labels
labels = ['age', 'job', 'marital','education','default','balance','housing',\
          'loan','contact','day','month','duration','campaign','pdays',\
              'previous','poutcome','y']
# labels = ['Outlook','Temperature','Humidity','Winds','Play?']
    
# Use Unknown As A Particular Attribute Value
isCategory = False




#%% Main Function

def main():
    ### Load Data
    print('Load data and attributes...')
    trainData = pd.read_csv(data_file, sep=',', header=None)
    trainData = trainData.to_numpy()
    
    ### Use 'Unknown' As A Particular Attribute Value
    if not(isCategory):
        trainData = replaceUnknowns(trainData)
    
    ### Create Dictionary and Change Numeric Values Into Categorical Values
    attr_dict = {}
    for idx in range(0, np.shape(trainData)[1]):
        if type(trainData[0,idx]) == int:
            attr_dict.update({labels[idx]: ['lower','upper']})
            
            median_numeric  = np.median(trainData[:,idx])
            lower = np.where(trainData[:,idx] < median_numeric)
            upper = np.where(trainData[:,idx] >= median_numeric)
            trainData[lower,idx] = 'lower'
            trainData[upper,idx] = 'upper'
            
        else:
            attr_dict.update({labels[idx]: np.unique(trainData[:,idx]).tolist()})
    
    ### Determine Head Node & Create Data Frame Containing Decision Tree
    print('Determine Head Node...')
    headNode            = pickAttribute(trainData, np.arange(0, len(labels)-1) )
    decisionTree_attr   = np.array([labels[headNode]] * len(attr_dict[labels[headNode]]), ndmin=2).T
    decisionTree_ctgr   = np.array(attr_dict[labels[headNode]], ndmin=2).T
    
    ### Save First Level
    dtOutcome = mostLikelyOutcome(decisionTree_attr, decisionTree_ctgr, trainData)
    if isCategory:
        pd.concat([pd.DataFrame(decisionTree_attr), 
                   pd.DataFrame(decisionTree_ctgr), pd.DataFrame(dtOutcome)]).to_csv(
                       'band_dt_' + data_file_name + '_' + algorithmType + '_' + 
                       str(1) + '_unknownAsAttr.csv', index = True, header = True)
    else:    
        pd.concat([pd.DataFrame(decisionTree_attr), 
                   pd.DataFrame(decisionTree_ctgr), pd.DataFrame(dtOutcome)]).to_csv(
                       'band_dt_' + data_file_name + '_' + algorithmType + '_' + 
                       str(1) + '_unknownNotAttr.csv', index = True, header = True)
    
    # decisionTree = {'decisionTree_attr':decisionTree_attr, 'decisionTree_ctgr':decisionTree_ctgr, 'dtOutcome':dtOutcome}
    # if isCategory:
    #     np.savetxt('band_dt_' + data_file_name + '_' + algorithmType + '_' + str(1) + '_unknownAsAttr.csv', [decisionTree], delimiter=',', fmt='%s')        
    # else:    
    #     np.savetxt('bank_dt_' + data_file_name + '_' + algorithmType + '_' + str(1) + '_unknownNotAttr.csv', [decisionTree], delimiter=',', fmt='%s')
    
    
    ### Loop to Create a Greater Than One Level Decision Tree
    level = 2
    while np.shape(decisionTree_attr)[1] < (maxTreeDepth) and np.shape(decisionTree_attr)[1] < (len(labels)-1):
        print('Determine ' + str((np.shape(decisionTree_attr)[1])+1) + ' Layer...')
        data_lngth = np.shape(trainData)[0]
        
        ### Create Temporary Arrays
        decisionTree_attrX = np.zeros((0,np.shape(decisionTree_attr)[1]+1))
        decisionTree_ctgrX = np.zeros((0,np.shape(decisionTree_ctgr)[1]+1))
        
        ### Loop Through Each Available Attribute Combination ###
        for branchX in range(0, np.shape(decisionTree_attr)[0]):
            ### Determine Used and Available Attributes
            used_attributes, avail_attributes = whichAttributes(decisionTree_attr, branchX)
            
            ### Determine if Another Row Is Needed
            if needAnotherNode(trainData, used_attributes, decisionTree_ctgr[branchX,:]):
                ### Determine Next Node
                decision_branch_idx = [i for i in range(data_lngth) if 
                                  np.array_equal(trainData[i, used_attributes], decisionTree_ctgr[branchX,:])]
                trainDataX  = trainData[:, np.append(avail_attributes,(len(labels)-1)).tolist()]
                branch_attr = pickAttribute(trainDataX[decision_branch_idx,:], avail_attributes)
                
                ### Add Attribute to Branch
                xx                  = np.column_stack(
                    [[decisionTree_attr[branchX]] * len(attr_dict[labels[branch_attr]]), 
                     [labels[branch_attr]]* len(attr_dict[labels[branch_attr]])])
                decisionTree_attrX  = np.concatenate([decisionTree_attrX, xx])
                
                xx                  = np.column_stack(
                    [[decisionTree_ctgr[branchX]] * len(attr_dict[labels[branch_attr]]),
                     np.array(attr_dict[labels[branch_attr]], ndmin=2).T])
                decisionTree_ctgrX  = np.concatenate([decisionTree_ctgrX, xx])
            else:
                # print('End of Branch')
                xx = np.column_stack([[decisionTree_attr[branchX]], ['']])
                decisionTree_attrX = np.concatenate([decisionTree_attrX, xx])
                
                xx = np.column_stack([[decisionTree_ctgr[branchX]],['']])
                decisionTree_ctgrX = np.concatenate([decisionTree_ctgrX, xx])
            
        ### Move Temporary Arrays into Permanent Arrays
        decisionTree_attr = decisionTree_attrX
        decisionTree_ctgr = decisionTree_ctgrX
    
        ### Save Decision Tree
        dtOutcome = mostLikelyOutcome(decisionTree_attr, decisionTree_ctgr, trainData)
        if isCategory:
            pd.concat([pd.DataFrame(decisionTree_attr), 
                       pd.DataFrame(decisionTree_ctgr), pd.DataFrame(dtOutcome)]).to_csv(
                           'band_dt_' + data_file_name + '_' + algorithmType + '_' + 
                           str(level) + '_unknownAsAttr.csv', index = True, header = True)
        else:    
            pd.concat([pd.DataFrame(decisionTree_attr), 
                       pd.DataFrame(decisionTree_ctgr), pd.DataFrame(dtOutcome)]).to_csv(
                           'band_dt_' + data_file_name + '_' + algorithmType + '_' + 
                           str(level) + '_unknownNotAttr.csv', index = True, header = True)
        
        # decisionTree = {'decisionTree_attr':decisionTree_attr, 'decisionTree_ctgr':decisionTree_ctgr, 'dtOutcome':dtOutcome}
        # if isCategory:
        #     np.savetxt('band_dt_' + data_file_name + '_' + algorithmType + '_' + str(level) + '_unknownAsAttr.csv', [decisionTree], delimiter=',', fmt='%s')        
        # else:    
        #     np.savetxt('bank_dt_' + data_file_name + '_' + algorithmType + '_' + str(level) + '_unknownNotAttr.csv', [decisionTree], delimiter=',', fmt='%s')
    
        level += 1
    
    
    

#%% Replace the Unknown Value with the Most Common Value

def replaceUnknowns(trainData):
    for attrX in range(0, np.shape(trainData)[1]):
        attr_ctgrs, attr_cnt = np.unique(trainData[:,attrX], return_counts=1)
        mostUsed = attr_ctgrs[int(np.argmax(attr_cnt))]
        
        for idx in range(0, np.shape(trainData)[0]):
            if trainData[idx, attrX] == 'unknown':
                trainData[idx, attrX] = mostUsed

    return trainData




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
        attr_ctgrs, attr_cnt = np.unique(trainingData[:,attrX], return_counts=1)
        
        ### Create Array for Info Loss For Each Attribute's Category
        attr_ctgrs_infoLoss = np.zeros((len(attr_ctgrs), 1))
        
        ### Loop Through Each Attribute's Category
        for attr_ctgrsX in np.arange(0, len(attr_ctgrs)):
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
        for idx in np.arange(0, length): 
            if len(counts) == 1:
                xx = 0
            elif counts[0] > counts[1]:
                xx = (counts[1] / counts[0])
            else:
                xx = (counts[0] / counts[1])
            
    else:
        sys.exit('Incorrect Algorithm Type')
        
    return xx




#%% Return the Available and Used Attributes

def whichAttributes(decisionTree_attr, branchX):
    used_attributes = np.empty([0,0])
    
    ### Loop Through Each Column of the Decision Tree
    for columns in decisionTree_attr[branchX,:]:
        ### If the Variable is Empty, Skip It. The Branch Has Reached Its End
        if columns == '':
            continue

        idx = labels.index(columns)
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
    
    for idx in range(0, np.shape(decisionTree_attr)[0]):
        ## Calculate the Most Likely Outcome
        used_attributes, avail_attributes = whichAttributes(decisionTree_attr, idx)
        decisionTree_ctgrX = decisionTree_ctgr[idx,:]
        
        decision_branch_idx = [i for i in range(data_lngth) if 
                              np.array_equal(trainData[i, used_attributes], decisionTree_ctgrX[decisionTree_ctgrX != ''])]
        outcome_ctgrs, outcome_cnt = np.unique(
            trainData[decision_branch_idx,len(labels)-1], return_counts=1)
        
        if len(outcome_cnt) == 0:
            dtOutcome = np.concatenate([dtOutcome, np.array('', ndmin=1)])
        else:
            dtOutcome = np.concatenate([dtOutcome, np.array(outcome_ctgrs[int(np.argmax(outcome_cnt, axis = 0))], ndmin=1)])
        
    return dtOutcome




#%% MAIN
main()



