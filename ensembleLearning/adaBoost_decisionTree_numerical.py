## Savanna Wolvin
# Created: Oct. 4th, 2021
# Edited: Oct. 12th, 2021

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
import matplotlib.pyplot as plt




#%% Variable Presets

# 1 through 16
maxTreeDepth = 2

# 'Entropy', 'GiniIndex', 'MajorityError'
algorithmType = 'Entropy'

# number of iterations
T = 500

# Data set
data_file_name = 'train'
data_file = 'bank-1/' + data_file_name + '.csv'

# column labels
labels = ['age', 'job', 'marital','education','default','balance','housing',\
          'loan','contact','day','month','duration','campaign','pdays',\
              'previous','poutcome','y']
# labels = ['Outlook','Temperature','Humidity','Winds','Play?']
    
# Use Unknown As A Particular Attribute Value
isCategory = True




#%% Main Function

def main():
    ### Load Data
    print('Load data and attributes...')
    trainData = pd.read_csv(data_file, sep=',', header=None)
    trainData = trainData.to_numpy()
    
    testData = pd.read_csv('bank-1/test.csv', sep=',', header=None)
    testData = testData.to_numpy()
    
    ### Use 'Unknown' As A Particular Attribute Value
    if not(isCategory):
        trainData = replaceUnknowns(trainData)
        testData = replaceUnknowns(testData)
    
    ### Create Dictionary and Change Numeric Values Into Categorical Values
    attr_dict = {}
    for idx in range(0, np.shape(trainData)[1]):
        if type(trainData[0,idx]) == int:
            
            if idx == 13:
                attr_dict.update({labels[idx]: ['lower','upper', 'nan']})
                
                nan_loc = np.where(trainData[:,idx] == 'nan')
                num_loc = np.where(trainData[:,idx] != 'nan')
                median_numeric  = np.median(trainData[num_loc[0],idx])
                lower = np.where(trainData[:,idx] < median_numeric)
                upper = np.where(trainData[:,idx] >= median_numeric)
                trainData[lower,idx] = 'lower'
                trainData[upper,idx] = 'upper'
                trainData[nan_loc[0],idx] = 'nan'
                
                nan_loc = np.where(testData[:,idx] == 'nan')
                num_loc = np.where(testData[:,idx] != 'nan')
                median_numeric  = np.median(testData[num_loc[0],idx])
                lower = np.where(testData[:,idx] < median_numeric)
                upper = np.where(testData[:,idx] >= median_numeric)
                testData[lower,idx] = 'lower'
                testData[upper,idx] = 'upper'
                testData[nan_loc[0],idx] = 'nan'
                
            else:
                attr_dict.update({labels[idx]: ['lower','upper']})   
                median_numeric  = np.median(trainData[:,idx])
                lower = np.where(trainData[:,idx] < median_numeric)
                upper = np.where(trainData[:,idx] >= median_numeric)
                trainData[lower,idx] = 'lower'
                trainData[upper,idx] = 'upper'
                
                median_numeric  = np.median(testData[:,idx])
                lower = np.where(testData[:,idx] < median_numeric)
                upper = np.where(testData[:,idx] >= median_numeric)
                testData[lower,idx] = 'lower'
                testData[upper,idx] = 'upper'
            
        else:
            attr_dict.update({labels[idx]: np.unique(trainData[:,idx]).tolist()})
    
    ### Create Array to Hold Sample Weight of Each Example
    weight = np.zeros([np.shape(trainData)[0],T+1])
    weight[:,0] = 1/np.shape(trainData)[0]
    
    dtOutcome_all_train = np.zeros([np.shape(trainData)[0],T], dtype=object)
    dtOutcome_all_test = np.zeros([np.shape(testData)[0],T], dtype=object)
    prdctnError_train = np.zeros([np.shape(trainData)[0],1], dtype=object)
    prdctnError_test = np.zeros([np.shape(testData)[0],1], dtype=object)
    amountOfSay = np.zeros([np.shape(trainData)[0],1], dtype=object)
    sum_error_train = np.zeros([np.shape(trainData)[0],1], dtype=object)
    sum_error_test = np.zeros([np.shape(testData)[0],1], dtype=object)
    
    ### Loop Through Each Iteration to Create a Forest of Stumps
    for tx in range(0, T):
        print('Iteration ' + str(tx+1))
        # ### Scale Weight to be Equal to Total Number of Examples
        # weightData = weight * np.shape(trainData)[0]
        
        ### Determine Head Node & Create Data Frame Containing Decision Tree
        # print('Determine Head Node...')
        headNode            = pickAttribute(trainData, np.arange(0, len(labels)-1), weight[:,tx])
        decisionTree_attr   = np.array([labels[headNode]] * len(attr_dict[labels[headNode]]), ndmin=2)
        decisionTree_ctgr   = np.array(attr_dict[labels[headNode]], ndmin=2)
        
        ### Loop to Create a Greater Than One Level Decision Tree
        level = 2
        while np.shape(decisionTree_attr)[0] < (maxTreeDepth) and np.shape(decisionTree_attr)[0] < (len(labels)-1):
            # print('Determine ' + str((np.shape(decisionTree_attr)[0])+1) + ' Layer...')
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
                    branch_attr = pickAttribute(trainDataX[decision_branch_idx,:], avail_attributes, weight[:,tx])
                    
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
            
            level += 1
        
        ### Save Decision Tree
        dtOutcome_train = mostLikelyOutcome(decisionTree_attr, decisionTree_ctgr, trainData, weight[:,tx])
        dtOutcome_all_train[:,tx] = dtOutcome_train[:,0]
        
        avg_PredictionError, weightx, amountOfSayx = avgPredictionError(trainData, decisionTree_attr, decisionTree_ctgr, dtOutcome_train[:,0], weight[:,tx])
        amountOfSay[tx] = amountOfSayx
        prdctnError_train[tx] = avg_PredictionError
        weight[:,tx+1] = weightx
        
        dtOutcome_test = mostLikelyOutcome(decisionTree_attr, decisionTree_ctgr, testData, weight[:,tx])
        dtOutcome_all_test[:,tx] = dtOutcome_test[:,0]
        avg_PredictionError, _, _ = avgPredictionError(testData, decisionTree_attr, decisionTree_ctgr, dtOutcome_test[:,0], weight[:,tx])
        prdctnError_test[tx] = avg_PredictionError
        
        fig1 = plt.gcf()
        plt.plot(np.arange(tx+1) , prdctnError_train[np.arange(tx+1)], color='blue', label='Training Data')
        plt.plot(np.arange(tx+1) , prdctnError_test[np.arange(tx+1)], color='red', label='Test Data')
        plt.xlabel('Iterations')
        plt.ylabel('Prediction Error')
        plt.title('Adaboost Decision Tree')
        plt.legend()
        # plt.show()
        fig1.savefig('test_train_error_adaboost_stumps.png', dpi=300)
        plt.clf()
               
        
        ### calc error as the stumps combine
        dtOutcome_train[:] = ''
        for row in range(np.shape(dtOutcome_all_train)[0]):
            labels_outcome = np.unique(dtOutcome_all_train[row, np.arange(tx+1)])
            
            outcome_sum = [np.sum(amountOfSay[np.where(dtOutcome_all_train[row, np.arange(tx+1)] == labels_outcome[lx])]) for lx in range(len(labels_outcome))]
            dtOutcome_train[row,0] = labels_outcome[int(np.argmax(outcome_sum, axis = 0))]
            
            
        dtOutcome_test[:] = ''
        for row in range(np.shape(dtOutcome_all_test)[0]):
            labels_outcome = np.unique(dtOutcome_all_test[row, np.arange(tx+1)])
            
            outcome_sum = [np.sum(amountOfSay[np.where(dtOutcome_all_test[row, np.arange(tx+1)] == labels_outcome[lx])]) for lx in range(len(labels_outcome))]
            dtOutcome_test[row,0] = labels_outcome[int(np.argmax(outcome_sum, axis = 0))]
        
        
        sum_error_train[tx] = np.sum(weight[np.where(dtOutcome_train[:,0] != trainData[:,16]),tx])
        sum_error_test[tx] = np.sum(weight[np.where(dtOutcome_test[:,0] != testData[:,16]),tx])
        
        fig2 = plt.gcf()
        plt.plot(np.arange(tx+1) , sum_error_train[np.arange(tx+1)], color='blue', label='Training Data')
        plt.plot(np.arange(tx+1) , sum_error_test[np.arange(tx+1)], color='red', label='Test Data')
        plt.xlabel('Iterations')
        plt.ylabel('Prediction Error')
        plt.title('Adaboost Decision Tree')
        plt.legend()
        # plt.show()
        fig2.savefig('test_train_error_adaboost_all.png', dpi=300)
        plt.clf()
            
            
            
    # if isCategory:
    #     pd.concat([pd.DataFrame(prdctnError)]).to_csv(
    #                    'bank_' + data_file_name + '_adaBoost_error_' + 
    #                    str(T) + '_unknownAsAttr.csv', index = True, header = True)
    #     # decisionTree = {'decisionTree_attr':decisionTree_attr, 'decisionTree_ctgr':decisionTree_ctgr, 'dtOutcome':dtOutcome}
    #     # np.savetxt('band_dt_' + data_file_name + '_' + algorithmType + '_' + str(level) + '_unknownAsAttr.csv', [decisionTree], delimiter=',', fmt='%s')
    # else:    
    #     pd.concat([pd.DataFrame(prdctnError)]).to_csv(
    #                    'bank_' + data_file_name + '_adaBoost_error_' + 
    #                    str(T) + '_unknownNotAttr.csv', index = True, header = True)
    #     # decisionTree = {'decisionTree_attr':decisionTree_attr, 'decisionTree_ctgr':decisionTree_ctgr, 'dtOutcome':dtOutcome}
    #     # np.savetxt('bank_dt_' + data_file_name + '_' + algorithmType + '_' + str(level) + '_unknownNotAttr.csv', [decisionTree], delimiter=',', fmt='%s')

            
    
    
    

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

def pickAttribute(trainingData, avail_attributes, weightData):
    ### Local Variables
    data_lngth          = np.shape(trainingData)[0]
    total_attributes    = len(avail_attributes)
    attributes_infoGain = np.zeros((total_attributes,1))
            

    ### Calculate Total Entropy/GiniIndex/MajorityError
    label_ctgrs = np.unique(trainingData[:,total_attributes])
    label_wghtd_cnt = [round(np.sum(weightData[np.where(trainingData[:,total_attributes] == label_ctgrs[i])]), 10) for i in range(len(label_ctgrs))]
    
    total_info = calcInformationGain(label_wghtd_cnt, sum(label_wghtd_cnt))
    
    

    ### Calculate Entropy/GiniIndex/MajorityError for Each Attribute
    for attrX in np.arange(0, total_attributes):
        attr_ctgrs, attr_cnt = np.unique(trainingData[:,attrX], return_counts=1)
        
        attr_wghtd_cnt = [round(np.sum(weightData[np.where(trainingData[:,attrX] == attr_ctgrs[i])]), 10) for i in range(len(attr_ctgrs))]
        
        ### Create Array for Info Loss For Each Attribute's Category
        attr_ctgrs_infoLoss = np.zeros((len(attr_ctgrs), 1))
        
        
        ### Loop Through Each Attribute's Categories
        for attr_ctgrsX in np.arange(0, len(attr_ctgrs)):
            attr_ctgrs_idx = [i for i in range(data_lngth) if np.array_equal(trainingData[i, attrX], attr_ctgrs[attr_ctgrsX])] # pull label for one attr category
            label_ctgrs  = np.unique(trainingData[attr_ctgrs_idx, total_attributes])
            
            weightDataX = weightData[attr_ctgrs_idx] # pull weights for attr of one label
            label_wghtd_cnt = [round(np.sum(weightDataX[np.where(trainingData[attr_ctgrs_idx, total_attributes] == label_ctgrs[i])]), 10) / attr_wghtd_cnt[attr_ctgrsX] for i in range(len(label_ctgrs))]
            
            attr_ctgrs_infoLoss[attr_ctgrsX] = calcInformationGain(label_wghtd_cnt, attr_wghtd_cnt[attr_ctgrsX]) * (attr_wghtd_cnt[attr_ctgrsX])
            
            
        ### Calculate Expected Value 
        attributes_infoGain[attrX] = total_info - np.sum(attr_ctgrs_infoLoss)
        
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
                xx = xx - ((counts[idx])*np.log(counts[idx]))
        
    elif algorithmType == 'GiniIndex':
        for idx in np.arange(0, length): 
            if total != 0:
                xx = xx + (counts[idx])**2
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
    for rows in decisionTree_attr[:,branchX]:
        ### If the Variable is Empty, Skip It. The Branch Has Reached Its End
        if rows == '':
            continue

        idx = labels.index(rows)
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

def mostLikelyOutcome(decisionTree_attr, decisionTree_ctgr, trainData, weightx):
    ### Preset Variables
    total_attributes = np.shape(trainData)[1]-1
    data_lngth = np.shape(trainData)[0]
    dtOutcome = np.zeros([np.shape(trainData)[0],1], dtype=object)
    
    for idx in range(0, np.shape(decisionTree_attr)[1]):
        ## Calculate the Most Likely Outcome
        used_attributes, avail_attributes = whichAttributes(decisionTree_attr, idx)
        decisionTree_ctgrX = decisionTree_ctgr[:,idx]
        
        decision_branch_idx = [i for i in range(data_lngth) if 
                              np.array_equal(trainData[i, used_attributes], decisionTree_ctgrX[decisionTree_ctgrX != ''])]
        
        outcome_ctgrs = np.unique(trainData[decision_branch_idx,len(labels)-1])
        
        outcome_wghtd_cnt = [round(np.sum(weightx[np.where(trainData[decision_branch_idx, total_attributes] == outcome_ctgrs[i])]), 10) for i in range(len(outcome_ctgrs))]
        
        if len(outcome_wghtd_cnt) == 0:
            dtOutcome[decision_branch_idx] = ''
            # dtOutcome = np.concatenate([dtOutcome, np.array('', ndmin=1)])
        else:
            dtOutcome[decision_branch_idx] = outcome_ctgrs[int(np.argmax(outcome_wghtd_cnt, axis = 0))]
            # dtOutcome = np.concatenate([dtOutcome, np.array(outcome_ctgrs[int(np.argmax(outcome_wghtd_cnt, axis = 0))], ndmin=1)])
        
    return dtOutcome




#%% Calculate Average Prediction Error

def avgPredictionError(trainData, decisionTree_attr, decisionTree_ctgr, dtOutcome, weightx):
    # data_lngth = np.shape(trainData)[0]
    total_attributes = np.shape(trainData)[1]-1
    # branches = np.shape(decisionTree_attr)[1]
    # errors = 0
    # incorrectPredict = np.zeros([0], dtype=int)
    
    # ### loop through each branch and count incorrect predictions
    # for idx in range(0, branches):
    #     dt_ctgr_branch = decisionTree_ctgr[:,idx]
        # 
    #     used_attributes, avail_attributes = whichAttributes(decisionTree_attr, idx)
        
    #     attr_ctgrs_idx          = [i for i in range(data_lngth) if 
    #                       np.array_equal(trainData[i, used_attributes], dt_ctgr_branch)]
        
    #     incorrectPredIdx = np.asarray(np.where(trainData[attr_ctgrs_idx, total_attributes] != dtOutcome[idx]), dtype=int).flatten()
    #     attr_ctgrs_idx = np.asarray(attr_ctgrs_idx)
    #     incorrectPredict = np.concatenate([incorrectPredict, attr_ctgrs_idx[incorrectPredIdx]])
        
        # errors += sum(weight[incorrectPredict])
    
    errors = np.sum(weightx[np.where(trainData[:,total_attributes] != dtOutcome)])
    print('Errors: ' + str(errors))
    amountOfSay = 0.5 * (np.log((1-errors)/(errors)))
    print('Amount of Say: ' + str(amountOfSay))
        
    for i in [np.where(trainData[:,total_attributes] != dtOutcome)]:
        weightx[i] = weightx[i] * np.exp(amountOfSay)
    for i in [np.where(trainData[:,total_attributes] == dtOutcome)]:
        weightx[i] = weightx[i] * np.exp(-amountOfSay)
    weightx = weightx / sum(weightx)
         
    
    return errors, weightx, amountOfSay


    
#%% MAIN
main()



