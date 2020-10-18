# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 23:54:03 2020

@author: ctlian
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.metrics import recall_score, precision_score, accuracy_score, confusion_matrix

def train_model(clf, name, X_train, y_train, X_test, y_test, true_label, false_label):
    clf = clf.fit(X_train, y_train)
    
    train_pred = clf.predict(X_train)
    test_pred = clf.predict(X_test)
    
    print ("{} ".format(clf))
    print ('Train Result:')
    print ('   Accuracy:',accuracy_score(train_pred, y_train))
    print ('   Recall:',recall_score(train_pred, y_train),)
    print ('   Precision:',precision_score(train_pred, y_train))
    print ('Test Result:')
    print ('   Accuracy:',accuracy_score(test_pred, y_test))
    print ('   Recall:',recall_score(test_pred, y_test),)
    print ('   Precision:',precision_score(test_pred, y_test))
    plot_confusion_matrix(confusion_matrix(train_pred, y_train), name, 'Train', true_label, false_label)
    plot_confusion_matrix(confusion_matrix(test_pred, y_test), name, 'Test', true_label, false_label)
    
    return clf

    
def plot_confusion_matrix(cm, name, mode, true_label, false_label):
    print(cm)
    fig, ax = plt.subplots(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt='d',xticklabels=[false_label, true_label],yticklabels=[false_label, true_label])
    plt.title(name + '-' + mode + '-' +'Confusion matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

data = pd.read_csv('out_data.txt', header=None) #ReadFrequencyData('out_data.txt')
data_frequency = data.iloc[:,1:]
ground_truth = pd.read_csv('simple_ground_truth.txt', header=None)
ground_truth = np.array(ground_truth).reshape((len(ground_truth),))

#Step 1
X = data_frequency
y = ground_truth.copy()
y[y != 'Unknown'] = 0
y[y == 'Unknown'] = 1
y = y.astype(int)

X_train, X_test, y_train, y_test  =  train_test_split(X,y,test_size=0.25)
rf = RandomForestClassifier(n_estimators= 15, max_depth= 5, class_weight='balanced')
rf_layer1 = train_model(rf, 'Random Forest', X_train, y_train, X_test, y_test, 'Unknown', 'Others')

dt = tree.DecisionTreeClassifier(max_depth = 3)
dt_layer1 = train_model(dt, 'Decision Tree', X_train, y_train, X_test, y_test, 'Unknown', 'Others')

#Step 2
X = data_frequency[ground_truth!='Unknown']
y = ground_truth[ground_truth!='Unknown']
y[y != 'Non Smoker'] = 0
y[y == 'Non Smoker'] = 1
y = y.astype(int)

X_train,  X_test,  y_train, y_test  =  train_test_split(X,y,test_size=0.25)
rf = RandomForestClassifier(n_estimators= 15, max_depth= 5, class_weight='balanced')
rf_layer2 = train_model(rf, 'Random Forest', X_train, y_train, X_test, y_test, 'Non-Smoker', 'Smoker')

dt = tree.DecisionTreeClassifier(max_depth = 5)
dt_layer2 = train_model(dt, 'Decision Tree',X_train, y_train, X_test, y_test, 'Non-Smoker', 'Smoker')

#Step 3
X = data_frequency[(ground_truth!='Unknown') & (ground_truth!='Non Smoker')]
y = ground_truth[(ground_truth!='Unknown') & (ground_truth!='Non Smoker')]
y[y == 'Past Smoker'] = 0
y[y == 'Current Smoker'] = 1
y = y.astype(int)

X_train,  X_test,  y_train, y_test  =  train_test_split(X,y,test_size=0.25)
rf = RandomForestClassifier(n_estimators= 15, max_depth= 3)
rf_layer3 = train_model(rf, 'Random Forest',X_train, y_train, X_test, y_test, 'Current-Smoker', 'Past-Smoker')

dt = tree.DecisionTreeClassifier(max_depth = 3)
dt_layer3 = train_model(dt, 'Decision Tree',X_train, y_train, X_test, y_test, 'Current-Smoker', 'Past-Smoker')


##### predict test data #####
test_data = pd.read_csv('test_data.txt', header=None)
test_frequency = test_data.iloc[:,1:]
rf_result = pd.DataFrame(np.zeros((len(test_data),2)), columns=['filename', 'pred'])
rf_result['filename'] = test_data.iloc[:,0]
dt_result = pd.DataFrame(np.zeros((len(test_data),2)), columns=['filename', 'pred'])
dt_result['filename'] = test_data.iloc[:,0]

pred = rf_layer1.predict(test_frequency)
rf_result['layer1'] = pred
pred = rf_layer2.predict(test_frequency)
rf_result['layer2'] = pred
pred = rf_layer3.predict(test_frequency)
rf_result['layer3'] = pred

rf_result.loc[(rf_result['layer1']==1) & (rf_result['pred']==0), 'pred'] = 'Unknown'
rf_result.loc[(rf_result['layer2']==1) & (rf_result['pred']==0), 'pred'] = 'Non-Smoker'
rf_result.loc[(rf_result['layer3']==1) & (rf_result['pred']==0), 'pred'] = 'Current-Smoker'
rf_result.loc[(rf_result['layer3']==0) & (rf_result['pred']==0), 'pred'] = 'Past-Smoker'

rf_result.iloc[:,:2].to_csv('rf_result.txt', sep='\t', index=False)


pred = dt_layer1.predict(test_frequency)
dt_result['layer1'] = pred
pred = dt_layer2.predict(test_frequency)
dt_result['layer2'] = pred
pred = dt_layer3.predict(test_frequency)
dt_result['layer3'] = pred

dt_result.loc[(dt_result['layer1']==1) & (dt_result['pred']==0), 'pred'] = 'Unknown'
dt_result.loc[(dt_result['layer2']==1) & (dt_result['pred']==0), 'pred'] = 'Non-Smoker'
dt_result.loc[(dt_result['layer3']==1) & (dt_result['pred']==0), 'pred'] = 'Current-Smoker'
dt_result.loc[(dt_result['layer3']==0) & (dt_result['pred']==0), 'pred'] = 'Past-Smoker'

dt_result.iloc[:,:2].to_csv('dt_result.txt', sep='\t', index=False)

print('same predict count: ', sum(dt_result['pred']==rf_result['pred']))
