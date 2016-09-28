# -*- coding: utf-8 -*-
"""
@author: Stefan Vučić 12428
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn import grid_search
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2, f_classif
from sklearn.preprocessing import Binarizer, scale
from sklearn.feature_selection import SelectKBest


########################################
#DEFINISANJE FUNKCIJA


def gridSearchParameters_KNeighborsClassifier(X, Y, gs_nFolds):   
    #Odredjivanje parametara za K Nearest Neighbors
    
    k = np.arange(10)+1
    param_grid_knn = [ 
    {'n_neighbors':k.tolist(), 'weights':['uniform', 'distance'], 'algorithm':['ball_tree', 'kd_tree', 'brute']}
    ]
    
    grid = grid_search.GridSearchCV(KNeighborsClassifier(), param_grid_knn, scoring='accuracy', n_jobs=-1, cv=gs_nFolds)
    grid.fit(X, Y)
   
    print('best_params - KNearestNeighbors:', grid.best_params_)

    return grid.best_estimator_



def gridSearchParameters_SVM_SVC(X, Y, gs_nFolds):
    #Odredjivanje parametara za Support Vector Machine     
    param_grid_svm = [
    {'kernel':['linear'], 'C':[0.1,1,10]},
    #{'kernel':['rbf'], 'C':[0.1,1,10], 'gamma':[0.01,0.001,0.0001]},
    {'kernel':['poly'], 'C':[0.1,1,10], 'gamma':[0.01,0.001,0.0001], 'degree':[2,3,4]}
    ]

    grid = grid_search.GridSearchCV(SVC(), param_grid_svm, scoring='accuracy', n_jobs=-1, cv=gs_nFolds)
    grid.fit(X, Y)

    print('best_params - SVM:', grid.best_params_)
    
    return grid.best_estimator_



def gridSearchParameters_SVM_Linear(X, Y, gs_nFolds):
    #Odredjivanje parametara za Support Vector Machine sa linearnim kernelom   
    param_grid_svm = [
    {'kernel':['linear'], 'C':[0.1, 1, 10]}
    ]

    grid = grid_search.GridSearchCV(SVC(), param_grid_svm, scoring='accuracy', n_jobs=-1, cv=gs_nFolds)
    grid.fit(X, Y)

    print('best_params - SVM:', grid.best_params_)
    
    return grid.best_estimator_



def gridSearchParameters_SVM_L1(X, Y, gs_nFolds):
    #Odredjivanje parametara za Support Vector Machine sa L1 regularizacijom    
    param_grid_svm = [
    {'penalty':['l1'], 'C':[1,0.1,0.01], 'dual':[False]}
    ]

    grid = grid_search.GridSearchCV(LinearSVC(), param_grid_svm, scoring='accuracy', n_jobs=-1, cv=gs_nFolds)
    grid.fit(X, Y)

    print('best_params - LinearSVC:', grid.best_params_)
    
    return grid.best_estimator_



def gridSearchParameters_BaggingClassifier(X, Y, gs_nFolds):   
    #Odredjivanje parametara za Bagging
    param_grid_dt = [
    {'criterion':['gini', 'entropy']}
    ]
    
    grid = grid_search.GridSearchCV(DecisionTreeClassifier(), param_grid_dt, scoring='accuracy', n_jobs=-1, cv=gs_nFolds)
    grid.fit(X, Y)
    
    dt_best_params_ = grid.best_params_
    decisionTree = grid.best_estimator_
    
 
    param_grid_bg = [
    {'n_estimators':[20,25,30,35,40,45,50,55,60], 'bootstrap':[True, False], 'bootstrap_features':[True, False]}
    ]

    grid = grid_search.GridSearchCV(BaggingClassifier(base_estimator=decisionTree), param_grid_bg, scoring='accuracy', n_jobs=-1, cv=gs_nFolds)
    grid.fit(X, Y)

    print('best_params - Bagging:', dt_best_params_, grid.best_params_)

    return grid.best_estimator_



def gridSearchParameters_RandomForestClassifier(X, Y, gs_nFolds):   
    #Odredjivanje parametara za Random Forests
    param_grid_rf = [
    {'n_estimators':[20,25,30,35,40,45,50,55,60], 'criterion':['gini', 'entropy'], 'max_features':['sqrt', 'log2', None], 'bootstrap':[True, False], 'n_jobs':[-1]}    
    ]

    grid = grid_search.GridSearchCV(RandomForestClassifier(), param_grid_rf, scoring='accuracy', n_jobs=-1, cv=gs_nFolds)
    grid.fit(X, Y)

    print('best_params - RandomForests:', grid.best_params_)

    return grid.best_estimator_



def gridSearchParameters_GradientBoostingClassifier(X, Y, gs_nFolds):   
    #Odredjivanje parametara za Gradient Boosting
    param_grid_gb = [
    {'learning_rate':[0.1, 0.05, 0.025], 'n_estimators':[90,100,110], 'max_depth':[4,5,6]}    
    ]
    
    grid = grid_search.GridSearchCV(GradientBoostingClassifier(), param_grid_gb, scoring='accuracy', n_jobs=-1, cv=gs_nFolds)
    grid.fit(X, Y)

    print('best_params - GradientBoosting:', grid.best_params_)

    return grid.best_estimator_



def crossValidationScore(X, Y, scoreDict, cv_nFolds, *args):
    #Primena cross validacije i prikaz rezultata
    for tup in args:
        scoreDict[tup[0]] = cross_val_score(tup[1], X, Y, scoring='accuracy', cv=cv_nFolds).tolist()
    
    print('\nScore accuracy')
    for key, value in scoreDict.items():
        print('\n',key,':')
        print('mean:', np.mean(value)*100)
        print('std:', np.std(value)*100)
        print('min:', np.min(value)*100)
        print('max:', np.max(value)*100)
        print('range:', (np.max(value) - np.min(value))*100)
    #end CrossValidationScore



def crossValidationBoxplot(scoreDict, figSizeWidth, figSizeHeight, *args):
    #Prikaz preko Box plot-a rezultata dobijenih cross validacijom 
    fig = plt.gcf()
    fig.set_size_inches(figSizeWidth, figSizeHeight)
    
    scoreValue = []
    lb = []
    for key in args:
        scoreValue.append(scoreDict[key])
        lb.append(key)

    bp = plt.boxplot(scoreValue, labels=lb, patch_artist=True, meanline=False, showmeans=True, showcaps=True, showbox=True, showfliers=True, manage_xticks=True)
    
    plt.setp(bp['boxes'], color='black', linewidth=1.0)
    plt.setp(bp['boxes'], facecolor='lightblue')
    plt.setp(bp['medians'], color='black', linewidth=1.0)    
    plt.setp(bp['whiskers'], color='black', linewidth=1.0)
    plt.setp(bp['caps'], color='black', linewidth=1.0)

    plt.grid(True, axis='y')
    
    plt.show()
    #end CrossValidationBoxplot



def crossValidationCompareBoxplot(scoreDict1, scoreDict2, opisScoreDict1, opisScoreDict2, figSizeWidth, figSizeHeight, *args):
    #Prikaz uporednih vrednosti preko Box plot-a   
    fig = plt.gcf()
    fig.set_size_inches(figSizeWidth, figSizeHeight)
    
    scoreValue = []
    lb = []
    box_colours = []
    for key in args:
        scoreValue.append(scoreDict1[key])
        scoreValue.append(scoreDict2[key])
        lb.append(key)
        lb.append(key)
        box_colours.append('lightblue')
        box_colours.append('tan')
    
    bp = plt.boxplot(scoreValue, labels=lb, patch_artist=True, meanline=False, showmeans=True, showcaps=True, showbox=True, showfliers=True, manage_xticks=True)
  
    for colour, box in zip(box_colours, bp['boxes']):
        box.set(color='black', linewidth=1.0 )
        box.set(facecolor=colour )

    plt.setp(bp['medians'], color='black', linewidth=1.0)    
    plt.setp(bp['whiskers'], color='black', linewidth=1.0)
    plt.setp(bp['caps'], color='black', linewidth=1.0)

    plt.grid(True, axis='y')       
    plt.figtext(0.14, 1.03, opisScoreDict1, backgroundcolor='lightblue', color='black')
    plt.figtext(0.14, 0.95, opisScoreDict2, backgroundcolor='tan', color='black')
   
    plt.show()
    #end crossValidationCompareBoxplot



def featureImprotancePlot(ensembleMethod, nazivMetoda):
    feat_imp = pd.Series(ensembleMethod.feature_importances_, X.columns.values).sort_values()
    feat_imp.plot(kind='barh', title=nazivMetoda, color='tan')
    plt.grid(True, axis='x') 
    plt.xlabel('Feature Importance')
    plt.show()
    #end featureImprotancePlot



def classBarhPlot(dataFrameClass):
    dictClass = {}
    i=1
    while i <= 16:
        dictClass[i] = 0
        i = i + 1

    for i in dataFrameClass:
        dictClass[i] = dictClass[i] + 1
    
    x = []
    i=1
    while i <= 16:
        x.append(dictClass[i])
        i = i + 1

    class_series = pd.Series(x, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
    class_series.plot(kind='barh', color='lightblue')
    plt.grid(True, axis='x') 
    plt.show()
    #end classBarhPlot



def brojInstanciPoKlasi(dataFrameClass):
    #Prikazuje broj instanci po klasi
    dictClass = {}
    i=1
    while i <= 16:
        dictClass[i] = 0
        i = i + 1

    for i in dataFrameClass:
        dictClass[i] = dictClass[i] + 1
    
    i=1
    while i <= 16:
        print(i,':',dictClass[i])
        i = i + 1

    #end brojInstanciPoKlasi


def featureRankingPlot(XcolumnsValues, scores, nazivMetode, xlab):
    #Crta dijagram rangiranja atributa na osnovu score vrednosti metoda za selekciju atributa    
    featScores = pd.Series(scores, XcolumnsValues).sort_values()
    featScores.plot(kind='barh', title=nazivMetode, color='tan')
    plt.grid(True, axis='x') 
    plt.xlabel(xlab)
    plt.show()
    #end featureImprotancePlot



def sortScore(scores):
    #Sortira score vrednosti metoda za selekciju atributa    
    sort_score_list = []
    k = 1
    for i in scores:
        sort_score_list.append([i, k])
        k = k + 1
    
    n = len(sort_score_list)
    i = 1
    while i < n:
        k = i
        while k > 0:
            if sort_score_list[k-1][0] < sort_score_list[i][0]:
                sort_score_list[k-1][1] = sort_score_list[k-1][1] + 1
                sort_score_list[i][1] = sort_score_list[i][1] - 1
            k = k - 1
        i = i + 1
    
    i = 0
    while i < len(sort_score_list):
        sort_score_list[i][0] = "%.2f" % sort_score_list[i][0]
        i = i + 1  
    
    return sort_score_list



########################################
#ANALIZA PODATAKA

##########
#Učitavanje podataka


arrhythmia = pd.read_csv('Projekat-Zavrsni_ispit/data/Arrhythmia.csv', delimiter=',')


#Kreiranje korelacione matrice
#arrhythmia_corr = arrhythmia.corr()
 

X = arrhythmia[['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11',
       'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'V29',
       'V30', 'V31', 'V32', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38',
       'V39', 'V40', 'V41', 'V42', 'V43', 'V44', 'V45', 'V46', 'V47',
       'V48', 'V49', 'V50', 'V51', 'V52', 'V53', 'V54', 'V55', 'V56',
       'V57', 'V58', 'V59', 'V60', 'V61', 'V62', 'V63', 'V64', 'V65',
       'V66', 'V67', 'V68', 'V69', 'V70', 'V71', 'V72', 'V73', 'V74',
       'V75', 'V76', 'V77', 'V78', 'V79', 'V80', 'V81', 'V82', 'V83',
       'V84', 'V85', 'V86', 'V87', 'V88', 'V89', 'V90', 'V91', 'V92',
       'V93', 'V94', 'V95', 'V96', 'V97', 'V98', 'V99', 'V100', 'V101',
       'V102', 'V103', 'V104', 'V105', 'V106', 'V107', 'V108', 'V109',
       'V110', 'V111', 'V112', 'V113', 'V114', 'V115', 'V116', 'V117',
       'V118', 'V119', 'V120', 'V121', 'V122', 'V123', 'V124', 'V125',
       'V126', 'V127', 'V128', 'V129', 'V130', 'V131', 'V132', 'V133',
       'V134', 'V135', 'V136', 'V137', 'V138', 'V139', 'V140', 'V141',
       'V142', 'V143', 'V144', 'V145', 'V146', 'V147', 'V148', 'V149',
       'V150', 'V151', 'V152', 'V153', 'V154', 'V155', 'V156', 'V157',
       'V158', 'V159', 'V160', 'V161', 'V162', 'V163', 'V164', 'V165',
       'V166', 'V167', 'V168', 'V169', 'V170', 'V171', 'V172', 'V173',
       'V174', 'V175', 'V176', 'V177', 'V178', 'V179', 'V180', 'V181',
       'V182', 'V183', 'V184', 'V185', 'V186', 'V187', 'V188', 'V189',
       'V190', 'V191', 'V192', 'V193', 'V194', 'V195', 'V196', 'V197',
       'V198', 'V199', 'V200', 'V201', 'V202', 'V203', 'V204', 'V205',
       'V206', 'V207', 'V208', 'V209', 'V210', 'V211', 'V212', 'V213',
       'V214', 'V215', 'V216', 'V217', 'V218', 'V219', 'V220', 'V221',
       'V222', 'V223', 'V224', 'V225', 'V226', 'V227', 'V228', 'V229',
       'V230', 'V231', 'V232', 'V233', 'V234', 'V235', 'V236', 'V237',
       'V238', 'V239', 'V240', 'V241', 'V242', 'V243', 'V244', 'V245',
       'V246', 'V247', 'V248', 'V249', 'V250', 'V251', 'V252', 'V253',
       'V254', 'V255', 'V256', 'V257', 'V258', 'V259', 'V260', 'V261',
       'V262', 'V263', 'V264', 'V265', 'V266', 'V267', 'V268', 'V269',
       'V270', 'V271', 'V272', 'V273', 'V274', 'V275', 'V276', 'V277',
       'V278', 'V279']]

Y = arrhythmia['Class']


"""
#Bez atributa koji imaju izgubljene vrednosti

X_bezV11doV15 = arrhythmia[['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 
                'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'V29',
       'V30', 'V31', 'V32', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38',
       'V39', 'V40', 'V41', 'V42', 'V43', 'V44', 'V45', 'V46', 'V47',
       'V48', 'V49', 'V50', 'V51', 'V52', 'V53', 'V54', 'V55', 'V56',
       'V57', 'V58', 'V59', 'V60', 'V61', 'V62', 'V63', 'V64', 'V65',
       'V66', 'V67', 'V68', 'V69', 'V70', 'V71', 'V72', 'V73', 'V74',
       'V75', 'V76', 'V77', 'V78', 'V79', 'V80', 'V81', 'V82', 'V83',
       'V84', 'V85', 'V86', 'V87', 'V88', 'V89', 'V90', 'V91', 'V92',
       'V93', 'V94', 'V95', 'V96', 'V97', 'V98', 'V99', 'V100', 'V101',
       'V102', 'V103', 'V104', 'V105', 'V106', 'V107', 'V108', 'V109',
       'V110', 'V111', 'V112', 'V113', 'V114', 'V115', 'V116', 'V117',
       'V118', 'V119', 'V120', 'V121', 'V122', 'V123', 'V124', 'V125',
       'V126', 'V127', 'V128', 'V129', 'V130', 'V131', 'V132', 'V133',
       'V134', 'V135', 'V136', 'V137', 'V138', 'V139', 'V140', 'V141',
       'V142', 'V143', 'V144', 'V145', 'V146', 'V147', 'V148', 'V149',
       'V150', 'V151', 'V152', 'V153', 'V154', 'V155', 'V156', 'V157',
       'V158', 'V159', 'V160', 'V161', 'V162', 'V163', 'V164', 'V165',
       'V166', 'V167', 'V168', 'V169', 'V170', 'V171', 'V172', 'V173',
       'V174', 'V175', 'V176', 'V177', 'V178', 'V179', 'V180', 'V181',
       'V182', 'V183', 'V184', 'V185', 'V186', 'V187', 'V188', 'V189',
       'V190', 'V191', 'V192', 'V193', 'V194', 'V195', 'V196', 'V197',
       'V198', 'V199', 'V200', 'V201', 'V202', 'V203', 'V204', 'V205',
       'V206', 'V207', 'V208', 'V209', 'V210', 'V211', 'V212', 'V213',
       'V214', 'V215', 'V216', 'V217', 'V218', 'V219', 'V220', 'V221',
       'V222', 'V223', 'V224', 'V225', 'V226', 'V227', 'V228', 'V229',
       'V230', 'V231', 'V232', 'V233', 'V234', 'V235', 'V236', 'V237',
       'V238', 'V239', 'V240', 'V241', 'V242', 'V243', 'V244', 'V245',
       'V246', 'V247', 'V248', 'V249', 'V250', 'V251', 'V252', 'V253',
       'V254', 'V255', 'V256', 'V257', 'V258', 'V259', 'V260', 'V261',
       'V262', 'V263', 'V264', 'V265', 'V266', 'V267', 'V268', 'V269',
       'V270', 'V271', 'V272', 'V273', 'V274', 'V275', 'V276', 'V277',
       'V278', 'V279']]

Y_bezV11doV15 = arrhythmia['Class']

print('\nBez atributa koji imaju izgubljene vrednosti')


#Class Distribution
print('\nClass Distribution')
classBarhPlot(Y_bezV11doV15)


print('\nNajbolji parametri dobijeni nakon primene Grid Search')


gs_nFolds = 10

#Gaussian Naive Bayes
naiveBayesG = GaussianNB()

#K Nearest Neighbors
kNeighbors = gridSearchParameters_KNeighborsClassifier(X_bezV11doV15, Y_bezV11doV15, gs_nFolds)

#Support Vector Machines
svm_svc = gridSearchParameters_SVM_SVC(X_bezV11doV15, Y_bezV11doV15, gs_nFolds)


#Ensemble Methods

#Bagging
bagging = gridSearchParameters_BaggingClassifier(X_bezV11doV15, Y_bezV11doV15, gs_nFolds)

#Random Forests
randomForest = gridSearchParameters_RandomForestClassifier(X_bezV11doV15, Y_bezV11doV15, gs_nFolds)

#Gradient Boosting 
gradientBoosting = gridSearchParameters_GradientBoostingClassifier(X_bezV11doV15, Y_bezV11doV15, gs_nFolds)
#gradientBoosting = GradientBoostingClassifier()


score_bezV11doV15 = {}
cv_nFolds = 10
crossValidationScore(X_bezV11doV15, Y_bezV11doV15, score_bezV11doV15, cv_nFolds, ('NaiveBayesG', naiveBayesG), ('k-NN', kNeighbors), ('SVM', svm_svc), ('Bagging', bagging), ('RandomForests', randomForest), ('GradientBoosting', gradientBoosting) )
crossValidationBoxplot(score_bezV11doV15, 10, 3, 'NaiveBayesG', 'k-NN', 'SVM', 'Bagging', 'RandomForests', 'GradientBoosting')
"""


"""
#Ukljuceni su svi atributi, izbaceni su redovi u kojima ima izgubljenih vrednosti ?

X_all = arrhythmia[['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11',
       'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'V29',
       'V30', 'V31', 'V32', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38',
       'V39', 'V40', 'V41', 'V42', 'V43', 'V44', 'V45', 'V46', 'V47',
       'V48', 'V49', 'V50', 'V51', 'V52', 'V53', 'V54', 'V55', 'V56',
       'V57', 'V58', 'V59', 'V60', 'V61', 'V62', 'V63', 'V64', 'V65',
       'V66', 'V67', 'V68', 'V69', 'V70', 'V71', 'V72', 'V73', 'V74',
       'V75', 'V76', 'V77', 'V78', 'V79', 'V80', 'V81', 'V82', 'V83',
       'V84', 'V85', 'V86', 'V87', 'V88', 'V89', 'V90', 'V91', 'V92',
       'V93', 'V94', 'V95', 'V96', 'V97', 'V98', 'V99', 'V100', 'V101',
       'V102', 'V103', 'V104', 'V105', 'V106', 'V107', 'V108', 'V109',
       'V110', 'V111', 'V112', 'V113', 'V114', 'V115', 'V116', 'V117',
       'V118', 'V119', 'V120', 'V121', 'V122', 'V123', 'V124', 'V125',
       'V126', 'V127', 'V128', 'V129', 'V130', 'V131', 'V132', 'V133',
       'V134', 'V135', 'V136', 'V137', 'V138', 'V139', 'V140', 'V141',
       'V142', 'V143', 'V144', 'V145', 'V146', 'V147', 'V148', 'V149',
       'V150', 'V151', 'V152', 'V153', 'V154', 'V155', 'V156', 'V157',
       'V158', 'V159', 'V160', 'V161', 'V162', 'V163', 'V164', 'V165',
       'V166', 'V167', 'V168', 'V169', 'V170', 'V171', 'V172', 'V173',
       'V174', 'V175', 'V176', 'V177', 'V178', 'V179', 'V180', 'V181',
       'V182', 'V183', 'V184', 'V185', 'V186', 'V187', 'V188', 'V189',
       'V190', 'V191', 'V192', 'V193', 'V194', 'V195', 'V196', 'V197',
       'V198', 'V199', 'V200', 'V201', 'V202', 'V203', 'V204', 'V205',
       'V206', 'V207', 'V208', 'V209', 'V210', 'V211', 'V212', 'V213',
       'V214', 'V215', 'V216', 'V217', 'V218', 'V219', 'V220', 'V221',
       'V222', 'V223', 'V224', 'V225', 'V226', 'V227', 'V228', 'V229',
       'V230', 'V231', 'V232', 'V233', 'V234', 'V235', 'V236', 'V237',
       'V238', 'V239', 'V240', 'V241', 'V242', 'V243', 'V244', 'V245',
       'V246', 'V247', 'V248', 'V249', 'V250', 'V251', 'V252', 'V253',
       'V254', 'V255', 'V256', 'V257', 'V258', 'V259', 'V260', 'V261',
       'V262', 'V263', 'V264', 'V265', 'V266', 'V267', 'V268', 'V269',
       'V270', 'V271', 'V272', 'V273', 'V274', 'V275', 'V276', 'V277',
       'V278', 'V279']][(arrhythmia.V11 != '?') & (arrhythmia.V12 != '?') & (arrhythmia.V13 != '?')  & (arrhythmia.V14 != '?') & (arrhythmia.V15 != '?')]

Y_all = arrhythmia['Class'][(arrhythmia.V11 != '?') & (arrhythmia.V12 != '?') & (arrhythmia.V13 != '?')  & (arrhythmia.V14 != '?') & (arrhythmia.V15 != '?')]


print('\nUkljuceni su svi atributi i iskljuceni redovi sa ?')


#Class Distribution
print('\nClass Distribution')
classBarhPlot(Y_all)


print('\nNajbolji parametri dobijeni nakon primene Grid Search')

gs_nFolds = 10

#Gaussian Naive Bayes
naiveBayesG = GaussianNB()

#K Nearest Neighbors
kNeighbors = gridSearchParameters_KNeighborsClassifier(X_all, Y_all, gs_nFolds)

#Support Vector Machines
svm_svc = gridSearchParameters_SVM_SVC(X_all, Y_all, gs_nFolds)


#Ensemble Methods

#Bagging
bagging = gridSearchParameters_BaggingClassifier(X_all, Y_all, gs_nFolds)

#Random Forests
randomForest = gridSearchParameters_RandomForestClassifier(X_all, Y_all, gs_nFolds)

#Gradient Boosting 
gradientBoosting = gridSearchParameters_GradientBoostingClassifier(X_all, Y_all, gs_nFolds)


scoreAll = {}
crossValidationScore(X_all, Y_all, scoreAll, 10, ('NaiveBayesG', naiveBayesG), ('k-NN', kNeighbors), ('SVM', svm_svc), ('Bagging', bagging), ('RandomForests', randomForest), ('GradientBoosting', gradientBoosting) )
crossValidationBoxplot(scoreAll, 10, 3, 'NaiveBayesG', 'k-NN', 'SVM', 'Bagging', 'RandomForests', 'GradientBoosting')
"""


#################################################################

#Iskljucen je atribut V14 koji ima najvise izgubljenih vrednosti
#ukljuceni su atributi V11, V12, V13, V15 koji takodje imaju puno izgubljenih vrednosti
#iskljuceni su redovi koji imaju izgubljene vrednosti ?

X_bezV14 = arrhythmia[['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11',
       'V12', 'V13', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'V29',
       'V30', 'V31', 'V32', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38',
       'V39', 'V40', 'V41', 'V42', 'V43', 'V44', 'V45', 'V46', 'V47',
       'V48', 'V49', 'V50', 'V51', 'V52', 'V53', 'V54', 'V55', 'V56',
       'V57', 'V58', 'V59', 'V60', 'V61', 'V62', 'V63', 'V64', 'V65',
       'V66', 'V67', 'V68', 'V69', 'V70', 'V71', 'V72', 'V73', 'V74',
       'V75', 'V76', 'V77', 'V78', 'V79', 'V80', 'V81', 'V82', 'V83',
       'V84', 'V85', 'V86', 'V87', 'V88', 'V89', 'V90', 'V91', 'V92',
       'V93', 'V94', 'V95', 'V96', 'V97', 'V98', 'V99', 'V100', 'V101',
       'V102', 'V103', 'V104', 'V105', 'V106', 'V107', 'V108', 'V109',
       'V110', 'V111', 'V112', 'V113', 'V114', 'V115', 'V116', 'V117',
       'V118', 'V119', 'V120', 'V121', 'V122', 'V123', 'V124', 'V125',
       'V126', 'V127', 'V128', 'V129', 'V130', 'V131', 'V132', 'V133',
       'V134', 'V135', 'V136', 'V137', 'V138', 'V139', 'V140', 'V141',
       'V142', 'V143', 'V144', 'V145', 'V146', 'V147', 'V148', 'V149',
       'V150', 'V151', 'V152', 'V153', 'V154', 'V155', 'V156', 'V157',
       'V158', 'V159', 'V160', 'V161', 'V162', 'V163', 'V164', 'V165',
       'V166', 'V167', 'V168', 'V169', 'V170', 'V171', 'V172', 'V173',
       'V174', 'V175', 'V176', 'V177', 'V178', 'V179', 'V180', 'V181',
       'V182', 'V183', 'V184', 'V185', 'V186', 'V187', 'V188', 'V189',
       'V190', 'V191', 'V192', 'V193', 'V194', 'V195', 'V196', 'V197',
       'V198', 'V199', 'V200', 'V201', 'V202', 'V203', 'V204', 'V205',
       'V206', 'V207', 'V208', 'V209', 'V210', 'V211', 'V212', 'V213',
       'V214', 'V215', 'V216', 'V217', 'V218', 'V219', 'V220', 'V221',
       'V222', 'V223', 'V224', 'V225', 'V226', 'V227', 'V228', 'V229',
       'V230', 'V231', 'V232', 'V233', 'V234', 'V235', 'V236', 'V237',
       'V238', 'V239', 'V240', 'V241', 'V242', 'V243', 'V244', 'V245',
       'V246', 'V247', 'V248', 'V249', 'V250', 'V251', 'V252', 'V253',
       'V254', 'V255', 'V256', 'V257', 'V258', 'V259', 'V260', 'V261',
       'V262', 'V263', 'V264', 'V265', 'V266', 'V267', 'V268', 'V269',
       'V270', 'V271', 'V272', 'V273', 'V274', 'V275', 'V276', 'V277',
       'V278', 'V279']][(arrhythmia.V11 != '?') & (arrhythmia.V12 != '?') & (arrhythmia.V13 != '?') & (arrhythmia.V15 != '?')]

Y_bezV14 = arrhythmia['Class'][(arrhythmia.V11 != '?') & (arrhythmia.V12 != '?') & (arrhythmia.V13 != '?')  & (arrhythmia.V15 != '?')]


print('\nIskljucen je V14, a izbaceni su redovi gde se nalazi ?')


#Class Distribution
print('\nClass Distribution')
classBarhPlot(Y_bezV14)

print('\nBroj instanci po klasi')
brojInstanciPoKlasi(Y_bezV14)


"""
print('\nNajbolji parametri dobijeni nakon primene Grid Search')

gs_nFolds = 10

#Gaussian Naive Bayes
naiveBayesG = GaussianNB()

#K Nearest Neighbors
kNeighbors = gridSearchParameters_KNeighborsClassifier(X_bezV14, Y_bezV14, gs_nFolds)

#Support Vector Machines
svm_svc = gridSearchParameters_SVM_SVC(X_bezV14, Y_bezV14, gs_nFolds)


#Ensemble Methods

#Bagging
bagging = gridSearchParameters_BaggingClassifier(X_bezV14, Y_bezV14, gs_nFolds)

#Random Forests
randomForest = gridSearchParameters_RandomForestClassifier(X_bezV14, Y_bezV14, gs_nFolds)

#Gradient Boosting 
gradientBoosting = gridSearchParameters_GradientBoostingClassifier(X_bezV14, Y_bezV14, gs_nFolds)
#gradientBoosting = GradientBoostingClassifier()


score_bezV14 = {}
crossValidationScore(X_bezV14, Y_bezV14, score_bezV14, 10, ('NaiveBayesG', naiveBayesG), ('k-NN', kNeighbors), ('SVM', svm_svc), ('Bagging', bagging), ('RandomForests', randomForest), ('GradientBoosting', gradientBoosting) )
crossValidationBoxplot(score_bezV14, 10, 3, 'NaiveBayesG', 'k-NN', 'SVM', 'Bagging', 'RandomForests', 'GradientBoosting')

"""

#Sa 9 klasa i 9 foldova

X_bezV14_9Class = arrhythmia[['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11',
       'V12', 'V13', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'V29',
       'V30', 'V31', 'V32', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38',
       'V39', 'V40', 'V41', 'V42', 'V43', 'V44', 'V45', 'V46', 'V47',
       'V48', 'V49', 'V50', 'V51', 'V52', 'V53', 'V54', 'V55', 'V56',
       'V57', 'V58', 'V59', 'V60', 'V61', 'V62', 'V63', 'V64', 'V65',
       'V66', 'V67', 'V68', 'V69', 'V70', 'V71', 'V72', 'V73', 'V74',
       'V75', 'V76', 'V77', 'V78', 'V79', 'V80', 'V81', 'V82', 'V83',
       'V84', 'V85', 'V86', 'V87', 'V88', 'V89', 'V90', 'V91', 'V92',
       'V93', 'V94', 'V95', 'V96', 'V97', 'V98', 'V99', 'V100', 'V101',
       'V102', 'V103', 'V104', 'V105', 'V106', 'V107', 'V108', 'V109',
       'V110', 'V111', 'V112', 'V113', 'V114', 'V115', 'V116', 'V117',
       'V118', 'V119', 'V120', 'V121', 'V122', 'V123', 'V124', 'V125',
       'V126', 'V127', 'V128', 'V129', 'V130', 'V131', 'V132', 'V133',
       'V134', 'V135', 'V136', 'V137', 'V138', 'V139', 'V140', 'V141',
       'V142', 'V143', 'V144', 'V145', 'V146', 'V147', 'V148', 'V149',
       'V150', 'V151', 'V152', 'V153', 'V154', 'V155', 'V156', 'V157',
       'V158', 'V159', 'V160', 'V161', 'V162', 'V163', 'V164', 'V165',
       'V166', 'V167', 'V168', 'V169', 'V170', 'V171', 'V172', 'V173',
       'V174', 'V175', 'V176', 'V177', 'V178', 'V179', 'V180', 'V181',
       'V182', 'V183', 'V184', 'V185', 'V186', 'V187', 'V188', 'V189',
       'V190', 'V191', 'V192', 'V193', 'V194', 'V195', 'V196', 'V197',
       'V198', 'V199', 'V200', 'V201', 'V202', 'V203', 'V204', 'V205',
       'V206', 'V207', 'V208', 'V209', 'V210', 'V211', 'V212', 'V213',
       'V214', 'V215', 'V216', 'V217', 'V218', 'V219', 'V220', 'V221',
       'V222', 'V223', 'V224', 'V225', 'V226', 'V227', 'V228', 'V229',
       'V230', 'V231', 'V232', 'V233', 'V234', 'V235', 'V236', 'V237',
       'V238', 'V239', 'V240', 'V241', 'V242', 'V243', 'V244', 'V245',
       'V246', 'V247', 'V248', 'V249', 'V250', 'V251', 'V252', 'V253',
       'V254', 'V255', 'V256', 'V257', 'V258', 'V259', 'V260', 'V261',
       'V262', 'V263', 'V264', 'V265', 'V266', 'V267', 'V268', 'V269',
       'V270', 'V271', 'V272', 'V273', 'V274', 'V275', 'V276', 'V277',
       'V278', 'V279']][(arrhythmia.V11 != '?') & (arrhythmia.V12 != '?') & (arrhythmia.V13 != '?') & (arrhythmia.V15 != '?') & (arrhythmia.Class != 7) & (arrhythmia.Class != 8) & (arrhythmia.Class != 14)]

Y_bezV14_9Class = arrhythmia['Class'][(arrhythmia.V11 != '?') & (arrhythmia.V12 != '?') & (arrhythmia.V13 != '?')  & (arrhythmia.V15 != '?') & (arrhythmia.Class != 7) & (arrhythmia.Class != 8) & (arrhythmia.Class != 14)]


print('\nIskljucen je V14, a izbaceni su redovi gde se nalazi ? i iskljucene su klase koje imaju manje od 9 instanci')


#Class Distribution
print('\nClass Distribution')
classBarhPlot(Y_bezV14_9Class)

print('\nBroj instanci po klasi')
brojInstanciPoKlasi(Y_bezV14)


print('\nNajbolji parametri dobijeni nakon primene Grid Search')

nFolds = 9

#Gaussian Naive Bayes
naiveBayesG = GaussianNB()

#K Nearest Neighbors
kNeighbors = gridSearchParameters_KNeighborsClassifier(X_bezV14_9Class, Y_bezV14_9Class, nFolds)

#Support Vector Machines
svm_svc = gridSearchParameters_SVM_SVC(X_bezV14_9Class, Y_bezV14_9Class, nFolds)


#Ensemble Methods

#Bagging
bagging = gridSearchParameters_BaggingClassifier(X_bezV14_9Class, Y_bezV14_9Class, nFolds)

#Random Forests
randomForest = gridSearchParameters_RandomForestClassifier(X_bezV14_9Class, Y_bezV14_9Class, nFolds)

#Gradient Boosting 
gradientBoosting = gridSearchParameters_GradientBoostingClassifier(X_bezV14_9Class, Y_bezV14_9Class, nFolds)
#gradientBoosting = GradientBoostingClassifier()


score_bezV14_9Class = {}
crossValidationScore(X_bezV14_9Class, Y_bezV14_9Class, score_bezV14_9Class, nFolds, ('NaiveBayesG', naiveBayesG), ('k-NN', kNeighbors), ('SVM', svm_svc), ('Bagging', bagging), ('RandomForests', randomForest), ('GradientBoosting', gradientBoosting) )
crossValidationBoxplot(score_bezV14_9Class, 10, 3, 'NaiveBayesG', 'k-NN', 'SVM', 'Bagging', 'RandomForests', 'GradientBoosting')






########################################
#Izbacivanje Features tame gde je varijansa 0 sa VarianceTreshold()
print('\nKoriscenjem VarianceThreshold() iskljucuju se Fetures kod kojih je varijansa jednaka 0')

vt = VarianceThreshold()
vt.fit(X_bezV14_9Class)

X_vt = X_bezV14_9Class[vt.get_support(indices=True).tolist()]


print('\nDimenzije X pre primene VarianceTreshold:', X_bezV14_9Class.shape)

print('\nDimenzije X nakon primene VarianceTreshold:', X_vt.shape)



########################################
#Selektovanje atributa tako sto na osnovu feature importance gradient Boosting metode
#izdvoje atributi ciji znacaj prelazi odredjeni zadati prag...

gradientBoosting = gridSearchParameters_GradientBoostingClassifier(X_vt, Y_bezV14_9Class, nFolds)

gradientBoosting.fit(X_vt, Y_bezV14_9Class)
sfm = SelectFromModel(gradientBoosting, threshold=0.008, prefit=True)

gradientBoosting_scores = sfm.get_support(indices=True).tolist()
X_fi = X_vt[sfm.get_support(indices=True).tolist()]
print(X_fi.shape)


nFolds = 9

#Gaussian Naive Bayes
naiveBayesG = GaussianNB()

#K Nearest Neighbors
kNeighbors = gridSearchParameters_KNeighborsClassifier(X_fi, Y_bezV14_9Class, nFolds)

#Support Vector Machines
svm_svc = gridSearchParameters_SVM_SVC(X_fi, Y_bezV14_9Class, nFolds)


#Ensemble Methods

#Bagging
bagging = gridSearchParameters_BaggingClassifier(X_fi, Y_bezV14_9Class, nFolds)

#Random Forests
randomForest = gridSearchParameters_RandomForestClassifier(X_fi, Y_bezV14_9Class, nFolds)

#Gradient Boosting 
gradientBoosting = gridSearchParameters_GradientBoostingClassifier(X_fi, Y_bezV14_9Class, nFolds)
#gradientBoosting = GradientBoostingClassifier()


score_fi = {}
crossValidationScore(X_fi, Y_bezV14_9Class, score_fi, nFolds, ('NaiveBayesG', naiveBayesG), ('k-NN', kNeighbors), ('SVM', svm_svc), ('Bagging', bagging), ('RandomForests', randomForest), ('GradientBoosting', gradientBoosting) )
crossValidationBoxplot(score_fi, 10, 3, 'NaiveBayesG', 'k-NN', 'SVM', 'Bagging', 'RandomForests', 'GradientBoosting')



########################################
#Uporedne vrednosti

print('\nUporedne vrednosti pre i nakon selekcije atributa')

print('\nUporedne vrednosti za Naive Bayes, K Nearest Neighbors i SVM')
crossValidationCompareBoxplot(score_bezV14_9Class, score_fi, 'Svi atributi', 'Selektovani atributi', 10, 3.1, 'NaiveBayesG', 'k-NN', 'SVM')

print('\nUporedne vrednosti za Ensemble metode')
crossValidationCompareBoxplot(score_bezV14_9Class, score_fi, 'Svi atributi', 'Selektovani atributi', 10, 3.1, 'Bagging', 'RandomForests', 'GradientBoosting')

