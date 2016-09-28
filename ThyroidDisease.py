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
from sklearn.feature_selection import RFE
from sklearn.feature_selection import chi2, f_classif
from sklearn.preprocessing import Binarizer, scale
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFromModel

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
    {'kernel':['linear'], 'C':[0.01,0.1,1,10]},
    {'kernel':['rbf'], 'C':[0.1,1,10], 'gamma':[0.01,0.001,0.0001]},
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
    {'n_estimators':[5,10,15,20,25,30], 'bootstrap':[True, False], 'bootstrap_features':[True, False]}
    ]

    grid = grid_search.GridSearchCV(BaggingClassifier(base_estimator=decisionTree), param_grid_bg, scoring='accuracy', n_jobs=-1, cv=gs_nFolds)
    grid.fit(X, Y)

    print('best_params - Bagging:', dt_best_params_, grid.best_params_)

    return grid.best_estimator_



def gridSearchParameters_RandomForestClassifier(X, Y, gs_nFolds):   
    #Odredjivanje parametara za Random Forests
    param_grid_rf = [
    {'n_estimators':[5,10,15,20,25,30], 'criterion':['gini', 'entropy'], 'max_features':['sqrt', 'log2', None], 'bootstrap':[True, False], 'n_jobs':[-1]}   
    ]

    grid = grid_search.GridSearchCV(RandomForestClassifier(), param_grid_rf, scoring='accuracy', n_jobs=-1, cv=gs_nFolds)
    grid.fit(X, Y)

    print('best_params - RandomForests:', grid.best_params_)

    return grid.best_estimator_



def gridSearchParameters_GradientBoostingClassifier(X, Y, gs_nFolds):   
    #Odredjivanje parametara za Gradient Boosting
    param_grid_gb = [
    {'learning_rate':[0.1, 0.01, 0.001], 'n_estimators':[40,50,60,70], 'max_depth':[3,4,5,6]}
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
    #Prikaz uporednih vrednosti preko Box plot-a za score accuracy dobijenih preko cross validacije    
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



def classPiePlot(dataFrameClass):
    #Crta pie plot kojim se predstavlja procentualni odnos instanca po klasi
    dictClass = {}
    i=1
    while i <= 3:
        dictClass[i] = 0
        i = i + 1
    
    for i in dataFrameClass:
        dictClass[i] = dictClass[i] + 1
    
    x = []
    i=1
    while i <= 3:
        x.append(dictClass[i])
        i = i + 1
       
    fig = plt.gcf()
    fig.set_size_inches(3, 3)
    plt.pie(x,labels=['normal','hyper','hypo'] , autopct='%1.2f%%', colors=['lightblue','tan','gray'])
    plt.show()
    #end classPiePlot



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


#Ucitavanje podataka
thyroidDisease = pd.read_csv('Diplomski_ispit/data/new-thyroid.csv', delimiter=',')


#Kreiranje korelacione matrice
thyroidDisease_corr = thyroidDisease[['T3-resin(percentage)', 'Thyroxin(Isotopic)', 'Triiodothyronine(radioimmuno)', 'TSH(radioimmuno)', 'TSH(after_injection)', 'Class']].corr()


#Procentualni odnos instanca po klasi
print('\nProcentualni odnos instanca po klasi')
classPiePlot(thyroidDisease['Class'])


#Postavljanje vrednosti X (predictor) i Y (response)
X = thyroidDisease[['T3-resin(percentage)', 'Thyroxin(Isotopic)', 'Triiodothyronine(radioimmuno)', 'TSH(radioimmuno)', 'TSH(after_injection)']]
Y = thyroidDisease['Class']


########################################
#Ukljuceni su svi atributi

print('\nSvi atributi su ukljuceni u obradu')

print('\nGrid Search - najbolji parametri')

#Broj fold-ova Grid Search cross validacije
gs_nFolds = 10

#Gaussian Naive Bayes
naiveBayesG = GaussianNB()

#K Nearest Neighbors
kNeighbors = gridSearchParameters_KNeighborsClassifier(X, Y, gs_nFolds)

#Support Vector Machines
svm_svc = gridSearchParameters_SVM_SVC(X, Y, gs_nFolds)


#Ensemble Methods

#Bagging
bagging = gridSearchParameters_BaggingClassifier(X, Y, gs_nFolds)

#Random Forests
randomForest = gridSearchParameters_RandomForestClassifier(X, Y, gs_nFolds)

#Gradient Boosting 
gradientBoosting = gridSearchParameters_GradientBoostingClassifier(X, Y, gs_nFolds)


scoreAll = {}
cv_nFolds = 10
crossValidationScore(X, Y, scoreAll, cv_nFolds, ('NaiveBayesG', naiveBayesG), ('k-NN', kNeighbors), ('SVM', svm_svc), ('Bagging', bagging), ('RandomForests', randomForest), ('GradientBoosting', gradientBoosting) )
crossValidationBoxplot(scoreAll, 10, 3, 'NaiveBayesG', 'k-NN', 'SVM', 'Bagging', 'RandomForests', 'GradientBoosting')




########################################
#Feature Selection
print('\n\nFeature selection\n')

#Dataframe u kome se nalaze rezultati metoda za selekciju atributa
ranking = pd.DataFrame(X.columns.values, columns=['columns'])



#Feature importance za Gradient Boosting
print('\nFeature importance za Gradient Boosting')

gradientBoosting.fit(X,Y)
gradientBoosting_score = (gradientBoosting.feature_importances_ * 100)

print('Feature ranking:', gradientBoosting_score.round(decimals=2))
featureRankingPlot(X.columns.values, gradientBoosting_score, 'Gradient Boosting', 'Feature importance [%]')


ranking = pd.concat( [ ranking, pd.Series( sortScore(gradientBoosting_score ) , name = 'gradientBoosting') ], axis=1  )



#chi2
print('\nchi2')

X_bin = Binarizer().fit_transform(scale(X))
selectK_chi2 = SelectKBest(chi2, k='all')
selectK_chi2.fit(X_bin,Y)

print(selectK_chi2.scores_.round(decimals=2))
featureRankingPlot(X.columns.values, selectK_chi2.scores_, 'chi-squared stats', 'Scores of features')


ranking = pd.concat( [ ranking, pd.Series(sortScore(selectK_chi2.scores_) , name = 'chi2') ], axis=1  )



#f_classif
print('\nf_classif')

selectK_f_classif = SelectKBest(f_classif, k='all')
selectK_f_classif.fit(X,Y)

print(selectK_f_classif.scores_.round(decimals=2))
featureRankingPlot(X.columns.values, selectK_f_classif.scores_, 'ANOVA F-value', 'Scores of features')


ranking = pd.concat( [ ranking, pd.Series(sortScore(selectK_f_classif.scores_) , name = 'F-value') ], axis=1  )



#Recursive feature elimination
print('\n\nRecursive feature elimination\n')

svm_svc_linear =  gridSearchParameters_SVM_Linear(X, Y, gs_nFolds)
crossValidationScore(X, Y, {}, 10, ('svm_svc_linear', svm_svc_linear) )

rfe = RFE(svm_svc_linear,n_features_to_select=1)
rfe.fit(X, Y)

print('\nFeature ranking:', rfe.ranking_)


ranking = pd.concat( [ ranking, pd.Series(rfe.ranking_ , name = 'RFE') ], axis=1  )



#L1-based feature selection
print('\n\nL1-based feature selection\n')

svm_l1 = gridSearchParameters_SVM_L1(X, Y, gs_nFolds)
crossValidationScore(X, Y, {}, 10, ('svm_l1', svm_l1) )

svm_l1.fit(X, Y)
model = SelectFromModel(svm_l1, prefit=True)
l1 = model.get_support()
print('\nl1 feature selection:', l1)

ranking = pd.concat( [ ranking, pd.Series(l1 , name = 'L1-SVM') ], axis=1  )



print('\n\nSelekcija atributa - uporedni rezultati\n')
print(ranking)



#Analiza sa selektovanim atributima

print('\n\nSelektovani su atributi: \'Thyroxin(Isotopic)\' i \'TSH(after_injection)\'\n ')
X_FS = thyroidDisease[['Thyroxin(Isotopic)', 'TSH(after_injection)']]

print('\nGrid Search - najbolji parametri')

gs_nFolds = 10

#Gaussian Naive Bayes
naiveBayesG = GaussianNB()

#K Nearest Neighbors
kNeighbors = gridSearchParameters_KNeighborsClassifier(X_FS, Y, gs_nFolds)

#Support Vector Machines
svm_svc = gridSearchParameters_SVM_SVC(X_FS, Y, gs_nFolds)


#Ensemble Methods

#Bagging
bagging = gridSearchParameters_BaggingClassifier(X_FS, Y, gs_nFolds)

#Random Forests
randomForest = gridSearchParameters_RandomForestClassifier(X_FS, Y, gs_nFolds)

#Gradient Boosting 
gradientBoosting = gridSearchParameters_GradientBoostingClassifier(X_FS, Y, gs_nFolds)
#gradientBoosting = GradientBoostingClassifier()


scoreFS = {}
cv_nFolds = 10
crossValidationScore(X_FS, Y, scoreFS, cv_nFolds, ('NaiveBayesG', naiveBayesG), ('k-NN', kNeighbors), ('SVM', svm_svc), ('Bagging', bagging), ('RandomForests', randomForest), ('GradientBoosting', gradientBoosting) )
crossValidationBoxplot(scoreFS, 10, 3, 'NaiveBayesG', 'k-NN', 'SVM', 'Bagging', 'RandomForests', 'GradientBoosting')


########################################
#Uporedne vrednosti

print('\nUporedne vrednosti pre i nakon selekcije atributa')

print('\nUporedne vrednosti za Naive Bayes, K Nearest Neighbors i SVM')
crossValidationCompareBoxplot(scoreAll, scoreFS, 'Svi atributi', 'Selektovani atributi', 10, 3.1, 'NaiveBayesG', 'k-NN', 'SVM')

print('\nUporedne vrednosti za Ensemble metode')
crossValidationCompareBoxplot(scoreAll, scoreFS, 'Svi atributi', 'Selektovani atributi', 10, 3.1, 'Bagging', 'RandomForests', 'GradientBoosting')
