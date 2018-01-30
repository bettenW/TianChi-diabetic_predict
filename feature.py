# -*- coding: utf-8 -*-

import time
import datetime
import numpy as np
import pandas as pd
from numpy import log1p
from sklearn.preprocessing import FunctionTransformer
from scipy.stats import skew
from scipy.special import boxcox1p
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from collections import Counter

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

def best_models(category_X, category_y):
    kfold = StratifiedKFold(n_splits=10)
    #ExtraTrees 
    ExtC = ExtraTreesClassifier()


    ## Search grid for optimal parameters
    ex_param_grid = {"max_depth": [None],
                  "max_features": [1, 3, 10],
                  "min_samples_split": [2, 3, 10],
                  "min_samples_leaf": [1, 3, 10],
                  "bootstrap": [False],
                  "n_estimators" :[100,300],
                  "criterion": ["gini"]}


    gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

    gsExtC.fit(category_X,category_y)

    ExtC_best = gsExtC.best_estimator_

    # Best score
    gsExtC.best_score_

    # RFC Parameters tunning 
    RFC = RandomForestClassifier()


    ## Search grid for optimal parameters
    rf_param_grid = {"max_depth": [None],
                  "max_features": [1, 3, 10],
                  "min_samples_split": [2, 3, 10],
                  "min_samples_leaf": [1, 3, 10],
                  "bootstrap": [False],
                  "n_estimators" :[100,300],
                  "criterion": ["gini"]}


    gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

    gsRFC.fit(category_X,category_y)

    RFC_best = gsRFC.best_estimator_

    # Best score
    gsRFC.best_score_

    # Gradient boosting tunning

    GBC = GradientBoostingClassifier()
    gb_param_grid = {'loss' : ["deviance"],
                  'n_estimators' : [100,200,300],
                  'learning_rate': [0.1, 0.05, 0.01],
                  'max_depth': [4, 8],
                  'min_samples_leaf': [100,150],
                  'max_features': [0.3, 0.1] 
                  }

    gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

    gsGBC.fit(category_X,category_y)

    GBC_best = gsGBC.best_estimator_

    # Best score
    gsGBC.best_score_

    return ExtC_best, RFC_best, GBC_best

def fuck_predict(category_X, category_y, test_X, train):
    
    ExtC_best, RFC_best, GBC_best = best_models(category_X, category_y)
    # Cross validate model with Kfold stratified cross val
    kfold = StratifiedKFold(n_splits=10)
    
    # Modeling step Test differents algorithms 
    random_state = 2
    # classifiers = []
    # classifiers.append(RandomForestClassifier(random_state=random_state))
    # classifiers.append(ExtraTreesClassifier(random_state=random_state))
    # classifiers.append(GradientBoostingClassifier(random_state=random_state))

    mlp = MLPClassifier(random_state=random_state)
    knn = KNeighborsClassifier()
    ld = LinearDiscriminantAnalysis()
    lr = LogisticRegression()
    
    real_res = pd.read_csv('./data/d_answer_a_20180128.csv', encoding='gb2312')
    real_label = pd.DataFrame()
    real_label['label'] = 0
    real_label['label'] = real_res['label']
    real_label[real_label['label']<6.5]=0
    real_label[real_label['label']>=6.5]=1
    
    votingC = VotingClassifier(estimators=[('rfc', RFC_best), ('extc', ExtC_best),
                                           ('gbc',GBC_best), ('mlp', mlp), ('knn', knn), ('ld', ld), ('lr', lr)], voting='soft', n_jobs=4)
    votingC = votingC.fit(category_X, category_y)

    # classifier.fit(category_X, category_y)
    pred_proba = votingC.predict_proba(test_X)
    pred_label= pd.Series(np.argmax(pred_proba,axis = 1))
    count = pred_label[pred_label==real_label['label']].count()
    
    print(count/1000)
    #训练模型
    # category_model = LogisticRegression()
    # category_model.fit(category_X, category_y)

    #预测
    # pred_proba = category_model.predict_proba(test_X)
    # pred_label = pd.Series(np.argmax(pred_proba,axis = 1))


    #拼接,将测试集预测出来的标签合并训练集标签, test_X合并category_X
    concat_y = category_y.append(pred_label,ignore_index =True )
    concat_X = category_X.append(test_X,ignore_index =True)
    #再次训练回归,并进行预测
    #category_model.fit(concat_X,concat_y)
    pred_proba = votingC.predict_proba(test_X)
    train = train.drop(['fuck'], axis=1)
    pred_train_proba = votingC.predict_proba(train)

    return concat_X, concat_y, pred_proba, pred_train_proba 

def fuck_columns(train, test, threshold):
    #根据血糖值来划分数据集,阈值设为6.5
    fuck_columns=train.columns.tolist()
    if fuck_columns.__contains__('血糖'):
        fuck_columns.remove('血糖')
    
    #根据阈值大小，构造两类训练集
    bigger_thr = train[train['血糖']>=threshold]
    bigger_thr_X=bigger_thr.drop(['血糖'],axis=1)
    bigger_thr_y=bigger_thr['血糖']
  
    less_thr=train[train['血糖']<threshold]
    less_thr_X=less_thr.drop(['血糖'],axis=1)
    less_thr_y=less_thr['血糖']
    
    print(len(bigger_thr), len(less_thr))
    #增加fuck标签,大于阈值为1,小于为0
    train['fuck']=((train['血糖']>=threshold)+0)
    #用作fuck类别预测
    category_X=train[fuck_columns]
    category_y=train['fuck']

    test_X=test[fuck_columns]
    train = train.drop(['血糖'], axis=1)
    
    #训练test中fuck标签分类概率
    concat_X, concat_y, pred_proba, pred_train_proba = fuck_predict(category_X, category_y, test_X, train)
    

    #将训练出来的fuck label当作特征放进去
    bigger_thr_X = pd.concat([bigger_thr_X, concat_y[bigger_thr_X.index]], axis = 1)
    less_thr_X = pd.concat([less_thr_X, concat_y[less_thr_X.index]], axis = 1)

    #测试集的拼接

    test_label = pd.DataFrame(concat_y[len(train):].reset_index())
    test_label.drop('index',axis = 1,inplace =True)
    test_concat = pd.concat([test_X,test_label],axis = 1)
    
    #返回根据阈值划分好的两类训练集和血糖值, 测试集拼接血糖分类标签, 不带血糖和fuck的训练集, 测试集血糖概率, 训练集血糖概率
    return bigger_thr_X, bigger_thr_y, less_thr_X, less_thr_y, test_concat, train, pred_proba, pred_train_proba

train = pd.read_csv('cleaned_train20180129_102513.csv')
test = pd.read_csv('cleaned_test20180129_102513.csv')
    
test = test.drop(['id'], axis=1)
train = train.drop(['id'], axis=1)
y_train = train['血糖']

#pred_proba为测试集血糖权重
threshold = 6.5
test_num = len(test)
train_num = len(train)
bigger_thr_X, bigger_thr_y, less_thr_X, less_thr_y, test_concat, X_train, pred_proba, pred_train_proba=fuck_columns(train, test, threshold)