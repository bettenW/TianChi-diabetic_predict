# -*- coding: utf-8 -*-

import time
import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression 
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC, Ridge
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
from mlxtend.regressor import StackingRegressor
#用于分类
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

def mse_cv(model, X_train, y_train):
    n_folds = 5
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X_train.values)
    mse= -cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv = kf)
    return(0.5*mse)

def mse(y, y_pred):

    return 0.5*(mean_squared_error(y, y_pred))

#初始分类模型
def fuck_predict(category_X, category_y, test_X, train):

    #训练模型
    category_model = LogisticRegression()
    category_model.fit(category_X, category_y)

    #预测
    pred_proba = category_model.predict_proba(test_X)
    pred_label = pd.Series(np.argmax(pred_proba,axis = 1))
    #拼接,将测试集预测出来的标签合并训练集标签, test_X合并category_X
    concat_y = category_y.append(pred_label,ignore_index =True )
    concat_X = category_X.append(test_X,ignore_index =True)
    #再次训练回归,并进行预测
    category_model.fit(concat_X,concat_y)
    pred_proba = category_model.predict_proba(test_X)
    train = train.drop(['fuck'], axis=1)
    pred_train_proba = category_model.predict_proba(train)

    return concat_X, concat_y, pred_proba, pred_train_proba 

#进化初始模型,进行多种模型融合分类
def fuck_predict_mix(category_X, category_y, test_X):
    
    # Cross validate model with Kfold stratified cross val
    kfold = StratifiedKFold(n_splits=10)
    
    # Modeling step Test differents algorithms 
    random_state = 2
    RFC_best = RandomForestClassifier(random_state=random_state)
    ExtC_best = ExtraTreesClassifier(random_state=random_state)
    GBC_best = GradientBoostingClassifier(random_state=random_state)
    mlp = MLPClassifier(random_state=random_state)
    knn = KNeighborsClassifier()
    ld = LinearDiscriminantAnalysis()
    lr = LogisticRegression()
    #进行投票法
    votingC = VotingClassifier(estimators=[('rfc', RFC_best), ('extc', ExtC_best),
                         ('gbc',GBC_best), ('mlp', mlp), ('knn', knn), ('ld', ld), ('lr', lr)], voting='soft', n_jobs=4)
    votingC = votingC.fit(category_X, category_y)

    # classifier.fit(category_X, category_y)
    pred_proba = votingC.predict_proba(test_X)
    pred_label= pd.Series(np.argmax(pred_proba,axis = 1))
    count = pred_label[pred_label==real_label['label']].count()

    #拼接,将测试集预测出来的标签合并训练集标签, test_X合并category_X
    concat_y = category_y.append(pred_label,ignore_index =True )
    concat_X = category_X.append(test_X,ignore_index =True)
    #再次训练回归,并进行预测
    pred_proba = votingC.predict_proba(test_X)

    return concat_X, concat_y, pred_proba

def fuck_columns(train, test, threshold):
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
    concat_X, concat_y, pred_proba = fuck_predict(category_X, category_y, test_X)

    #将训练出来的fuck label当作特征放进去
    bigger_thr_X = pd.concat([bigger_thr_X, concat_y[bigger_thr_X.index]], axis = 1)
    less_thr_X = pd.concat([less_thr_X, concat_y[less_thr_X.index]], axis = 1)

    #测试集的拼接
    test_label = pd.DataFrame(concat_y[len(train):].reset_index())
    test_label.drop('index',axis = 1,inplace =True)
    test_concat = pd.concat([test_X,test_label],axis = 1)
    
    #返回根据阈值划分好的两类训练集和血糖值, 测试集拼接血糖分类标签, 不带血糖和fuck的训练集, 测试集血糖概率
    return bigger_thr_X, bigger_thr_y, less_thr_X, less_thr_y, test_concat, train, pred_proba

def linear_model(bigger_thr_X, bigger_thr_y, less_thr_X, less_thr_y, test_concat):
    linear_bigger = LinearRegression()
    linear_less = LinearRegression()

    linear_bigger.fit(bigger_thr_X,bigger_thr_y)
    linear_less.fit(less_thr_X,less_thr_y)
    
    pred_bigger = linear_bigger.predict(test_concat)
    pred_less = linear_less.predict(test_concat)

    return pred_bigger, pred_less, linear_bigger, linear_less

def lasso_model(bigger_thr_X, bigger_thr_y, less_thr_X, less_thr_y, test_concat):
    lasso_bigger = make_pipeline(RobustScaler(), Lasso(alpha =0.005, random_state=1))
    lasso_less = make_pipeline(RobustScaler(), Lasso(alpha =0.005, random_state=1))
    
    lasso_bigger.fit(bigger_thr_X,bigger_thr_y)
    lasso_less.fit(less_thr_X,less_thr_y)

    pred_bigger = lasso_bigger.predict(test_concat)
    pred_less = lasso_less.predict(test_concat)

    return pred_bigger, pred_less, lasso_bigger, lasso_less

def ENet_model(bigger_thr_X, bigger_thr_y, less_thr_X, less_thr_y, test_concat):
    ENet_bigger = make_pipeline(RobustScaler(), ElasticNet(alpha=0.005, l1_ratio=.9, random_state=3))
    ENet_less = make_pipeline(RobustScaler(), ElasticNet(alpha=0.005, l1_ratio=.9, random_state=3))
    
    ENet_bigger.fit(bigger_thr_X,bigger_thr_y)
    ENet_less.fit(less_thr_X,less_thr_y)

    pred_bigger = ENet_bigger.predict(test_concat)
    pred_less = ENet_less.predict(test_concat)

    return  pred_bigger, pred_less, ENet_bigger, ENet_less

def rf_model(bigger_thr_X, bigger_thr_y, less_thr_X, less_thr_y, test_concat):
    rf_bigger = RandomForestRegressor(n_estimators=1000,criterion='mse',max_depth=5,max_features=30,min_samples_leaf=8,n_jobs=12,random_state=17)#min_samples_leaf: 5~10
    rf_less = RandomForestRegressor(n_estimators=1000,criterion='mse',max_depth=5,max_features=30,min_samples_leaf=8,n_jobs=12,random_state=17)#min_samples_leaf: 5~10
    
    rf_bigger.fit(bigger_thr_X,bigger_thr_y)
    rf_less.fit(less_thr_X,less_thr_y)

    pred_bigger = rf_bigger.predict(test_concat)
    pred_less = rf_less.predict(test_concat)

    return  pred_bigger, pred_less, rf_bigger, rf_less

def GBoost_model(bigger_thr_X, bigger_thr_y, less_thr_X, less_thr_y, test_concat):
    gb_bigger = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
    gb_less = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
    
    gb_bigger.fit(bigger_thr_X,bigger_thr_y)
    gb_less.fit(less_thr_X,less_thr_y)

    pred_bigger = gb_bigger.predict(test_concat)
    pred_less = gb_less.predict(test_concat)

    return pred_bigger, pred_less, gb_bigger, gb_less

def LGBM_model(bigger_thr_X, bigger_thr_y, less_thr_X, less_thr_y, test_concat):
    lgb_bigger = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 30)
    lgb_less = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 30)

    lgb_bigger.fit(bigger_thr_X,bigger_thr_y)
    lgb_less.fit(less_thr_X,less_thr_y)

    pred_bigger = lgb_bigger.predict(test_concat)
    pred_less = lgb_less.predict(test_concat)

    return pred_bigger, pred_less, lgb_bigger, lgb_less

def xgb_model(bigger_thr_X, bigger_thr_y, less_thr_X, less_thr_y, test_concat):
    xgb_bigger = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                             learning_rate=0.05, max_depth=5, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.8, silent=1,
                             random_state =7, nthread = -1)
    xgb_less = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                             learning_rate=0.05, max_depth=5, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.8, silent=1,
                             random_state =7, nthread = -1)
    
    xgb_bigger.fit(bigger_thr_X,bigger_thr_y)
    xgb_less.fit(less_thr_X,less_thr_y)

    pred_bigger = xgb_bigger.predict(test_concat)
    pred_less = xgb_less.predict(test_concat)

    return  pred_bigger, pred_less, xgb_bigger, xgb_less

def main():
    print("Reading in Data")
    #最终预测
    train = pd.read_csv('cleaned_train20180129_111517.csv')
    test = pd.read_csv('cleaned_test20180129_111517.csv')

    #验证A榜结果
    #train = pd.read_csv('cleaned_train20180129_102513.csv')
    #test = pd.read_csv('cleaned_test20180129_102513.csv')
    
    test = test.drop(['id'], axis=1)
    train = train.drop(['id'], axis=1)
    y_train = train['血糖']

    #pred_proba为测试集血糖权重
    threshold = 6.5
    test_num = len(test)
    train_num = len(train)
    bigger_thr_X, bigger_thr_y, less_thr_X, less_thr_y, test_concat, X_train, pred_proba = fuck_columns(train, test, threshold)
    

    print("linear model 开始训练")
    pred_bigger, pred_less, linear_bigger, linear_less = linear_model(bigger_thr_X, bigger_thr_y, less_thr_X, less_thr_y, test_concat)   
    #预测结果结合权重
    linear_pred_res=np.array([pred_less[i]*pred_proba[i][0]+pred_bigger[i]*pred_proba[i][1] for i in range(test_num)])
    
    print("lasso model 开始训练")
    pred_bigger, pred_less, lasso_bigger, lasso_less = lasso_model(bigger_thr_X, bigger_thr_y, less_thr_X, less_thr_y, test_concat)   
    #预测结果结合权重
    lasso_pred_res=np.array([pred_less[i]*pred_proba[i][0]+pred_bigger[i]*pred_proba[i][1] for i in range(test_num)])
    
    print("ENet model 开始训练")
    pred_bigger, pred_less, ENet_bigger, ENet_less = ENet_model(bigger_thr_X, bigger_thr_y, less_thr_X, less_thr_y, test_concat)   
    #预测结果结合权重
    ENet_pred_res=np.array([pred_less[i]*pred_proba[i][0]+pred_bigger[i]*pred_proba[i][1] for i in range(test_num)])

    print("集成模型开始训练...")
    print("RandomForestRegressor...")
    pred_bigger, pred_less, rf_bigger, rf_less = rf_model(bigger_thr_X, bigger_thr_y, less_thr_X, less_thr_y, test_concat)   
    #预测结果结合权重
    rf_pred_res=np.array([pred_less[i]*pred_proba[i][0]+pred_bigger[i]*pred_proba[i][1] for i in range(test_num)])

    print("GradientBoostingRegressor...")
    pred_bigger, pred_less, gb_bigger, gb_less = GBoost_model(bigger_thr_X, bigger_thr_y, less_thr_X, less_thr_y, test_concat)   
    #预测结果结合权重
    gb_pred_res=np.array([pred_less[i]*pred_proba[i][0]+pred_bigger[i]*pred_proba[i][1] for i in range(test_num)])

    print("LGBMRegressor...")
    pred_bigger, pred_less, lgb_bigger, lgb_less = LGBM_model(bigger_thr_X, bigger_thr_y, less_thr_X, less_thr_y, test_concat)   
    #预测结果结合权重
    lgb_pred_res=np.array([pred_less[i]*pred_proba[i][0]+pred_bigger[i]*pred_proba[i][1] for i in range(test_num)])

    print("XGBRegressor...")
    pred_bigger, pred_less, xgb_bigger, xgb_less = xgb_model(bigger_thr_X, bigger_thr_y, less_thr_X, less_thr_y, test_concat)   
    #预测结果结合权重
    xgb_pred_res=np.array([pred_less[i]*pred_proba[i][0]+pred_bigger[i]*pred_proba[i][1] for i in range(test_num)])
    
    '''
    Stacking Learning
    '''
    print("StackingRegressor...")
    stacked_averaged_bigger_models = StackingRegressor(
        regressors=[linear_bigger, lasso_bigger ,ENet_bigger],
        meta_regressor= gb_bigger
    )
    stacked_averaged_less_models = StackingRegressor(
        regressors=[linear_less, lasso_less ,ENet_less],
        meta_regressor= gb_less
    )
    #拟合模型
    stacked_averaged_bigger_models.fit(bigger_thr_X,bigger_thr_y)
    stacked_averaged_less_models.fit(less_thr_X,less_thr_y)
    #测试集预测
    stacked_bigger_pred = stacked_averaged_bigger_models.predict(test_concat)
    stacked_less_pred = stacked_averaged_less_models.predict(test_concat)
    #预测结果结合权重
    stacked_pred_res=np.array([stacked_less_pred[i]*pred_proba[i][0]+stacked_bigger_pred[i]*pred_proba[i][1] for i in range(test_num)])
    
    ensemble = stacked_pred_res*0.40+xgb_pred_res*0.40+lgb_pred_res*0.20
    #stacking融合linear
    new_ensemble = np.array([linear_pred_res[i]*pred_proba[i][0]+ensemble[i]*pred_proba[i][1] for i in range(test_num)])
   
    sub = pd.DataFrame({'pred':ensemble})
    sub_wig = pd.DataFrame({'pred':new_ensemble})
    sub.to_csv('submission_b.csv', header=None, index=False)
    sub_wig.to_csv('submission_b_wig.csv', header=None, index=False)

if __name__ == '__main__':
	main()
