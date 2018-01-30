# -*- coding: utf-8 -*-

import time
import datetime
import numpy as np
import pandas as pd
from dateutil.parser import parse

'''
对数据集进行清理,主要为填充缺失值, 处理异常值
'''
def imput_miss_outlier_value(train, test):

    #记录初始特征列的顺序
    train_cols = train.columns.tolist()
    test_cols = test.columns.tolist()
    
    data = pd.concat([train,test])
    data = data.loc[:,train_cols]
    
    # 过滤未知性别的
    data['性别'] = data['性别'].map({'男':1, '女':0})

    # 转化体检日期
    data['体检日期'] = pd.to_datetime(data['体检日期'])
    data['weekday'] = data['体检日期'].apply(lambda r: r.weekday())  
    data['weekendFlag']= 0
    data['weekendFlag'] = (data['weekday']>5)+0
    data['weekday'] = data['weekday'].apply(lambda r: 'd'+str((r+1)))  

    # 删除体检日期
    data=data.drop(['体检日期'],axis=1)
    
    #缺失值填充中位数
    data = pd.get_dummies(data) #进行one-hot编码
    data = data.fillna(data.mean())

    train_len = len(train)
    train = data[0:train_len]
    test = data[train_len:]

    test = test.drop(['血糖'], axis=1)

    return train, test
    

def main():
    print('Starting load data...')
    data_path = './data/'
    train = pd.read_csv(data_path+'merge_train.csv',encoding='gb2312')
    test = pd.read_csv(data_path+'d_test_B_20180128.csv',encoding='gb2312')
    print('train shape:', train.shape)
    print('test shape:', test.shape)
    
    print('Imputing missing values')
    train, test = imput_miss_outlier_value(train, test)

    train.to_csv(r'cleaned_train{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), index=False, float_format='%.4f')
    test.to_csv(r'cleaned_test{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),  index=False, float_format='%.4f')


if __name__ == '__main__':
    main()

