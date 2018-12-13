xximpo# -*- coding: utf-8 -*-
import numpy as np

import pandas as pd

train_x=pd.read_csv(r'C:\Users\chd\Desktop\train_x.csv')

train_xy=pd.read_csv(r'C:\Users\chd\Desktop\train_xy.csv')

test_all=pd.read_csv(r'C:\Users\chd\Desktop\test_all.csv')

train_x.replace(-99,np.nan,inplace=True)

train_xy.replace(-99,np.nan,inplace=True)

test_all.replace(-99,np.nan,inplace=True)

train_x.dropna(axis=1,thresh=8000,inplace=True)

train_xy.dropna(axis=1,thresh=12000,inplace=True)

test_all.dropna(axis=1,thresh=8000,inplace=True)

train_x_nogroup=train_x[[i for i in train_x.columns if i!='cust_group']]
train_xy_nogroup=train_xy[[i for i in train_xy.columns if i!='cust_group']]
test_all_nogroup=test_all[[i for i in test_all.columns if i!='cust_group']]
train_x_nogroup.shape
train_xy_nogroup.shape
test_all_nogroup.shape
import lightgbm as lgb
from sklearn.tree import DecisionTreeClassifier
train_x_nogroup.to_csv(r'C:\Users\chd\Desktop\train_x_nogroup.csv')
from sklearn.preprocessing import OneHotEncoder
model1=OneHotEncoder()
train_x_nogroup.fillna(method='bfill',inplace=True)
for i in train_x_nogroup.columns:
      print((train_x_nogroup[[i]].count()/10000))
      

train_x_nogroup.shape
train_x_nogroup.fillna(method='bfill',inplace=True)
train_xy_nogroup.fillna(method='bfill',inplace=True)
test_all_nogroup.fillna(method='bfill',inplace=True)

train_x_nogroup=pd.read_csv(r'C:\Users\chd\Desktop\train_x_nogroup.csv')
train_x_nogroup.fillna(method='bfill',inplace=True)
train_x_num=train_x_nogroup[[i for i in train_x_nogroup.columns[1:57]]]
train_x_num.columns
train_x_num_new.columns
train_x_cat=train_x_nogroup[[i for i in train_x_nogroup.columns[57:]]]
train_x_cat.columns
train_x_cat=model1.fit_transform(train_x_cat)
train_x_cat=train_x_cat.todense()
train_x_cat=pd.DataFrame(train_x_cat)
train_x_cat.head()
train_x_new=pd.concat([train_x_num,train_x_cat],axis=1)
train_x.columns
train_xy_num=train_xy_nogroup[train_xy_nogroup.columns[0:58]]
 train_xy_num.drop('y',axis=1)
gs=GridSearchCV(lgbmodel1,param_grid={'lgbmodel1__boosting_type':['gbdt','rf','dart'],
import 'lgbmodel1__learning_rate':[0.0001,0.0002,0.001,0.01],'lgbmodel1__n_estimators':[100,200,300,400,500]})
