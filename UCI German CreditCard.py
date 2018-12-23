import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import SGDClassifier,LogisticRegressionCV,RidgeClassifierCV
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
with open(r'C:\Users\chd\Pictures\UCI Statlog (German Credit Data) 原始数据数值化.csv','r') as f_uci_german:
    uci_german=pd.read_csv(f_uci_german)#读取完数据自动关闭文件    
uci_german.shape
temp=uci_german.isnull().any()
type(temp)
temp[temp==True]#返回空Series，说明所有特征没有空值
train_data=uci_german[uci_german.columns[0:-1]]
train_label=uci_german[uci_german.columns[-1]]
x_train,x_test,y_train,y_test=train_test_split(train_data,train_label,test_size=0.3,random_state=0)#随机划分30%为测试特征集和标签集
linearSVC_model1=LinearSVC(max_iter=1500).fit(x_train,y_train)
pred1=linearSVC_model1.predict(x_test)
error1=mean_squared_error(y_test,pred1)
pred2=svc_model1.predict_proba(x_test)
error2=mean_squared_error(y_test,pred2)
sgd_model1=SGDClassifier(max_iter=2000).fit(x_train,y_train)
pred3=sgd_model1.predict(x_test)
error3=mean_squared_error(y_test,pred3)
lrcv=LogisticRegressionCV(max_iter=2000).fit(x_train,y_train)
pred4=lrcv.predict(x_test)#加了交叉验证误差率下降了0.04
error4=mean_squared_error(y_test,pred4)
rcv=RidgeClassifierCV().fit(x_train,y_train)
pred5=rcv.predict(x_test)
error5=mean_squared_error(y_test,pred5)
