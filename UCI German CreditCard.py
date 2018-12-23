import pandas as pd
import numpy as np
with open(r'C:\Users\chd\Pictures\UCI Statlog (German Credit Data) 原始数据数值化.csv','r') as f_uci_german:
    uci_german=pd.read_csv(f_uci_german)#读取完数据自动关闭文件    
uci_german.shape
temp=uci_german.isnull().any()
type(temp)
temp[temp==True#返回空Series，说明所有特征没有空值
