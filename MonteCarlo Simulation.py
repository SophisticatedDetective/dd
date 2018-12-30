import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tushare as ts
sjht=ts.get_hist_data('002602')
#sjht.shape
#sjht.head()
alter_sjht=sjht.sort_index(ascending=True)#按时间升序排列
t=180
s0=alter_sjht['close'][0]#初始股价
#s0
data=alter_sjht['close']#收盘价数据
#data.mean()平均收盘价
#Out[30]: 31.59659619450316
return_on_data=data/data.shift(1)
return_on_data-=1#收益率序列
re=return_on_data.mean()#平均收益率
#re
#Out[35]: 0.0008092307033132518
sigma=np.std(return_on_data)#收益率标准差
#sigma
#Out[37]: 0.031073640396856346
n=180
r1=return_on_data[1]
dt=t/n
I=10000
d=(re-0.5*sigma**2)*dt#drift项
K=np.zeros((n+1,I))
K[0,:]=[r1]*I
for i in range(1,n+1):
    K[i,:]=K[i-1,:]+d+sigma*np.sqrt(dt)*np.random.uniform(-0.1,0.1,I)
#matplotlib inline  #启用该项可以窗口内查看图片
plt.plot(s0*K))
plt.rcParams['font.sans-serif']=['SimHei']#正常显示中文
plt.rcParams['axes.unicode_minus']=False#正常显示负号
plt.title('世纪华通股价蒙特卡洛模拟，arthor:chendu')
plt.legend('模拟股价')
plt.xlabel('步数')
plt.ylabel('股价')
#plt.gcf()
