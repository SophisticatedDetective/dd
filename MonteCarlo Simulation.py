import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tushare as ts
def get_mtklresult(stock_code='000001',t=180,I=10000,n=1,print_hist=True,print_plot=True):#stock_code是股票代码，t是模拟期数，n是区间数,I是自定义模拟次数，print_hist是True输出频率分布图，False不输出；print_plot是True输出蒙特卡洛模拟股价图，False则不输出
       price=ts.get_hist_data(stock_code)
       # price.shape
       #price.head()
       alter_price=price.sort_index(ascending=True)#按时间升序排列
       s0=alter_price['close'][0]#初始股价
       #s0
       close_price=alter_price['close']#收盘价数据
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
       if print_hist==True:
            plt.hist(K[-1],bins=I/20)
       if print_plot==True:
            plt.plot(s0*K)
            plt.rcParams['font.sans-serif']=['SimHei']#正常显示中文
            plt.rcParams['axes.unicode_minus']=False#正常显示负号
            plt.title('股价蒙特卡洛模拟')
            plt.legend('模拟股价')
            plt.xlabel('步数')
            plt.ylabel('股价')
            #plt.gcf()
