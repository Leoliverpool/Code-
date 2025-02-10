#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 03:59:19 2024

@author: leoliverpool
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as sml


# %%0.读取所有数据，并将所有能改的date变为日期型数据
# %%0.1读取所有数据
amt_day = pd.read_csv('/Users/leoliverpool/Desktop/量化投资2023/期中作业（何康）/amt_day.csv',encoding = 'gbk')
close_adj_day = pd.read_csv('/Users/leoliverpool/Desktop/量化投资2023/期中作业（何康）/close_adj_day.csv',encoding = 'gbk')
close_day = pd.read_csv('/Users/leoliverpool/Desktop/量化投资2023/期中作业（何康）/close_day.csv',encoding = 'gbk')
cs_indus_code_day = pd.read_csv('/Users/leoliverpool/Desktop/量化投资2023/期中作业（何康）/cs_indus_code_day.csv',encoding = 'gbk')
csiall_day = pd.read_csv('/Users/leoliverpool/Desktop/量化投资2023/期中作业（何康）/csiall_day.csv',encoding = 'gbk')
day = pd.read_csv('/Users/leoliverpool/Desktop/量化投资2023/期中作业（何康）/day.csv',encoding = 'gbk')
delist_date_info = pd.read_csv('/Users/leoliverpool/Desktop/量化投资2023/期中作业（何康）/delist_date_info.csv',encoding = 'gbk')
float_a_shares_day = pd.read_csv('/Users/leoliverpool/Desktop/量化投资2023/期中作业（何康）/float_a_shares_day.csv',encoding = 'gbk')
IPO_date_info = pd.read_csv('/Users/leoliverpool/Desktop/量化投资2023/期中作业（何康）/IPO_date_info.csv',encoding = 'gbk')
month = pd.read_csv('/Users/leoliverpool/Desktop/量化投资2023/期中作业（何康）/month.csv',encoding = 'gbk')
pb_lf_day = pd.read_csv('/Users/leoliverpool/Desktop/量化投资2023/期中作业（何康）/pb_lf_day.csv',encoding = 'gbk')
share_totala_day = pd.read_csv('/Users/leoliverpool/Desktop/量化投资2023/期中作业（何康）/share_totala_day.csv',encoding = 'gbk')
stock_code_info = pd.read_csv('/Users/leoliverpool/Desktop/量化投资2023/期中作业（何康）/stock_code_info.csv',encoding = 'gbk')
turn_day = pd.read_csv('/Users/leoliverpool/Desktop/量化投资2023/期中作业（何康）/turn_day.csv',encoding = 'gbk')
monthselected = pd.read_csv('/Users/leoliverpool/Desktop/量化投资2023/期中作业（何康）/monthselected.csv',encoding = 'gbk')



# %%1.生成retrun_1m原始因子
# %%1.1 生成因子

close_adj_day.head()
close_adj_day.set_index('Unnamed: 0', inplace = True)
close_adj_day.head()
close_adj_day = close_adj_day.T
close_adj_day.head()
close_adj_day.index.name = 'trade_date'
close_adj_day.head()
close_adj_day = close_adj_day.pct_change(20) #注意这个技巧
#如果当前期或 20 期前的任何一个值为 NaN，则对应结果也会是 NaN。
#注意：pct_change（ ）始终是将当前值与前值进行比较，同时返回的百分比是-1过后的大小。
close_adj_day.head()
close_adj_day.tail()



# %%1.2 剔除新股

close_adj_day = close_adj_day.stack(dropna=False) #重要技巧
#注意：dropna=False很重要，默认为True，这样会出问题，最后生成的df大小不对。
close_adj_day.info()
close_adj_day.head()
close_adj_day.tail()
close_adj_day = close_adj_day.reset_index()
close_adj_day = close_adj_day.rename(columns = {'Unnamed: 0':'stock_code', '0': 'return_1m'}) #注意是大括号
close_adj_day.head()
close_adj_day.tail()

month.head()
month.drop('Unnamed: 0', axis = 1, inplace = True) #注意要加axis=1
month.info()
month['date'] = pd.to_datetime(month['date']) #可以看见，pd.to_datetime很好用。不管原始时间是什么格式，都可以转换

#close_adj_day = pd.concat([close_adj_day, month], axis=1) 
#注意这样做有问题，因为二者的行数不同。因此，concat只适用于将dataframe合并，而不进行数据处理。
#同时，直接合并必须使用concat，因为merge无法直接合并，除非添加一个相同的列。
#注意：需要将时间改成统一格式
close_adj_day['trade_date'] = pd.to_datetime(close_adj_day['trade_date'])
close_adj_day.info()


#close_adj_day = pd.merge(close_adj_day, month, how = 'left', left_on = 'trade_date', right_on = 'date') 
#注意，left要加引号

# %%1.3 试图生成截面日的滞后252个交易日的日期

close_adj_day.head()
month.head()
day.info()
day.head()
day.drop('Unnamed: 0', axis=1, inplace=True)
day['date'] = pd.to_datetime(day['date'])



#注意这里合并的技巧，新创造复制的一列，从而实现month和day的合并
month['value_month'] = month['date']
# 以 day 为基准进行合并
result = pd.merge(day, month, how='left', on='date')
print(result)

#注意这里的技巧：运用了shift，isin，where三个方法来找到某一列在另一列出现的行，且在该行生成另一列中滞后252行的项
result['252_days_threshold'] = result['date'].shift(252)
#注意：1.column name可以数字开头；2.series和df都可以shift，缺少的用na自动填充
result['is_last_day'] = result['value_month'].isin(result['date'])
result['threshold'] = result['252_days_threshold'].where(result['is_last_day'] ==True)
result.drop(['value_month','is_last_day','252_days_threshold'], axis=1, inplace=True)
print(result)


close_adj_day = pd.merge(close_adj_day,result, how = 'left', left_on = 'trade_date', right_on = 'date') 
#注意这一步会添加result中的‘date’列

close_adj_day.info() #行数太多导致无法显示null的个数
close_adj_day.isnull().sum() #利用这个方法可以解决，这是一个series

IPO_date_info.info()
#IPO_date_info.rename({'Unnamed: 0': 'stock_code'}, inplace = True)#没有用不知道为什么，改为另外的改列名的方法
#IPO_date_info.columns.values[0] = 'trade_date' 这种方法会出错！！，不要用
IPO_date_info.columns = ['stock_code', 'IPO_date']
IPO_date_info.info()

close_adj_day = pd.merge(close_adj_day,IPO_date_info, how = 'left', on = 'stock_code')
close_adj_day.info()
close_adj_day['IPO_date'] = pd.to_datetime(close_adj_day['IPO_date'])
close_adj_day.drop('date', axis=1, inplace=True)

close_adj_day.columns = ['trade_date', 'stock_code', 'return_1m', 'threshold', 'IPO_date']
close_adj_day.loc[close_adj_day['IPO_date'] > close_adj_day['threshold'],'return_1m'] = np.nan 
#注意这种用法，能够帮你选中部分行与列来赋值，用到loc
close_adj_day.isnull().sum() 



# %%1.4 剔除已退市股票 (部分股票有问题，如000003，在退市后仍有价格，因此需要剔除)
delist_date_info.info()
delist_date_info.columns = ['stock_code', 'delist_date']
close_adj_day = pd.merge(close_adj_day,delist_date_info, how = 'left', on = 'stock_code')
close_adj_day.info()
close_adj_day['delist_date'] = pd.to_datetime(close_adj_day['delist_date'])
close_adj_day.loc[close_adj_day['delist_date'] < close_adj_day['trade_date'],'return_1m'] =np.nan
close_adj_day.isnull().sum() 



# %%1.5 剔除月末截面日停牌股票
amt_day = pd.read_csv('/Users/leoliverpool/Desktop/量化投资2023/期中作业（何康）/amt_day.csv',encoding = 'gbk')
amt_day.info()
amt_day.set_index('Unnamed: 0', inplace = True)
#注意必须先将stock_index设为index，否则stack时会出错
amt_day = amt_day.stack(dropna=False)
#注意stack后，python会自动丢掉nan的值，因此导致两个大小相同的df：close_adj_day和amt_day在stack后长度不一样。
amt_day.head()
amt_day = amt_day.reset_index()
amt_day.head()
amt_day.columns = ['stock_code', 'trade_date', 'amt']
amt_day.head()
amt_day['trade_date'] = pd.to_datetime(amt_day['trade_date'])
close_adj_day.head()
amt_day.tail()
close_adj_day.tail()
close_adj_day = pd.merge(close_adj_day, amt_day, how = 'left', left_on = ['trade_date','stock_code'], \
                         right_on = ['trade_date','stock_code'])
#注意这种用法，on/left_on/right_on均可以使用列表，从而实现多键匹配
close_adj_day.info()

close_adj_day.loc[close_adj_day['amt'] ==0, 'return_1m'] = np.nan
close_adj_day.isnull().sum() 



# %%1.6 输出dataframe为excel
#close_adj_day.drop(['threshold','IPO_date','delist_date','amt'], axis=1,inplace = True) 应该不用去
return_1m = close_adj_day.pivot(index = 'trade_date', columns = 'stock_code', values = 'return_1m')

#return_1m.to_csv('/Users/leoliverpool/Desktop/量化投资2023/期中作业1/return_1m.csv')




# %%1.7 将日期限制于2010/12/31至2023/05/31
return_1m.info()
return_1m = return_1m[(return_1m.index >= pd.to_datetime('2010-12-31')) & (return_1m.index <= pd.to_datetime('2023-5-31'))]
#return_1m = return_1m[(return_1m.index >= pd.to_datetime('2010-05-31')) and (return_1m.index <= pd.to_datetime('2024-12-31'))]
#注意：and 是 Python 的逻辑操作符，它只能处理两个单独的布尔值。因此这里不能用and。而用&时需要注意使用括号括起来




# %%1.8 将日期限制于月末
#截取时，应当自觉想到pd.merge()
return_1m = pd.merge(month,return_1m,how='inner',left_on = 'date', right_index = True)
return_1m.drop('value_month', axis = 1, inplace = True)
return_1m.set_index('date', inplace = True)
#注意：（1）要用inner （2）可以用index来筛选 （right_index）
return_1m.to_csv('/Users/leoliverpool/Desktop/量化投资2023/期中作业1/factor_1_raw.csv')





# %%2.生成turn_1m原始因子
turn_day = pd.read_csv('/Users/leoliverpool/Desktop/量化投资2023/期中作业（何康）/turn_day.csv',encoding = 'gbk')

turn_day.head()
turn_day.set_index('Unnamed: 0', inplace = True)
turn_day.head()
turn_day = turn_day.T
turn_day.head()
turn_day.index.name = 'trade_date'
turn_day.head()

# 使用 rolling 计算每一行和前 20 行（共 21 行）的简单平均值
turn_day = turn_day.rolling(window=21, min_periods=21).mean()

#turn_day = turn_day.rolling(window =21, min_periods=21).mean() 多写几遍
#rolling(window=21)：创建一个滚动窗口，包含当前行和前 20 行（共 21 行）。
#min_periods=21：
    #确保窗口必须有完整的 21 行才能计算均值，否则结果为 NaN。
    #如果省略 min_periods，默认等于 window。
#这里等于21是合理的，因为后续是用这些因子来进行分层回测，一方面，将一些股票剔除不会有太大影响；
#另一方面，若个数不足21个，则加权平均个数不为21，失去原本的经济学含义。

#接下来，重复以上操作，写成函数形式。
def stock_pool_filter_and_date_filter_and_factor_generator(close_adj_day, factor_name,filename):
    

    # %%1.2 剔除新股
    
    close_adj_day = close_adj_day.stack(dropna=False) #重要技巧
    #注意：dropna=False很重要，默认为True，这样会出问题，最后生成的df大小不对。
    close_adj_day.info()
    close_adj_day.head()
    close_adj_day.tail()
    close_adj_day = close_adj_day.reset_index()
    close_adj_day = close_adj_day.rename(columns = {'Unnamed: 0':'stock_code', '0': factor_name}) #注意是大括号
    close_adj_day.head()
    close_adj_day.tail()
    
    month = pd.read_csv('/Users/leoliverpool/Desktop/量化投资2023/期中作业（何康）/month.csv',encoding = 'gbk')
    month.head()
    month.drop('Unnamed: 0', axis = 1, inplace = True) #注意要加axis=1
    month.info()
    month['date'] = pd.to_datetime(month['date']) #可以看见，pd.to_datetime很好用。不管原始时间是什么格式，都可以转换
    
    #close_adj_day = pd.concat([close_adj_day, month], axis=1) 
    #注意这样做有问题，因为二者的行数不同。因此，concat只适用于将dataframe合并，而不进行数据处理。
    #同时，直接合并必须使用concat，因为merge无法直接合并，除非添加一个相同的列。
    #注意：需要将时间改成统一格式
    close_adj_day['trade_date'] = pd.to_datetime(close_adj_day['trade_date'])
    close_adj_day.info()
    
    
    #close_adj_day = pd.merge(close_adj_day, month, how = 'left', left_on = 'trade_date', right_on = 'date') 
    #注意，left要加引号
    
    # %%1.3 试图生成截面日的滞后252个交易日的日期
    
    close_adj_day.head()
    month.head()
    day = pd.read_csv('/Users/leoliverpool/Desktop/量化投资2023/期中作业（何康）/day.csv',encoding = 'gbk')
    day.info()
    day.head()
    day.drop('Unnamed: 0', axis=1, inplace=True)
    day['date'] = pd.to_datetime(day['date'])
    
    
    
    #注意这里合并的技巧，新创造复制的一列，从而实现month和day的合并
    month['value_month'] = month['date']
    # 以 day 为基准进行合并
    result = pd.merge(day, month, how='left', on='date')
    print(result)
    
    #注意这里的技巧：运用了shift，isin，where三个方法来找到某一列在另一列出现的行，且在该行生成另一列中滞后252行的项
    result['252_days_threshold'] = result['date'].shift(252)
    #注意：1.column name可以数字开头；2.series和df都可以shift，缺少的用na自动填充
    result['is_last_day'] = result['value_month'].isin(result['date'])
    result['threshold'] = result['252_days_threshold'].where(result['is_last_day'] ==True)
    result.drop(['value_month','is_last_day','252_days_threshold'], axis=1, inplace=True)
    print(result)
    
    
    close_adj_day = pd.merge(close_adj_day,result, how = 'left', left_on = 'trade_date', right_on = 'date') 
    #注意这一步会添加result中的‘date’列
    
    close_adj_day.info() #行数太多导致无法显示null的个数
    close_adj_day.isnull().sum() #利用这个方法可以解决，这是一个series
    
    IPO_date_info = pd.read_csv('/Users/leoliverpool/Desktop/量化投资2023/期中作业（何康）/IPO_date_info.csv',encoding = 'gbk')
    IPO_date_info.info()
    #IPO_date_info.rename({'Unnamed: 0': 'stock_code'}, inplace = True)#没有用不知道为什么，改为另外的改列名的方法
    #IPO_date_info.columns.values[0] = 'trade_date' 这种方法会出错！！，不要用
    IPO_date_info.columns = ['stock_code', 'IPO_date']
    IPO_date_info.info()
    
    close_adj_day = pd.merge(close_adj_day,IPO_date_info, how = 'left', on = 'stock_code')
    close_adj_day.info()
    close_adj_day['IPO_date'] = pd.to_datetime(close_adj_day['IPO_date'])
    close_adj_day.drop('date', axis=1, inplace=True)
    
    close_adj_day.columns = ['trade_date', 'stock_code', factor_name, 'threshold', 'IPO_date']
    close_adj_day.loc[close_adj_day['IPO_date'] > close_adj_day['threshold'],factor_name] = np.nan 
    #注意这种用法，能够帮你选中部分行与列来赋值，用到loc
    close_adj_day.isnull().sum() 
    
    
    
    # %%1.4 剔除已退市股票 (部分股票有问题，如000003，在退市后仍有价格，因此需要剔除)
    delist_date_info = pd.read_csv('/Users/leoliverpool/Desktop/量化投资2023/期中作业（何康）/delist_date_info.csv',encoding = 'gbk')
    delist_date_info.info()
    delist_date_info.columns = ['stock_code', 'delist_date']
    close_adj_day = pd.merge(close_adj_day,delist_date_info, how = 'left', on = 'stock_code')
    close_adj_day.info()
    close_adj_day['delist_date'] = pd.to_datetime(close_adj_day['delist_date'])
    close_adj_day.loc[close_adj_day['delist_date'] < close_adj_day['trade_date'],factor_name] =np.nan
    close_adj_day.isnull().sum() 
    
    
    
    # %%1.5 剔除月末截面日停牌股票
    amt_day = pd.read_csv('/Users/leoliverpool/Desktop/量化投资2023/期中作业（何康）/amt_day.csv',encoding = 'gbk')
    amt_day.info()
    amt_day.set_index('Unnamed: 0', inplace = True)
    #注意必须先将stock_index设为index，否则stack时会出错
    amt_day = amt_day.stack(dropna=False)
    #注意stack后，python会自动丢掉nan的值，因此导致两个大小相同的df：close_adj_day和amt_day在stack后长度不一样。
    amt_day.head()
    amt_day = amt_day.reset_index()
    amt_day.head()
    amt_day.columns = ['stock_code', 'trade_date', 'amt']
    amt_day.head()
    amt_day['trade_date'] = pd.to_datetime(amt_day['trade_date'])
    close_adj_day.head()
    amt_day.tail()
    close_adj_day.tail()
    close_adj_day = pd.merge(close_adj_day, amt_day, how = 'left', left_on = ['trade_date','stock_code'], \
                             right_on = ['trade_date','stock_code'])
    #注意这种用法，on/left_on/right_on均可以使用列表，从而实现多键匹配
    close_adj_day.info()
    
    close_adj_day.loc[close_adj_day['amt'] ==0, factor_name] = np.nan
    close_adj_day.isnull().sum() 
    
    
    
    # %%1.6 输出dataframe为excel
    #close_adj_day.drop(['threshold','IPO_date','delist_date','amt'], axis=1,inplace = True) 应该不用去
    return_1m = close_adj_day.pivot(index = 'trade_date', columns = 'stock_code', values = factor_name)
    
    #return_1m.to_csv('/Users/leoliverpool/Desktop/量化投资2023/期中作业1/return_1m.csv')
    
    
    
    
    # %%1.7 将日期限制于2010/12/31至2023/05/31
    return_1m.info()
    return_1m = return_1m[(return_1m.index >= pd.to_datetime('2010-12-31')) & (return_1m.index <= pd.to_datetime('2023-5-31'))]
    #return_1m = return_1m[(return_1m.index >= pd.to_datetime('2010-05-31')) and (return_1m.index <= pd.to_datetime('2024-12-31'))]
    #注意：and 是 Python 的逻辑操作符，它只能处理两个单独的布尔值。因此这里不能用and。而用&时需要注意使用括号括起来
    
    
    
    
    # %%1.8 将日期限制于月末
    #截取时，应当自觉想到pd.merge()
    return_1m = pd.merge(month,return_1m,how='inner',left_on = 'date', right_index = True)
    return_1m.drop('value_month', axis = 1, inplace = True)
    return_1m.set_index('date', inplace = True)
    #注意：（1）要用inner （2）可以用index来筛选 （right_index）
    return_1m.to_csv(filename)
    

#以下为检验，检验函数生成的与手动生成的一致。
#stock_pool_filter_and_date_filter_and_factor_generator(close_adj_day, 'return_1m','/Users/leoliverpool/Desktop/量化投资2023/期中作业1/factor_11_raw.csv')
#A = pd.read_csv('/Users/leoliverpool/Desktop/量化投资2023/期中作业1/factor_11_raw.csv')    
#B = pd.read_csv('/Users/leoliverpool/Desktop/量化投资2023/期中作业1/factor_1_raw.csv')  
#is_equal = A.equals(B)
#注意equals方法，很好用。
#注意，在函数中的输出，输入文件均可实现

stock_pool_filter_and_date_filter_and_factor_generator(turn_day, 'turn_1m','/Users/leoliverpool/Desktop/量化投资2023/期中作业1/factor_2_raw.csv')
#C = pd.read_csv('/Users/leoliverpool/Desktop/量化投资2023/期中作业1/factor_2_raw.csv')  
#C.info()
    

    

# %%3.生成std_1m原始因子
    

close_adj_day = pd.read_csv('/Users/leoliverpool/Desktop/量化投资2023/期中作业（何康）/close_adj_day.csv',encoding = 'gbk')

close_adj_day.head()
close_adj_day.set_index('Unnamed: 0', inplace = True)
close_adj_day.head()
close_adj_day = close_adj_day.T
close_adj_day.head()
close_adj_day.index.name = 'trade_date'
close_adj_day.head()

close_adj_day = close_adj_day.rolling(window = 21, min_periods =21).std()

stock_pool_filter_and_date_filter_and_factor_generator(close_adj_day, 'std_1m','/Users/leoliverpool/Desktop/量化投资2023/期中作业1/factor_3_raw.csv')
    
    
    

# %%4.生成std_FF3factor_1m原始因子


# %%4.1 生成日收益率
close_adj_day = pd.read_csv('/Users/leoliverpool/Desktop/量化投资2023/期中作业（何康）/close_adj_day.csv',encoding = 'gbk')

close_adj_day.head()
close_adj_day.set_index('Unnamed: 0', inplace = True)
close_adj_day.head()
close_adj_day = close_adj_day.T
close_adj_day = close_adj_day.pct_change(1)
close_adj_day = close_adj_day.T

close_adj_day = close_adj_day.stack(dropna=False) 
#注意：stack不支持inplace=True
#注意：stack后均为一个series（除非你是双层索引，一般不会）
close_adj_day = close_adj_day.reset_index()




# %%4.2 生成中证全指日收益率
csiall_day = pd.read_csv('/Users/leoliverpool/Desktop/量化投资2023/期中作业（何康）/csiall_day.csv',encoding = 'gbk')
csiall_day = csiall_day.T
csiall_day.drop('Unnamed: 0', inplace = True)
csiall_day.info()
csiall_day = csiall_day.pct_change(1) #注意：pct_change(1)会自动将object转为float
csiall_day.info()



# %%4.3 生成SMB因子日收益率

close_day = pd.read_csv('/Users/leoliverpool/Desktop/量化投资2023/期中作业（何康）/close_day.csv',encoding = 'gbk')
share_totala_day = pd.read_csv('/Users/leoliverpool/Desktop/量化投资2023/期中作业（何康）/share_totala_day.csv',encoding = 'gbk')
close_day.set_index('Unnamed: 0', inplace = True)
share_totala_day.set_index('Unnamed: 0', inplace = True)
market_value = close_day * share_totala_day
market_value = market_value.stack(dropna=False)
market_value = market_value.reset_index()




# %%4.4 生成HML因子日收益率

pb_lf_day = pd.read_csv('/Users/leoliverpool/Desktop/量化投资2023/期中作业（何康）/pb_lf_day.csv',encoding = 'gbk')
pb_lf_day.set_index('Unnamed: 0', inplace = True)
pb_lf_day = 1/pb_lf_day
pb_lf_day = pb_lf_day.stack(dropna=False)
pb_lf_day = pb_lf_day.reset_index()
pb_lf_day.head()
market_value.head()
close_adj_day.head()


# %%4.5 合并收益率，csiall，SMB与HML于一个dataframe

close_adj_day.columns = ['stock_code','trade_date','return']
market_value.columns = ['stock_code','trade_date','mkt_value']
pb_lf_day.columns = ['stock_code','trade_date','bp']
#close_day = pd.read_csv('/Users/leoliverpool/Desktop/量化投资2023/期中作业（何康）/close_day.csv',encoding = 'gbk')
#close_day.set_index('Unnamed: 0', inplace = True)
#A = close_day.columns #这一数据类型为index，需要将其转为Series
csiall_day.columns = ['csi_ret']

close_adj_day = pd.merge(close_adj_day, market_value, on= ['stock_code', 'trade_date'], how = 'inner')
close_adj_day = pd.merge(close_adj_day, pb_lf_day, on= ['stock_code', 'trade_date'], how = 'inner')
close_adj_day = pd.merge(close_adj_day, csiall_day, left_on = 'trade_date', right_index = True, how='left')
close_adj_day1= close_adj_day
#close_adj_day = close_adj_day1




# %%4.5生成HML因子日收益率

def find_max_and_min_bp(data):
    upper_limit = data.bp.quantile(0.7) #注意quantile的用法，可以用分位数用于筛选出某一比例的组别
    lower_limit = data.bp.quantile(0.3) #尽管data.bp 与 data['bp']两者在多数情况下等价，但出于通用性和安全性考虑，推荐使用 data['bp']。
    big_return = data.loc[data.bp >=upper_limit, 'return'].mean()
    small_return = data.loc[data.bp <=lower_limit, 'return'].mean()
    #big_return = data['return'][data.bp >=  upper_limit].mean() 
    #注意尽量避免链式索引（如 data['return'][...]），因为这可能导致潜在的性能问题或不一致的行为。推荐用 .loc 明确表达。
    #small_return = data['return'][data.bp <=  lower_limit].mean()
    return_final = big_return - small_return
    data['final_HML'] = return_final #注意必须选择在里面来填充dataframe
    return data #注意不要忘记return data


close_adj_day = close_adj_day.groupby('trade_date').apply(find_max_and_min_bp) #在groupby中只需放入列名即可

#def calculate_bp_diff(data):
    #upper_limit = data.quantile(0.7)
    #lower_limit = data.quantile(0.3)
    #big_return = data.loc[data >= upper_limit, 'return'].mean()
    #small_return = data.loc[data <= lower_limit, 'return'].mean()
    #return big_return - small_return

# 对 trade_date 分组计算差值
#close_adj_day['final_bp'] = close_adj_day.groupby('trade_date')['bp'].transform(calculate_bp_diff)


#假设利用transform则无需在函数中填充dataframe，因此transform未来也可以考虑使用

#关键区别总结
#特性	transform	                                        apply
#返回结果形状	与原 DataFrame 的行数相同	                    与分组的组数相同
#适用场景	按组操作后返回逐行计算的结果，并与原表对齐	         适用于返回聚合值或复杂操作
#典型用途	生成与原表对齐的新列，如均值、标准化等	       计算分组的汇总值或更复杂的操作




# %%4.6生成SMB因子日收益率
close_adj_day.columns
close_adj_day.set_index('trade_date', inplace = True) #这一步即可替代双索引

def find_max_and_min_mkt(data):
    upper_limit = data['mkt_value'].quantile(0.7)
    lower_limit = data['mkt_value'].quantile(0.3)
    return_high = data.loc[data['mkt_value'] >= upper_limit, 'return'].mean()
    return_low = data.loc[data['mkt_value'] <= lower_limit, 'return'].mean()
    return_diff = return_low - return_high
    data['final_SMB'] = return_diff
    return data

close_adj_day = close_adj_day.groupby('trade_date').apply(find_max_and_min_mkt)

close_adj_day = close_adj_day.droplevel(level=0) #注意：这个方法可以删除双层索引



# %%4.7生成线性回归的dataframe
close_adj_day = close_adj_day.sort_values(by=['stock_code','trade_date'])
#注意：想要变换行的顺序，应用sort_values方法，这里可以灵活识别index的column name


# %%4.8 进行线性回归并计算因子值

# %%4.8.1 第一次尝试（这里的分析与学习见另一文件，为了更加清晰）
import statsmodels.api as sm
#注意：之所以用sm是因为sm的特征是（1）模型较简单（没有tree model等模型，但有时间序列logistic模型）；（2）更适合统计分析和经济计量学中的使用场景。
#sklearn的特征是：（1）模型较复杂；（2）更适合更适合预测任务和建模自动化。这里显然sm更合适

#def calculate_volatility(window):
    # 如果窗口内任何一列有缺失值，返回 np.nan
    #if window.isna().any().any(): #注意这样的表示技巧，很好用。
    #window.isna()：
    #返回一个布尔型 DataFrame，标记 window 中每个元素是否为 NaN。
    #window.isna().any()：
    #检查每一列是否有 NaN，返回一个布尔型 Series。
    #window.isna().any().any()：
    #检查布尔型 Series 是否有 True，即窗口中是否有任何缺失值。
        #return np.nan
    
    # 因变量和自变量
    #y = window[:, 0]
    #X = window[:, 1:4]
    #注意这样提取时需要两个[]，同时.loc也可以实现相同的任务。即X = window.loc[:, ['csi_ret', 'final_HML', 'final_SMB']]
    # 添加常数项
    #X = sm.add_constant(X)#多了一列，该列全为1
    
    # 拟合线性回归模型
    #model = sm.OLS(y, X).fit() #注意这里将OLS等价于线性回归。原因在于往往只有在线性回归中才会用OLS
    #不要忘记fit
    
    # 计算残差（一个向量）
    #residuals = y - model.predict(X) 
    
    # 返回残差的波动率（标准差）
    #return residuals.std()
    
# 选择数值列进行 rolling 操作
#numeric_columns = ['return', 'csi_ret', 'final_HML', 'final_SMB']
#close_adj_day['std_FF3_1m'] = close_adj_day[numeric_columns].rolling(window=21, min_periods=21).apply(calculate_volatility, raw = False)





# %%4.8.2 第二次尝试
def calculate_volatility(window):
    # 检查窗口中是否有 NaN
    if np.isnan(window).any():  # NumPy 的方法
    # np.isnan() 会返回一个布尔型数组，其中每个元素表示原始数组中对应位置的值是否为 NaN。
    #.any()这是一个 NumPy 数组的方法，用于检查数组中是否至少有一个元素是 True。
        return np.nan
    # 因变量和自变量
    y = window[:, 0]
    X = window[:, 1:4]
    # 添加常数项
    X = sm.add_constant(X)
    # 拟合线性回归模型
    model = sm.OLS(y, X).fit()
    # 计算残差
    residuals = y - model.predict(X)
    # 返回残差的波动率（标准差）
    return residuals.std()


def apply_rolling_with_numpy(df, window_size, func):
    values = df.to_numpy()      #这一操作可以直接将df变为一个numpy，既可以仅包含数的ndarray
    #之所以用numpy（ndarray），是因为其运行速度是最快的。
    result = np.empty(len(df))  #生成一个长度与df相等，里面全是0的ndarray
    result[:] = np.nan  # 初始化为 NaN 

    for i in range(window_size - 1, len(values)): #注意这个技巧：从哪行开始生成，则把i设为该行的行数最方便
        window = values[i - window_size + 1:i + 1, :] #切片出ndarray
        result[i] = func(window) #将结果输入进result

    return pd.Series(result, index=df.index)

numeric_columns = ['return', 'csi_ret', 'final_HML', 'final_SMB']

# 使用自定义函数进行滚动计算
close_adj_day['std_FF3_1m'] = apply_rolling_with_numpy(
    close_adj_day[numeric_columns],
    window_size=21,
    func=calculate_volatility
)

'''
总结
1.想要滚动提取多行df，只能使用for循环
2.以后在大数据运算时，要将其转为numpy的ndarray，以实现运算最快。其中会用到to_numpy(),np.empty(),numpy的切片[:,:]，
numpy的赋值[i]，np.isnan().any()等操作

'''
#close_adj_day.to_csv('/Users/leoliverpool/Desktop/量化投资2023/期中作业1/std_ff3_generator.csv')


# %%4.9 读取生成因子的文件并给出标准形式的因子。（column为stockcode，index为date）

std_ff3_generator = pd.read_csv('/Users/leoliverpool/Desktop/量化投资2023/期中作业1/std_ff3_generator.csv')
std_ff3_generator.head()
std_ff3_generator.columns
std_ff3_generator.drop(['return', 'mkt_value', 'bp', 'csi_ret', 'final_HML', 'final_SMB'], axis=1, inplace = True)
std_ff3_generator.isna().sum()#注意是sum! 前面是isna或isnull都可以
std_ff3 = std_ff3_generator.pivot(index = 'trade_date', columns ='stock_code', values = 'std_FF3_1m')



# %%4.10 将因子进行股票池筛选，日期限制操作，并输出csv（利用前述函数）

stock_pool_filter_and_date_filter_and_factor_generator(std_ff3, 'std_FF3_1m','/Users/leoliverpool/Desktop/量化投资2023/期中作业1/factor_4_raw.csv')

close_adj_day2 = close_adj_day



# %%5.因子预处理






for i in range(2,0):
    print(i)



























