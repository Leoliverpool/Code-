#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 18:28:26 2024

@author: leoliverpool
"""



# %%0 调取数据 (两年中有9天的数据无法在tushare网站调取，但影响不大)
import pandas as pd
import tushare as ts
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
print(ts.__version__)
ts.set_token('19fe88084dba6bc2490eba7829bcdcb3ec5b7805da01a9db78e06245')
pro = ts.pro_api()

# 设置查询的起始日期和结束日期
start_date = '20170101'
end_date = '20181231'

# 生成日期范围
dates = pd.date_range(start=start_date, end=end_date)

# 初始化一个空的DataFrame用于存储结果
df_total = pd.DataFrame()

for single_date in dates:
    trade_date = single_date.strftime("%Y%m%d")
    try:
        df = pro.margin_detail(trade_date=trade_date)
        df_total = pd.concat([df_total, df], ignore_index=True)
    except Exception as e:
        print(f"Error fetching data for {trade_date}: {e}")
    time.sleep(0.32)  # 在每次请求后暂停1秒

# 现在df_total包含了2018年8月1日到2018年8月31日的融资融券详细数据

df_total.info()
unique_ts_codes = df_total['ts_code'].unique()

# 计算唯一值的数量
num_unique_ts_codes = len(unique_ts_codes)

print(f"ts_code列中有 {num_unique_ts_codes} 种不同的取值。")




# %%1. 生成原始因子值并筛选有效样本
# %%1.1 生成原始因子值
# 假设df_total是你的DataFrame

# 计算df_total中trade_date的唯一值数量
unique_trade_dates_count = df_total['trade_date'].nunique()

# 对df_total按ts_code分组，并计算每个ts_code对应的trade_date的唯一值数量
ts_code_trade_dates_count = df_total.groupby('ts_code')['trade_date'].nunique()

# 过滤出在所有trade_date中都出现过的ts_code
ts_codes_in_all_trade_dates = ts_code_trade_dates_count[ts_code_trade_dates_count == unique_trade_dates_count].index

# 计算这些ts_code的数量
num_ts_codes_in_all_trade_dates = len(ts_codes_in_all_trade_dates)

print(f"在所有trade_date中都出现的ts_code的数量是: {num_ts_codes_in_all_trade_dates}")
print(f"这些ts_code包括: {ts_codes_in_all_trade_dates.tolist()}")


# 筛选df_total中，ts_code存在于ts_codes_in_all_trade_dates中的行
df_filtered = df_total[df_total['ts_code'].isin(ts_codes_in_all_trade_dates)]

# 打印筛选后的DataFrame的信息，确认筛选成功
df_filtered.info()

# 选择df_filtered中的特定列
df_restricted = df_filtered[['trade_date', 'ts_code', 'rqmcl']]

# 打印筛选后的DataFrame的前几行，以确认是否正确执行了列选择
print(df_restricted.head())

# 按ts_code分组，然后检查每个分组中rqmcl值是否都为0
ts_codes_all_zero_rqmcl = df_restricted.groupby('ts_code').filter(lambda x: (x['rqmcl'] == 0).all())

# 获取满足条件的ts_code的唯一值
unique_ts_codes_with_all_zero_rqmcl = ts_codes_all_zero_rqmcl['ts_code'].unique()

# 打印结果
print(f"所有trade_date的rqmcl量都为0的ts_code包括: {unique_ts_codes_with_all_zero_rqmcl.tolist()}")

# 假设unique_ts_codes_with_all_zero_rqmcl是你找到的所有在每个trade_date的rqmcl都为0的ts_code列表
# unique_ts_codes_with_all_zero_rqmcl = [...上一步的代码结果...]

# 从df_restricted中删除这些ts_code对应的行
df_restricted_filtered = df_restricted[~df_restricted['ts_code'].isin(unique_ts_codes_with_all_zero_rqmcl)]
df_restricted_filtered['trade_date'] = pd.to_datetime(df_restricted_filtered['trade_date'], format='%Y%m%d')
df_restricted_filtered = df_restricted_filtered.rename(columns={df_restricted_filtered.columns[1]: 'stock_code'})
df_restricted_filtered = df_restricted_filtered.rename(columns={df_restricted_filtered.columns[0]: 'date'})

# 打印新的DataFrame的信息以确认删除操作
df_restricted_filtered.info()

float_a_shares_day = pd.read_csv('/Users/leoliverpool/Desktop/量化投资/期中作业（何康）/float_a_shares_day.csv',encoding = 'gbk')

float_a_shares_day.stack()

def transform(close_adj_day):
    close_adj_day = close_adj_day.T
    close_adj_day.head()
    close_adj_day.columns = close_adj_day.iloc[0]  # 将第 0 行的值设置为列名
    close_adj_day.head()
    close_adj_day = close_adj_day[1:]              # 删除原来的第 0 行
    close_adj_day.head()
    close_adj_day.index.name = None
    close_adj_day.head()
    data_type = close_adj_day.iloc[:, 0]
    print(data_type)
    index_data_type = close_adj_day.index.dtype
    print(index_data_type)
    index_content = close_adj_day.index
    print(index_content)
    close_adj_day.index = pd.to_datetime(close_adj_day.index)
    index_content = close_adj_day.index
    print(index_content)
    return close_adj_day

float_a_shares_day1 = transform(float_a_shares_day)
float_a_shares_day2 = float_a_shares_day1.stack()
float_a_shares_day2 = float_a_shares_day2.reset_index()
# 第一步：转换第一列为日期格式
float_a_shares_day2.iloc[:, 0] = pd.to_datetime(float_a_shares_day2.iloc[:, 0])

# 第二步：设置该日期列为索引
float_a_shares_day2 = float_a_shares_day2.set_index(float_a_shares_day2.columns[0])

# 第三步：重命名第二和第三列
float_a_shares_day2 = float_a_shares_day2.rename(columns={float_a_shares_day2.columns[0]: 'stock_code', float_a_shares_day2.columns[1]: 'shares'})

# 打印修改后的DataFrame以确认更改
print(float_a_shares_day2.head())


float_a_shares_day2 = float_a_shares_day2.rename_axis('date')

# 打印修改后的DataFrame以确认索引名称已更改
print(float_a_shares_day2.head())
# 重设float_a_shares_day2的索引，将其变为一列
float_a_shares_day2 = float_a_shares_day2.reset_index()

# 现在 'date' 是float_a_shares_day2的一列，确保它是日期类型
float_a_shares_day2['date'] = pd.to_datetime(float_a_shares_day2['date'], format='%Y%m%d')
float_a_shares_day2.head()
df_restricted_filtered.head()
float_a_shares_day2.info()
df_restricted_filtered.info()

# 合并两个DataFrame
merged_df = pd.merge(df_restricted_filtered, float_a_shares_day2, 
                     on=['date', 'stock_code'], 
                     how='inner')

# 打印合并后的DataFrame的前几行以确认合并成功
print(merged_df.head())
merged_df.info()

# 从df_restricted_filtered中提取股票代码
stock_codes_original = set(df_restricted_filtered['stock_code'].unique())

# 从merged_df中提取股票代码
stock_codes_merged = set(merged_df['stock_code'].unique())

# 找出在df_restricted_filtered中但不在merged_df中的股票代码
excluded_stock_codes = stock_codes_original - stock_codes_merged

# 打印被排除的股票代码
print(f"被排除的股票代码有: {excluded_stock_codes}")

# 计算新的变量 rqmclzb
merged_df['rqmclzb'] = merged_df['rqmcl'] / merged_df['shares']

# 删除原来的rqmcl和shares列
merged_df.drop(columns=['rqmcl', 'shares'], inplace=True)

# 打印修改后的DataFrame以确认更改
print(merged_df.head())


# %%1.2 筛选有效样本
merged_df.head()

df = merged_df
# 确保rqmclzb列的数据类型正确
df['rqmclzb'] = pd.to_numeric(df['rqmclzb'], errors='coerce')

# 计算每支股票rqmclzb为0的天数
zero_days_count = df[df['rqmclzb'] == 0].groupby('stock_code').size().reset_index(name='zero_days_count')

print(zero_days_count)

average_zero_days = zero_days_count['zero_days_count'].mean()

# 计算zero_days_count的概率分布
#probability_distribution = zero_days_count['zero_days_count'].value_counts(normalize=True)

# 打印结果
print("Average zero days count:", average_zero_days)
#print("Probability distribution of zero days count:")
#print(probability_distribution)

df.info()
# 假设no_margin_days_df已经包含每个股票没有融券的天数统计
# 以下步骤基于这个假设进行

# 筛选出300天以上没有融券的股票代码
stocks_to_exclude = zero_days_count[zero_days_count['zero_days_count'] > 200]['stock_code'].unique()

# 从原始DataFrame df 中去除这些股票
df_filtered = df[~df['stock_code'].isin(stocks_to_exclude)]

# 打印过滤后的DataFrame的信息，确保操作成功
print(df_filtered.info())

# 注意：请确保zero_days_count已经按照之前的指导计算完成，并且包含了'zero_days_count'和'stock_code'列
# df_filtered 现在不包含那些300天以上没有融券的股票
df = df_filtered
merged_df.info()
import pandas as pd

# 假设df是你的DataFrame，包含所有数据
# 请确保你的DataFrame df 已经被正确加载，下面的代码将基于这个假设进行

# 计算每一天没有融券（rqmclzb为0）的股票数量
no_margin_days = df[df['rqmclzb'] == 0].groupby('date').size()

# 将结果转换为DataFrame
no_margin_days_df = no_margin_days.reset_index(name='no_margin_stock_count')

# 打印每一天没有融券的股票数量
print(no_margin_days_df)


# 然后，计算这一数量的平均值
average_no_margin_per_day = no_margin_days_df.mean()

# 打印结果
print(f"平均每一天没有融券的股票数量为: {average_no_margin_per_day}")



pivot_table1 = df_filtered.pivot(index='date', columns='stock_code', values='rqmclzb')

# 打印透视表的前几行以确认更改
print(pivot_table1.head())
pivot_table1.info()
pivot_table1.to_csv('/Users/leoliverpool/Desktop/量化投资/期末作业/factor_rqmclzb(487).csv')





# %%2. 生成相应样本的日度收益率
close_adj_day = pd.read_csv('/Users/leoliverpool/Desktop/量化投资/期中作业（何康）/close_adj_day.csv',encoding = 'gbk')


close_adj_day1 = transform(close_adj_day)
close_adj_day1.head()
# 计算每一行的变化百分比
close_adj_day1  = close_adj_day1.pct_change()
close_adj_day1  = close_adj_day1 .shift(-1)
# 打印计算结果的前几行
print(close_adj_day1 .head())


close_adj_day2 = close_adj_day1.stack()
close_adj_day2 = close_adj_day2.reset_index()
close_adj_day2.head()
# 第一步：转换第一列为日期格式
close_adj_day2.iloc[:, 0] = pd.to_datetime(close_adj_day2.iloc[:, 0])

# 第二步：设置该日期列为索引
close_adj_day2= close_adj_day2.set_index(close_adj_day2.columns[0])

# 第三步：重命名第二和第三列
close_adj_day2 = close_adj_day2.rename(columns={close_adj_day2.columns[0]: 'stock_code', close_adj_day2.columns[1]: 'percentagechange'})

# 打印修改后的DataFrame以确认更改
print(close_adj_day2.head())

close_adj_day2 = close_adj_day2.rename_axis('date')

close_adj_day2 = close_adj_day2.reset_index()
close_adj_day2.head()

# %%3. 生成预处理后的因子值
#去极值、行业市值中性化与zscore
def winsorize_std(data,scale=3): 
    upper = data.mean() + scale*data.std()
    lower = data.mean() - scale*data.std()
    return np.clip(data,lower,upper)


close_day = pd.read_csv('/Users/leoliverpool/Desktop/量化投资/期中作业（何康）/close_day.csv',encoding = 'gbk')
float_a_shares_day = pd.read_csv('/Users/leoliverpool/Desktop/量化投资/期中作业（何康）/float_a_shares_day.csv',encoding = 'gbk')
close_day.head()
float_a_shares_day.head()
close_day = transform(close_day)
float_a_shares_day = transform(float_a_shares_day)
mktvalue = close_day*float_a_shares_day

mktvalue2 = mktvalue.stack()
mktvalue2.head()
mktvalue2 = mktvalue2.reset_index()
# 第一步：转换第一列为日期格式
mktvalue2.iloc[:, 0] = pd.to_datetime(mktvalue2.iloc[:, 0])

# 第二步：设置该日期列为索引
mktvalue2 = mktvalue2.set_index(mktvalue2.columns[0])

# 第三步：重命名第二和第三列
mktvalue2 = mktvalue2.rename(columns={mktvalue2.columns[0]: 'stock_code', mktvalue2.columns[1]: 'mktcap'})

# 打印修改后的DataFrame以确认更改
print(mktvalue2.head())


mktvalue2 = mktvalue2.rename_axis('date')

# 打印修改后的DataFrame以确认索引名称已更改
print(mktvalue2.head())
# 重设float_a_shares_day2的索引，将其变为一列
mktvalue2 = mktvalue2.reset_index()
mktvalue2['date'] = pd.to_datetime(mktvalue2['date'], format='%Y%m%d')


merged_df = df
# 合并两个DataFrame
merged_df1 = pd.merge(merged_df, mktvalue2, 
                     on=['date', 'stock_code'], 
                     how='inner')

# 打印合并后的DataFrame的前几行以确认合并成功
print(merged_df1.head())
merged_df1.info()




cs_indus_code_day = pd.read_csv('/Users/leoliverpool/Desktop/量化投资/期中作业（何康）/cs_indus_code_day.csv',encoding = 'gbk')
cs_indus_code_day = transform(cs_indus_code_day)
cs_indus_code_day2 = cs_indus_code_day.stack()
cs_indus_code_day2.head()
cs_indus_code_day2 = cs_indus_code_day2.reset_index()
# 第一步：转换第一列为日期格式
cs_indus_code_day2.iloc[:, 0] = pd.to_datetime(cs_indus_code_day2.iloc[:, 0])

# 第二步：设置该日期列为索引
cs_indus_code_day2 = cs_indus_code_day2.set_index(cs_indus_code_day2.columns[0])

# 第三步：重命名第二和第三列
cs_indus_code_day2 = cs_indus_code_day2.rename(columns={cs_indus_code_day2.columns[0]: 'stock_code', cs_indus_code_day2.columns[1]: 'classname'})

# 打印修改后的DataFrame以确认更改
print(cs_indus_code_day2.head())


cs_indus_code_day2 = cs_indus_code_day2.rename_axis('date')
cs_indus_code_day2.info() #classname是object

cs_indus_code_day2 = cs_indus_code_day2.reset_index()
cs_indus_code_day2['date'] = pd.to_datetime(cs_indus_code_day2['date'], format='%Y%m%d')


# 合并两个DataFrame
merged_df2 = pd.merge(merged_df1, cs_indus_code_day2, 
                     on=['date', 'stock_code'], 
                     how='inner')

# 打印合并后的DataFrame的前几行以确认合并成功
print(merged_df2.head())
merged_df2.info()

def OlsResid(y,x):
    # axis=1表示在行方向上拼接
    df = pd.concat([y,x],axis = 1)

#    print(df)
    if df.dropna().shape[0]>0:    
        resid = sm.OLS(y,x,missing='drop').fit().resid   
        return resid.reindex(df.index)
    else:
        return y
    
merged_df2['rqmclzb'] = pd.to_numeric(merged_df2['rqmclzb'], errors='coerce')
merged_df2['mktcap'] = pd.to_numeric(merged_df2['mktcap'], errors='coerce')
    
def norm(data,if_neutral): #包含去极值过程

    # data = data.copy()
    """
    数据预处理，标准化
    数据集（这里是固定一个日期，所有的股票数据）是一个dataframe, 其中包括'classname','mktcap'两列，
    分别代表每个股票所属的行业名称以及对应的市值，其他列为因子值
    """    
 
    datax = data.copy()
    if data.shape[0] != 0:
        classname = data['classname']
        mkt = data['mktcap']
        # 删除指定的列，axis = 1代表按列删除
        data = data.drop(['classname','mktcap'],axis = 1)
        ## 去极值, axis = 0代表对每一列数据进行操作
        data = data.apply(lambda x:winsorize_std(x),axis = 0)
        ## 中性化
        if if_neutral:  # 是否中性          
            class_var = pd.get_dummies(classname,columns=['classname'],prefix='classname',
                                       prefix_sep="_", dummy_na=False, drop_first=True)   
            class_var['mktcap'] = np.log(mkt)
            class_var['Intercept'] = 1
            x = class_var                
            x = x.astype(int)
        # 每个因子对所有自变量做回归，得到残差值
            data = data.apply(func = OlsResid, args = (x,), axis = 0)                              
           ## zscore
        data1 = (data - data.mean())/data.std() 
        # 缺失部分补进去
        data1 = data1.reindex(datax.index)
    else:
        data1 = data                  
    return data1

merged_df3 = merged_df2.drop(['date', 'stock_code'],axis = 1)
merged_df3.info()
factor_processed = norm(merged_df3, if_neutral= True)
factor_processed.info()
# 将 'date' 和 'stock_code' 作为新的 DataFrame
info_df = merged_df2[['date', 'stock_code']]

# 将 info_df 与 factor_processed 进行横向拼接
merged_df3 = pd.concat([info_df, factor_processed], axis=1)

# 查看最终结果的前几行
print(merged_df3.head())






pivot_table1 = merged_df3.pivot(index='date', columns='stock_code', values='rqmclzb')

# 打印透视表的前几行以确认更改
print(pivot_table1.head())
pivot_table1.info()

# 将DataFrame保存到特定路径的CSV文件
pivot_table1.to_csv('/Users/leoliverpool/Desktop/量化投资/期末作业/factor_rqmclzb_processed(487).csv')


# %%4. 分层回测
#首先合并出需要的dataframe
close_adj_day2.head()
merged_df3.head()
# 直接在原始 DataFrame 上重命名列
close_adj_day2.rename(columns={'date': 'tradedate', 'stock_code': 'stockcode','percentagechange':'ret'}, inplace=True)
merged_df3.rename(columns={'date': 'tradedate', 'stock_code': 'stockcode'}, inplace=True)




def ifst(x):
    # 如果x.entry_dt为缺失，那么认为这一行数据不是ST
    if pd.isnull(x.entry_dt):
        return 0
    # 满足条件的数据（行记录）不在ST时期内
    elif (x.tradedate < x.entry_dt) |(x.tradedate > x.remove_dt):
        return 0
    else:
        return 1


def GroupTestFactors_1(factors,ret,groups):
    
    # 单个因子的分组分析
    # factors = fnorm.copy();groups = 10
    '''
    Parameters
    ----------
    factors : 数值型
        dataframe包括3列，列名分别为tradedate, stockcode和因子的名称，这里为mom.
        index为默认的index
    ret : 数值型
        dataframe包括3列，列名分别为tradedate, stockcode和ret，ret为股票收益率.
        index为默认的index
    
    groups : TYPE
        可以取数字，5或者10，代表按照quantile分成5组或者10组.

    Returns
    -------
    '''
    fnames = factors.columns
    fall = pd.merge(factors,ret,left_on = ['stockcode','tradedate'],right_on = ['stockcode','tradedate'])
    
    # 提取因子的名称
    f= fnames[2]
    if ((f != 'stockcode')&(f != 'tradedate')):
            # 计算因子收益率
            fuse = fall.copy()[['stockcode','tradedate','ret',f]]
            # 计算每只股票在每个日期所属的group，按照因子值从小到大等分，注意这里
            # 每组的标志分别为1，2，3，4，5
            fuse['groups'] = fuse[f].groupby(fuse.tradedate).transform(lambda x: np.ceil(x.rank()/(len(x)/groups)))
            # 将每个日期每个group中的股票等权重构造投资组合，并计算组合收益率
            result = fuse.groupby(['tradedate','groups']).apply(lambda x:x.ret.mean())
            # 构建一个透视表，使用tradedate, groups和收益率
            result = result.unstack().reset_index()
            # 构建多空投资组合
            # 如果第五组的收益高于第一组，构建高减低投资组合，保存每个日期属于第五组的股票
            if result.iloc[:,-1].mean() > result.iloc[:,-groups].mean():
                result['L-S'] = result.iloc[:,-1] - result.iloc[:,-groups]
                #stock_l = fuse.loc[fuse.groups == 1]
                #stock_l = fuse.loc[fuse.groups == groups]
            else: # 如果第一组的收益高于第五组，构建低减高投资组合，保存每个日期属于第一组的股票
                result['S-L'] = result.iloc[:,-groups] - result.iloc[:,-1]
                #stock_l = fuse.loc[fuse.groups == groups]
                #stock_l = fuse.loc[fuse.groups == 1]
    result.insert(0,'factor',f)
    Groupret = result.copy()
    
    # 对于每个factor，计算每一分组的累积收益率
    Groupnav = Groupret.iloc[:,2:].apply(lambda x:(1 + x).cumprod())
    # 这里的Groupnav没有包括'tradedate','factor'，加上'tradedate','factor'这两列
    Groupnav = pd.concat([Groupret[['tradedate','factor']],Groupnav],axis = 1)
    
    return Groupnav    


def plotnav(Groupnav):
    """
    GroupTest作图
    """
    for f in Groupnav.factor.unique():  # f = Groupnav.factor.unique()[0]
        fnav = Groupnav.loc[Groupnav.factor == f, :].set_index('tradedate').iloc[:, 1:]
        groups = fnav.shape[1] - 1  # 减1因为最后一个是S-L，不是分组
        lwd = [2] * groups + [2]  # 线宽，最后一个是S-L
        ls = ['-'] * groups + ['--']  # 线型，最后一个S-L可以用不同的线型表示
        
        plt.figure(figsize=(10, 5))
        for i in range(groups + 1):  # 包括S-L
            plt.plot(fnav.iloc[:, i], linewidth=lwd[i], linestyle=ls[i])
        
        # 生成标签列表，前5个是1-5，最后一个是S-L
        labels = list(range(1, groups + 1)) + ['S-L']
        plt.legend(labels)
        plt.title('Factor Group Test: ' + f, fontsize=20)



result1 = GroupTestFactors_1(merged_df3,close_adj_day2,5)
plotnav(result1)



# %%5. IC值分析
def getICSeries(factors, ret, method):
    # 合并因子和收益率数据，然后去除 'stockcode' 列
    fall = pd.merge(factors, ret, on=['tradedate', 'stockcode']).drop(columns='stockcode')
    
    # 计算并返回每个交易日的信息系数（IC）
    ic_series = fall.groupby('tradedate').apply(lambda x: x.corr(method=method).at['ret', 'rqmclzb'])
    icall = ic_series.reset_index(name='IC')

    return icall.set_index('tradedate')

            
          
def plotIC1(ic_f):
    """
    IC作图
    """
    fnames = ic_f.columns
    for fname in fnames:
        fig = plt.figure(figsize=(12, 6))  # 增加图表尺寸
        ax = fig.add_subplot(111)
        ax.bar(np.arange(ic_f.shape[0]), ic_f[fname], color='darkred')
        ax1 = ax.twinx()
        ax1.plot(np.arange(ic_f.shape[0]), ic_f[fname].cumsum(), color='orange')
        
        xtick = np.arange(0, ic_f.shape[0], 36)  # 增加刻度间隔
        xticklabel = pd.Series(ic_f.index[xtick]).astype("str").str[:7]
        
        ax.set_xticks(xtick)
        ax.set_xticklabels(xticklabel, rotation=45)  # 旋转刻度标签
        ax.set_title(fname + ' IC Average = {}, Annualized ICIR = {}'.format(
            round(ic_f[fname].mean(), 4),
            round(ic_f[fname].mean() / ic_f[fname].std() * np.sqrt(12), 4)))
        plt.tight_layout()  # 自动调整子图参数，使之填充整个图表区域。




merged_df3.head()
close_adj_day2.head()
result3 = getICSeries(merged_df3,close_adj_day2,'spearman')
result3.head()
# 假设result3是已经加载到环境中的DataFrame
result3 = result3.rename(columns={'IC': 'rqmclzb'})

# 查看更改后的DataFrame以确认列名已经被正确修改
print(result3.head())
plotIC1(result3)
result3 = result3.rename(columns={'rqmclzb': 'IC'})






def GroupTestFactors_2(factors,ret,groups):
    
    # 单个因子的分组分析
    # factors = fnorm.copy();groups = 10
    '''
    Parameters
    ----------
    factors : 数值型
        dataframe包括3列，列名分别为tradedate, stockcode和因子的名称，这里为mom.
        index为默认的index
    ret : 数值型
        dataframe包括3列，列名分别为tradedate, stockcode和ret，ret为股票收益率.
        index为默认的index
    
    groups : TYPE
        可以取数字，5或者10，代表按照quantile分成5组或者10组.

    Returns
    -------
    '''
    fnames = factors.columns
    fall = pd.merge(factors,ret,left_on = ['stockcode','tradedate'],right_on = ['stockcode','tradedate'])
    
    # 提取因子的名称
    f= fnames[2]
    if ((f != 'stockcode')&(f != 'tradedate')):
            # 计算因子收益率
            fuse = fall.copy()[['stockcode','tradedate','ret',f]]
            # 计算每只股票在每个日期所属的group，按照因子值从小到大等分，注意这里
            # 每组的标志分别为1，2，3，4，5
            fuse['groups'] = fuse[f].groupby(fuse.tradedate).transform(lambda x: np.ceil(x.rank()/(len(x)/groups)))
            # 将每个日期每个group中的股票等权重构造投资组合，并计算组合收益率
            result = fuse.groupby(['tradedate','groups']).apply(lambda x:x.ret.mean())
            # 构建一个透视表，使用tradedate, groups和收益率
            result = result.unstack().reset_index()
            # 构建多空投资组合
            # 如果第五组的收益高于第一组，构建高减低投资组合，保存每个日期属于第五组的股票
            if result.iloc[:,-1].mean() > result.iloc[:,-groups].mean():
                result['L-S'] = result.iloc[:,-1] - result.iloc[:,-groups]
                #stock_l = fuse.loc[fuse.groups == 1]
                #stock_l = fuse.loc[fuse.groups == groups]
            else: # 如果第一组的收益高于第五组，构建低减高投资组合，保存每个日期属于第一组的股票
                result['S-L'] = result.iloc[:,-groups] - result.iloc[:,-1]
                #stock_l = fuse.loc[fuse.groups == groups]
                #stock_l = fuse.loc[fuse.groups == 1]
    result.insert(0,'factor',f)
    Groupret = result.copy()
    
    # 对于每个factor，计算每一分组的累积收益率
    Groupnav = Groupret.iloc[:,2:].apply(lambda x:(1 + x).cumprod())
    # 这里的Groupnav没有包括'tradedate','factor'，加上'tradedate','factor'这两列
    Groupnav = pd.concat([Groupret[['tradedate','factor']],Groupnav],axis = 1)
    
    return Groupret   

#输出结果
result4 = GroupTestFactors_2(merged_df3,close_adj_day2,5)
result4 = result4.drop(['factor', 'S-L'],axis = 1)

result3 = result3.reset_index()
result5 = pd.merge(result3, result4, on='tradedate')

result5.to_csv('/Users/leoliverpool/Desktop/量化投资/期末作业/factor_rqmclzb_IC_LevRet(487).csv')





