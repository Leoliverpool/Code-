import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


# %%0. 函数
# 定义 EMA 和 XMA 函数
# %%0.1 EMA函数
def EMA(series, N):
    ema = []  # 用于存储EMA值的列表
    multiplier = 2 / (N + 1)  # 计算EMA的乘数因子
    valid_series = series.dropna()  # 去除数据中的缺失值

    # 确保初始的EMA是前N天的简单平均，确保包含N个非缺失值
    initial_index = valid_series.index[N-1] if len(valid_series) >= N else None

    if initial_index:
        # 计算初始的EMA，使用前N天的简单平均
        ema_initial = valid_series[:N].mean()
        # 在结果列表中填充NaN值直到初始EMA位置
        ema = [np.nan] * (series.index.get_loc(initial_index) ) + [ema_initial]
        
        # 从第N天开始计算EMA
        for price in valid_series[N:]:
            # 根据公式计算EMA
            ema.append((price - ema[-1]) * multiplier + ema[-1])
    else:
        # 如果有效数据不足N个，结果全为NaN
        ema = [np.nan] * len(series)

    # 返回一个与输入series具有相同索引的pd.Series
    return pd.Series(ema, index=series.index)



# %%0.2 新XMA函数 （用于计算某一特定日期下的XMA）
def XMA1(series, N):
    result = []
    half_window = N // 2 + 1 if N % 2 == 0 else math.floor(N / 2) + 1
    predictnumber = N // 2 - 1 if N % 2 == 0 else math.floor(N / 2)
    
    
    for i in range(len(series) - half_window, len(series)):
        past_data = series[ i-half_window+1:i+1].tolist()
        future_data = []
        
        for j in range(predictnumber):
            if i + j + 1 < len(series):
                future_data.append(series[i + j + 1])  
                past_data.append(series[i + j + 1])
                
            else:
                future_value = np.mean(past_data[j:])
                future_data.append(future_value)
                past_data.append(future_value)
                    
        past_data = series[i-half_window+1:i+1].tolist()
        full_window_data = past_data + future_data   
        xma_value = np.mean(full_window_data)
        result.append(xma_value)
    
    return result





# %%0.3 新XMA函数 （用于计算某一特定日期下的XMA）
def XMA2(series, N):

    predictnumber = N // 2 - 1 if N % 2 == 0 else math.floor(N / 2)
    past_data = series
    
    for j in range(predictnumber):
        future_value = np.mean(past_data[j:])
        past_data.append(future_value)
                    
    xma_value = np.mean(past_data)
    result1 = xma_value
    
    
    return result1


data = pd.read_excel('/Users/leoliverpool/Desktop/银万实习/1min_所有数据.xlsx')

data['date'] = pd.to_datetime(data['trade_time'])
data.set_index('date', inplace=True)
data.drop(columns=['trade_time'], inplace=True)


# List of dates and the corresponding adjustments
adjustments = [
    ('2024-01-17', 0.069),
    ('2023-01-13', 0.064),
    ('2022-01-18', 0.075),
    ('2021-01-15', 0.072),
    ('2019-12-10', 0.062),
    ('2019-01-15', 0.059),
    ('2018-01-22', 0.046),
    ('2017-01-20', 0.055),
    ('2016-01-19', 0.051),
    ('2015-01-19', 0.035),
    ('2014-01-20', 0.048),
    ('2012-12-17', 0.033)
]

# Applying the adjustments
for date, adjustment in adjustments:
    data.loc[:date, ['close', 'open', 'high', 'low']] -= adjustment

#data.to_excel('/Users/leoliverpool/Desktop/银万实习/已前复权_1min_所有数据.xlsx', index=True)