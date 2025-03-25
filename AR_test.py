import pandas as pd
import numpy as np
import statsmodels.api as sm

np.seterr(divide='ignore', invalid='ignore')  # 忽略0不能做被除数的情况

# 读取CSV文件
df = pd.read_csv("inputdata_table.csv")

# 设置滑动窗口大小
window_size = 60  # 3年滑动窗口，每年12个月，共计60个数据点

# 创建一个列表用于存储一阶自相关系数和残差
autocorr_data = []  # 存储自相关系数的列表
residual_data = []  # 存储残差的列表

# 计算每个滑动窗口的一阶自相关系数和残差
for i in range(len(df) - window_size + 1):
    window_data = df.iloc[i:i + window_size, :]  # 保留原始列顺序

    window_autocorr_data = {}
    window_residual_data = {}

    for column in window_data.columns:
        valid_data = window_data[column].dropna()

        if len(valid_data) > 1 and (valid_data != 0).any():  # 确保窗口内有足够的有效数据进行计算
            # 计算一阶自相关系数
            autocorr = sm.tsa.acf(valid_data, fft=False, nlags=1)[1]
            window_autocorr_data[column] = autocorr

            # 计算残差（使用差分计算）
            residual = np.diff(valid_data.values)
            window_residual_data[column] = residual.mean() if len(residual) > 0 else 0
        else:
            window_autocorr_data[column] = 0  # 若数据无效，则输出0
            window_residual_data[column] = 0  # 若数据无效，则输出0

    autocorr_data.append(window_autocorr_data)
    residual_data.append(window_residual_data)

# 将列表转换为DataFrame
autocorr_df = pd.DataFrame(autocorr_data)
residual_df = pd.DataFrame(residual_data)

# 保持与原始数据相同的列顺序
autocorr_df = autocorr_df.reindex(columns=df.columns)
residual_df = residual_df.reindex(columns=df.columns)

# 保存结果到CSV文件
autocorr_df.to_csv("autocorrelation_table.csv", index=False)
residual_df.to_csv("residual_table.csv", index=False)

# 输出结果
print("一阶自相关系数结果已保存到文件")
print("残差结果已保存到文件")
