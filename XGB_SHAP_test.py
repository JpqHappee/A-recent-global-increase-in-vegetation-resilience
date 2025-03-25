import os
import glob
import rasterio
import numpy as np
import xgboost as xgb
import shap  # 引入SHAP库
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def read_tiff_files(file_pattern):
    file_list = glob.glob(file_pattern)
    if not file_list:
        raise FileNotFoundError(f"No files found for pattern: {file_pattern}")

    band_stack = []
    for file in file_list:
        with rasterio.open(file) as src:
            data = src.read(1)
            band_stack.append(data)
    return np.stack(band_stack, axis=0)

# 定义文件路径模式
file_paths = [
    r'input_X1_data.tif',
   ...
]

# 读取自变量的TIFF影像数据
features = []
for file_path in file_paths:
    feature_data = read_tiff_files(file_path)
    features.append(feature_data)

# 将自变量合并为一个四维矩阵（时间，变量，高度，宽度）
X = np.stack(features, axis=1)

# 读取因变量的TIFF影像数据
target_files_pattern = r'input_y_data.tif'
y = read_tiff_files(target_files_pattern)

# 获取数据维度
num_timesteps, num_features, height, width = X.shape

# 初始化特征重要性数组
importance_array = np.zeros((num_features, height, width))
importance_means = np.zeros(num_features)  # 用于存储每个特征的平均重要性

# 遍历每个像素点
for i in range(height):
    for j in range(width):
        # 提取该像素点的时间序列数据
        X_pixel = X[:, :, i, j]
        y_pixel = y[:, i, j]

        # 去除包含NaN值的样本
        mask = ~np.isnan(y_pixel) & ~np.isnan(X_pixel).any(axis=1)
        X_pixel_clean = X_pixel[mask]
        y_pixel_clean = y_pixel[mask]

        if len(y_pixel_clean) < 10:  # 如果清理后的数据样本不足，跳过该像素点
            continue

        # 标准化输入数据
        scaler = StandardScaler()
        X_pixel_clean = scaler.fit_transform(X_pixel_clean)

        # 将数据分为训练集和测试集
        train_size = int(0.7 * len(y_pixel_clean))
        X_train, X_test = X_pixel_clean[:train_size], X_pixel_clean[train_size:]
        y_train, y_test = y_pixel_clean[:train_size], y_pixel_clean[train_size:]

        # 创建并训练XGBoost回归模型
        model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
        model.fit(X_train, y_train)

        # 计算SHAP值
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_train)

        # 计算每个特征的平均绝对SHAP值，作为特征重要性
        #importance = np.mean(np.abs(shap_values.values), axis=0)
        # 计算每个特征的平均 SHAP 值
        #importance = np.mean(shap_values.values, axis=0)
        importance_array[:, i, j] = importance  # 将特征重要性保存到数组中
        importance_means += importance  # 累积每个特征的平均重要性

# 计算平均特征重要性
importance_means /= (height * width)

# 绘制特征重要性排序图
feature_names = ['Pre', 'SM', 'Tmp', 'GPP', 'POP', 'CO2']
plt.figure(figsize=(10, 6))
plt.barh(feature_names, importance_means, color='skyblue')
plt.xlabel("Mean Absolute SHAP Value (Feature Importance)")
plt.title("Feature Importance Ranking")
plt.gca().invert_yaxis()  # 使最高重要性的特征排在最上方
plt.show()

# 确保输出目录存在
output_directory = r'output/'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# 保存特征重要性结果为TIFF文件
output_files = [
    r'output_X1_data.tif',
    ...,
    ...,

]

# 使用一个样本文件来获取元数据
sample_file = glob.glob(file_paths[0])[0]
with rasterio.open(sample_file) as src:
    profile = src.profile

for k, output_file in enumerate(output_files):
    with rasterio.open(
            output_file,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=importance_array[k].dtype,
            crs=profile['crs'],
            transform=profile['transform']
    ) as dst:
        dst.write(importance_array[k], 1)

print('Feature importance calculation with SHAP and saving complete.')
