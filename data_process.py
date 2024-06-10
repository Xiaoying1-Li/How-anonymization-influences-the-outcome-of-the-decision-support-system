import pandas as pd
import random
import numpy as np
import os

# 读取CSV文件
data = pd.read_csv('data/heart.csv')

# 计算要重复的数据数量
num_duplicates = int(0.2 * len(data))

# 随机选择要重复的数据索引
duplicate_indices = np.random.choice(data.index, num_duplicates, replace=False)

# 根据索引复制数据
duplicated_data = data.loc[duplicate_indices].copy()

# 将复制的数据追加到原始数据末尾
data_with_duplicates = pd.concat([data, duplicated_data])

# 打印结果
#print(data_with_duplicates)

# 保存带有重复记录的新数据集
data_with_duplicates.to_csv('data/process/heart_with_duplicates.csv', index=False)

shuffled_data = data_with_duplicates.sample(frac=1, random_state=42)

# 计算数据集的中间位置
mid_point = len(shuffled_data) // 2

# 将数据集分成两个子数据集
heart_disease_a = shuffled_data.iloc[:mid_point].reset_index(drop=True)
heart_disease_b = shuffled_data.iloc[mid_point:].reset_index(drop=True)

# 对数据集去重
heart_disease_a = heart_disease_a.drop_duplicates().reset_index(drop=True)
heart_disease_b = heart_disease_b.drop_duplicates().reset_index(drop=True)

# 为每个数据集添加index列的值
heart_disease_a['index'] = [str(i) + '_a' for i in range(1, 1 + len(heart_disease_a))]
heart_disease_b['index'] = [str(i) + '_b' for i in range(1, 1 + len(heart_disease_b))]

# 保存两个数据集
heart_disease_a.to_csv('data/process/heart_disease_a.csv', index=False)
heart_disease_b.to_csv('data/process/heart_disease_b.csv', index=False)

heart_disease_a = pd.DataFrame(heart_disease_a)
heart_disease_b = pd.DataFrame(heart_disease_b)

# 定义要匹配的列
merge_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
                 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

# 进行内连接合并
merged_df = pd.merge(heart_disease_a, heart_disease_b, on=merge_columns, how='inner', suffixes=('_a', '_b'))
result_df = merged_df[['index_a', 'index_b']]

result_df.to_csv('data/process/duplicatedindex.csv', index = False)
# 打印合并后的数据框
print(result_df)