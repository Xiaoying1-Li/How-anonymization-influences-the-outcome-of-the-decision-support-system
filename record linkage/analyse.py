import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
duplicated_df = pd.read_csv('../data/process/duplicatedindex.csv')
comparison_result_df = pd.read_csv('../data/record linkage/comparison_result.csv')
anonymized_a_df = pd.read_csv('../data/anonymized_result/anonymized_a.data',sep=';')
anonymized_a_df.to_csv('../data/anonymized_result/anonymized_a.csv', index=False,sep=',')
anonymized_b_df = pd.read_csv('../data/anonymized_result/anonymized_b.data',sep=';')
anonymized_b_df.to_csv('../data/anonymized_result/anonymized_b.csv', index=False,sep=',')

# 合并数据
merged_df = pd.merge(duplicated_df, comparison_result_df, on=['index_a', 'index_b'], how='inner')


# 计算'*'占比
def calculate_star_count(df):
    return df.apply(lambda row: row.str.count('\*').sum(), axis=1)

# 构建新DataFrame
new_df = pd.DataFrame({
    'index_a': merged_df['index_a'],
    'index_b': merged_df['index_b'],
    'total_similarity': merged_df['total_similarity'],
    'total_star_a': calculate_star_count(pd.merge(merged_df, anonymized_a_df, left_on='index_a', right_on='index', how='inner')),
    'total_star_b': calculate_star_count(pd.merge(merged_df, anonymized_b_df, left_on='index_b', right_on='index', how='inner')),
})

# 计算percentage_star
new_df['percentage_star'] = (new_df['total_star_a'] + new_df['total_star_b']) / 28

# 绘图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
ax1.scatter(new_df['total_similarity'], new_df['percentage_star'])
ax1.set_title('Percentage Star vs. Total Similarity (Scatter Plot)')
ax1.set_xlabel('Total Similarity')
ax1.set_ylabel('Percentage Star')
ax1.set_xlim(10.5, 14.5)

ax2.bar(new_df['total_similarity'].value_counts().sort_index().index, new_df['total_similarity'].value_counts().sort_index().values)
ax2.set_title('Count of Data Points for Each Total Similarity Value (Bar Plot)')
ax2.set_xlabel('Total Similarity')
ax2.set_ylabel('Count')
ax2.set_xlim(10.5, 14.5)

plt.tight_layout()
plt.show()