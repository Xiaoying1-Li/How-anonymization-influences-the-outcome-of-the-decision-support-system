import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from recordlinkage.index import Block
from recordlinkage.base import BaseCompareFeature
import recordlinkage as rl
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_data(file_a, file_b):
    return pd.read_csv(f'../data/anonymized_result/{file_a}', sep=";"), pd.read_csv(f'../data/anonymized_result/{file_b}', sep=";")

def process_range_int(range_string):
    return [int(i) for i in range_string.replace(' ', '').split('-')]

def process_range_float(range_string):
    return [float(i) for i in range_string.replace(' ', '').split('-')]

def preprocess_data(df):
    for col in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']:
        if col == 'oldpeak':
            df[col] = df[col].apply(process_range_float).tolist()
        else:
            df[col] = df[col].apply(process_range_int).tolist()
    return df


def fill_missing_values(df_a, df_b):
    # 替换 * 为 NaN
    df_a.replace('*', np.nan, inplace=True)
    df_b.replace('*', np.nan, inplace=True)

    # 找出含有 NaN 值的列
    cols_with_nan = set(df_a.columns[df_a.isna().any()].tolist() + df_b.columns[df_b.isna().any()].tolist())

    # 将这些列转换为 float64 类型，计算众数并填充 NaN 值
    for df in [df_a, df_b]:
        for col in cols_with_nan:
            df[col] = df[col].astype('float64')
        mode_values = df.mode().iloc[0]
        df.fillna(mode_values, inplace=True)
        for col in cols_with_nan:
            df[col] = df[col].astype('int64')

    return df_a, df_b


def convert_range_columns(df):
    # 定义一个正则表达式模式来识别范围值
    range_pattern = r'^\d+(\.\d+)?-\d+(\.\d+)?$'

    # 定义一个通用函数来处理列的值
    def convert_value(value, col):
        if not pd.isna(value) and not pd.Series([value]).str.match(range_pattern).any():
            try:
                if col == 'oldpeak':
                    num = float(value)
                    return f'0.0-{num}'
                else:
                    num = int(value)
                    return f'0-{num}'
            except ValueError:
                return value
        return value

    # 应用函数到指定列
    columns_to_convert = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    for col in columns_to_convert:
        df[col] = df[col].apply(lambda x: convert_value(x, col))

    return df

def process_dataframes(df_a, df_b):
    df_a = convert_range_columns(df_a)
    df_b = convert_range_columns(df_b)
    return df_a, df_b

def find_matching_pairs_optimized(df_a, df_b):
    matching_pairs = []

    for idx_a, row_a in df_a.iterrows():
        matching_records_b = df_b[(df_b['sex'] == row_a['sex']) & (df_b['target'] == row_a['target'])]
        if not matching_records_b.empty:
            for idx_b, row_b in matching_records_b.iterrows():
                matching_pairs.append((idx_a, idx_b))

    return matching_pairs

class CompareEuclideanDistance(BaseCompareFeature):
    def __init__(self, left_on, right_on, *args, **kwargs):
        super().__init__(left_on, right_on, *args, **kwargs)

    def _compute_vectorized(self, s1, s2):
        overlap_ratios = []

        for left, right in zip(s1, s2):

            overlap_min = max(left[0], right[0])
            overlap_max = min(left[1], right[1])

                # 如果重叠部分的最小值大于最大值，表示没有重叠
            if overlap_min> overlap_max:
                overlap_ratios.append(0.0)
            elif left[0] == right[0] or left[1] == right[1]:
                overlap_ratios.append(1.0)
            elif left[0] == right[1] or left[1] == right[0]:
                overlap_ratios.append(1.0)

            else:
                # 计算重叠部分的长度

                s1_mid = (left[0]+left[1] ) /2
                s2_mid = (right[0] +right[1]) /2
                distance = np.sqrt((s1_mid - s2_mid) ** 2)
                overlap_ratio = 1 - distance/(max(left[1], right[1]) - min(left[0], right[0]))

                overlap_ratios.append(overlap_ratio)

        return pd.Series(overlap_ratios)


def compare_records(candidate_pairs, df_a, df_b):
    comp = rl.Compare()
    features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    for feature in features:
        comp.add(CompareEuclideanDistance(feature, feature))
    for feature in ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'target']:
        comp.exact(feature, feature)
    return comp.compute(candidate_pairs, df_a, df_b)

def calculate_accuracy1():
    # 读取数据
    comparison_result_df = pd.read_csv(f'../data/record linkage/comparison_result.csv')
    new_df = pd.read_csv(f'../data/process/duplicatedindex.csv')

    # 初始化变量
    thresholds = np.linspace(7, 14, 14)
    accuracies = []

    # 遍历每个阈值，计算准确率
    for threshold in thresholds:
        total_prediction = comparison_result_df[comparison_result_df['total_similarity'] >= threshold]

        # 进行inner join
        correct_prediction = pd.merge(total_prediction, new_df, left_on=['index_a', 'index_b'],
                                      right_on=['index_a', 'index_b'])

        # 计算准确率
        total_pred_count = len(total_prediction)//2
        correct_pred_count = len(correct_prediction)
        accuracy = correct_pred_count / total_pred_count if total_pred_count > 0 else 0
        accuracies.append(accuracy)
        print(f'Threshold: {threshold}, Total Predictions: {total_pred_count}, Correct Predictions: {correct_pred_count}, Accuracy: {accuracy}')

    # 绘制准确率变化图
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, accuracies, marker='o')
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Threshold')
    plt.grid(True)
    plt.show()

def main():
    df_a, df_b = load_data('anonymized_a.data', 'anonymized_b.data')

    df_a, df_b = fill_missing_values(df_a,df_b)
    df_a, df_b = process_dataframes(df_a, df_b)
    df_a, df_b =preprocess_data(df_a), preprocess_data(df_b)

    matching_pairs = find_matching_pairs_optimized(df_a, df_b)
    df_matching_pairs = pd.DataFrame(matching_pairs, columns=['index_a', 'index_b'])
    df_matching_pairs.to_csv(f'../data/record linkage/candidate_paris.csv', index=False)

    total_comparison_pairs = len(df_a) * len(df_b)
    reduced_comparison_pairs = len(df_matching_pairs)
    blocking_efficiency = (total_comparison_pairs - reduced_comparison_pairs) / total_comparison_pairs
    print("Total Comparison Pairs:", total_comparison_pairs)
    print("Reduced Comparison Pairs:", reduced_comparison_pairs)
    print("Blocking Efficiency: {:.2%}".format(blocking_efficiency))

    matching_pairs1_index = pd.MultiIndex.from_frame(df_matching_pairs)
    matching_pairs1_index.names = ['index_a', 'index_b']

    variable_names = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak','sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'target']
    comparison_result = compare_records(matching_pairs1_index, df_a, df_b)
    comparison_result.columns = variable_names
    comparison_result = comparison_result.reset_index()

    #print(comparison_result)
    comparison_result['index_a'] = comparison_result['index_a'].map(df_a['index'])
    comparison_result['index_b'] = comparison_result['index_b'].map(df_b['index'])
    comparison_result.set_index(['index_a', 'index_b'], inplace=True)

    comparison_result['total_similarity'] = comparison_result[variable_names].sum(axis=1)

    comparison_result.to_csv(f'../data/record linkage/comparison_result.csv', index = True)
    calculate_accuracy1()

if __name__ == "__main__":
    main()

## 在matched中：计算*的数量和comparescore的对比
#135_a,104_b