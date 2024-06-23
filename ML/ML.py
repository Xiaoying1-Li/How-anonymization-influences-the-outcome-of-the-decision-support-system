import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix,accuracy_score,roc_curve,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

ml = pd.read_csv(f'../data/heart.csv')
#print(ml)
ml.info()

#******************************* Basic information ***************************************#
# Define the continuous features
continuous_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# Identify the features to be converted to object data type
features_to_convert = [feature for feature in ml.columns if feature not in continuous_features]

# Convert the identified features to object data type
ml[features_to_convert] = ml[features_to_convert].astype('object')

ml.info()
print(ml.describe().T)
print(ml.describe(include = 'object'))

#******************************* EDA ***************************************#

# Filter out continuous features for the univariate analysis
df_continuous = ml[continuous_features]

# 创建子图
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

# 循环绘制每个连续特征的直方图
for i, col in enumerate(df_continuous.columns):
    x = i // 3
    y = i % 3

    # 绘制直方图
    values, bins, patches = ax[x, y].hist(df_continuous[col], bins='auto', color='green', alpha=0.6, edgecolor='none')

    # 设置标签和刻度
    ax[x, y].set_xlabel(col, fontsize=15)
    ax[x, y].set_ylabel('Count', fontsize=12)
    ax[x, y].set_xticks(np.round(bins, 1))
    ax[x, y].set_xticklabels(ax[x, y].get_xticks(), rotation=45)
    ax[x, y].grid(color='lightgrey')

    # 在每个柱子上添加高度注释
    for j, p in enumerate(patches):
        height = p.get_height()
        ax[x, y].annotate('{}'.format(int(height)), (p.get_x() + p.get_width() / 2, height + 1),
                          ha='center', fontsize=10, fontweight="bold")

    # 添加均值和标准差文本框
    textstr = '\n'.join((
        r'$\mu=%.2f$' % df_continuous[col].mean(),
        r'$\sigma=%.2f$' % df_continuous[col].std()
    ))
    ax[x, y].text(0.75, 0.9, textstr, transform=ax[x, y].transAxes, fontsize=12, verticalalignment='top',
                  color='white', bbox=dict(boxstyle='round', facecolor='green', edgecolor='white', pad=0.5))

# 隐藏多余的子图
ax[1, 2].axis('off')

# 添加标题和调整布局
plt.suptitle('Distribution of Continuous Variables', fontsize=20)
plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.show()

# Filter out categorical features for the univariate analysis
categorical_features = ml.columns.difference(continuous_features)
df_categorical = ml[categorical_features]

# Set up the subplot for a 4x2 layout
fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(13, 15))

# Loop to plot bar charts for each categorical feature in the 4x2 layout
for i, col in enumerate(categorical_features):
    row = i // 2
    col_idx = i % 2

    # Calculate frequency percentages
    value_counts = ml[col].value_counts(normalize=True).mul(100).sort_values()

    # Plot bar chart
    value_counts.plot(kind='barh', ax=ax[row, col_idx], width=0.8, color='green', alpha = 0.6, edgecolor = 'none')

    # Add frequency percentages to the bars
    for index, value in enumerate(value_counts):
        ax[row, col_idx].text(value, index, str(round(value, 1)) + '%', fontsize=15, weight='bold', va='center')

    ax[row, col_idx].set_xlim([0, 95])
    ax[row, col_idx].set_xlabel('Frequency Percentage', fontsize=12)
    ax[row, col_idx].set_title(f'{col}', fontsize=12)

ax[4, 1].axis('off')
plt.suptitle('Distribution of Categorical Variables', fontsize=20)
plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.show()

#******************************* Outlier ***************************************#

# 创建子图
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

# 循环绘制每个连续特征的箱线图
for i, col in enumerate(df_continuous.columns):
    x = i // 3
    y = i % 3

    # 绘制箱线图
    ax[x, y].boxplot(df_continuous[col], patch_artist=True, boxprops=dict(facecolor='green', color='green'))

    # 设置标签和刻度
    ax[x, y].set_xlabel(col, fontsize=15)
    ax[x, y].set_ylabel('Value', fontsize=12)
    ax[x, y].grid(color='lightgrey')

    # 添加均值和标准差文本框
    textstr = '\n'.join((
        r'$\mu=%.2f$' % df_continuous[col].mean(),
        r'$\sigma=%.2f$' % df_continuous[col].std()
    ))
    ax[x, y].text(0.75, 0.9, textstr, transform=ax[x, y].transAxes, fontsize=12, verticalalignment='top',
                  color='white', bbox=dict(boxstyle='round', facecolor='green', edgecolor='white', pad=0.5))

    # 计算异常值数量
    Q1 = df_continuous[col].quantile(0.25)
    Q3 = df_continuous[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df_continuous[col] < (Q1 - 1.5 * IQR)) | (df_continuous[col] > (Q3 + 1.5 * IQR))).sum()

    # 显示异常值数量
    outlier_text = 'Outliers: {}'.format(outliers)
    ax[x, y].text(0.75, 0.1, outlier_text, transform=ax[x, y].transAxes, fontsize=12, verticalalignment='bottom',
                  color='white', bbox=dict(boxstyle='round', facecolor='green', edgecolor='white', pad=0.5))

# 隐藏多余的子图
if len(df_continuous.columns) < 6:
    for i in range(len(df_continuous.columns), 6):
        x = i // 3
        y = i % 3
        ax[x, y].axis('off')

# 添加标题和调整布局
plt.suptitle('Outlier of Continuous Variables', fontsize=20)
plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.show()

Q1 = ml[continuous_features].quantile(0.25)
Q3 = ml[continuous_features].quantile(0.75)
IQR = Q3 - Q1
outliers_count_specified = ((ml[continuous_features] < (Q1 - 1.5 * IQR)) | (ml[continuous_features] > (Q3 + 1.5 * IQR))).sum()
outliers_mask = (ml[continuous_features] < (Q1 - 1.5 * IQR)) | (ml[continuous_features] > (Q3 + 1.5 * IQR))

print(outliers_count_specified)
df_no_outliers = ml[~outliers_mask.any(axis=1)]
df_no_outliers.info()
features_to_convert = ['sex', 'fbs', 'exang', 'slope', 'ca', 'target']
for feature in features_to_convert:
    df_no_outliers[feature] = df_no_outliers[feature].astype(int)
#******************************* Model prepration ***************************************#

y = df_no_outliers['target']
X = df_no_outliers.drop('target', axis= 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print(y_test.unique())
print(Counter(y_train))

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#******************************* Logistic Regression ***************************************#

m1 = 'Logistic Regression'
lr = LogisticRegression()
model = lr.fit(X_train, y_train)
lr_predict = lr.predict(X_test)
lr_conf_matrix = confusion_matrix(y_test, lr_predict)
lr_acc_score = accuracy_score(y_test, lr_predict)
print("confussion matrix")
print(lr_conf_matrix)
print("\n")
print("Accuracy of Logistic Regression:",lr_acc_score*100,'\n')
print(classification_report(y_test,lr_predict))

#******************************* Random Forest ***************************************#

m3 = 'Random Forest Classfier'
rf = RandomForestClassifier(n_estimators=20, random_state=12,max_depth=5)
rf.fit(X_train,y_train)
rf_predicted = rf.predict(X_test)
rf_conf_matrix = confusion_matrix(y_test, rf_predicted)
rf_acc_score = accuracy_score(y_test, rf_predicted)
print("confussion matrix")
print(rf_conf_matrix)
print("\n")
print("Accuracy of Random Forest:",rf_acc_score*100,'\n')
print(classification_report(y_test,rf_predicted))

#******************************* DecisionTreeClassifier ***************************************#
m6 = 'DecisionTreeClassifier'
dt = DecisionTreeClassifier(criterion = 'entropy',random_state=0,max_depth = 6)
dt.fit(X_train, y_train)
dt_predicted = dt.predict(X_test)
dt_conf_matrix = confusion_matrix(y_test, dt_predicted)
dt_acc_score = accuracy_score(y_test, dt_predicted)
print("confussion matrix")
print(dt_conf_matrix)
print("\n")
print("Accuracy of DecisionTreeClassifier:",dt_acc_score*100,'\n')
print(classification_report(y_test,dt_predicted))

#******************************* SVC ***************************************#

svc =  SVC(kernel='rbf', C=2)
svc.fit(X_train, y_train)
svc_predicted = svc.predict(X_test)
svc_conf_matrix = confusion_matrix(y_test, svc_predicted)
svc_acc_score = accuracy_score(y_test, svc_predicted)
print("confussion matrix")
print(svc_conf_matrix)
print("\n")
print("Accuracy of Support Vector Classifier:",svc_acc_score*100,'\n')
print(classification_report(y_test,svc_predicted))

#******************************* KNN ***************************************#

m5 = 'K-NeighborsClassifier'
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
knn_predicted = knn.predict(X_test)
knn_conf_matrix = confusion_matrix(y_test, knn_predicted)
knn_acc_score = accuracy_score(y_test, knn_predicted)
print("confussion matrix")
print(knn_conf_matrix)
print("\n")
print("Accuracy of K-NeighborsClassifier:",knn_acc_score*100,'\n')
print(classification_report(y_test,knn_predicted))

#******************************* Model evaluation ***************************************#

model_ev = pd.DataFrame({'Model': ['Logistic Regression','Random Forest','K-Nearest Neighbour','Decision Tree','Support Vector Machine'], 'Accuracy': [lr_acc_score*100,
                    rf_acc_score*100,knn_acc_score*100,dt_acc_score*100,svc_acc_score*100]})
print(model_ev)

colors = ['red','green','blue','gold','silver']
plt.figure(figsize=(12,5))
plt.title("barplot Represent Accuracy of different models")
plt.xlabel("Accuracy %")
plt.ylabel("Algorithms")
bars = plt.bar(model_ev['Model'],model_ev['Accuracy'],color = colors)
# 在每个条形图上添加具体的准确率值
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}%', ha='center', va='bottom')

plt.show()

