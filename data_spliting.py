import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("mental_health.csv")
print(df.head())  # 查看前几行

X = df.drop('label', axis=1)  # 所有特征列
y = df['label']               # 目标变量列

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,       # 20% 用作测试集
    random_state=42,     # 保证可复现
    stratify=y           # 如果是分类任务，这个可以保持类别比例
)

# 先划出临时训练集和测试集
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 再从临时训练集中划出验证集（20%验证 + 60%训练）
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)
# 最后比例为 60% train, 20% val, 20% test

print("Train:", y_train.value_counts(normalize=True))
print("Test :", y_test.value_counts(normalize=True))

# Combine features and target back together
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# Save to CSV
train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)

