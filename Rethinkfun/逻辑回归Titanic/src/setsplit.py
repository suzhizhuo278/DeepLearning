import pandas as pd

df = pd.read_csv("..\\data\\train.csv")

# 随机打乱
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 划分1
val_df = df.iloc[:91]
train_df = df.iloc[91:]

train_df.to_csv("..\\data\\train_split.csv", index=False)
val_df.to_csv("..\\data\\validation_split.csv", index=False)


# 独热编码 操作方法是独热编码->算标准差获得初值->把独热编码注释掉重新生成
df = pd.get_dummies(df, columns=["Sex", "Embarked"], dtype=int)

# 划分2
val_df = df.iloc[:91]
train_df = df.iloc[91:]

train_df.to_csv("..\\data\\train_split_temp.csv", index=False)
val_df.to_csv("..\\data\\validation_split_temp.csv", index=False)

print("划分完成！训练集和验证集已保存。")