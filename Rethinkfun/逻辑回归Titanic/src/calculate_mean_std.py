import pandas as pd

df = pd.read_csv(r"..\data\train_split_temp.csv")  # 训练集

all_features = ["Pclass", "Age", "SibSp", "Parch", "Fare",
            "Sex_female", "Sex_male",
            "Embarked_C", "Embarked_Q", "Embarked_S"]

mean_dict = {}
std_dict = {}

for feature in all_features:
    mean_dict[feature] = df[feature].mean()
    std_dict[feature] = df[feature].std()

print("均值：", mean_dict)
print("标准差：", std_dict)






# 保存为 JSON，方便后续读取
import json
with open(r"..\data\stats.json", "w") as f:
    json.dump({"mean": mean_dict, "std": std_dict}, f)
