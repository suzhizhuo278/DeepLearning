#需要Active Recall
"""
目标是用逻辑回归解决titanic的幸存者分类问题
回忆一下由几部分组成？
1.数据集Dataloader与Dataset类，数据预处理可以比较简单，
数据集划分为训练集，验证集，测试集，如70：15：15
2.损失函数与优化器
采用MSE与SGD
3.模型主体nn.Module，调用Linear函数，这里存在一些奇特的地方，Linear函数可以提供多个输出
*额外可以尝试model.to("cuda")
*补充的一些训练深度神经网络的技巧，暂时用不到：
1.dropout
2.L1,L2正则化
3.权重衰减？
"""


"""
需要用到什么库？
1.pandas，用来读取数据
2.DataLoader and Dataset
3.torch.nn as nn
"""
import pandas as pd
from torch.utils.data import DataLoader,Dataset, random_split
import torch
import torch.nn as nn

"""
数据预处理
框架要求Dataset要求强制写入方法__len__以及__getitem__
"""
class TitanicDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.mean = {'Pclass': 2.31875, 'Age': 29.783181818181816, 'SibSp': 0.5325, 'Parch': 0.37375, 'Fare': 32.135837625, 'Sex_female': 0.34625, 'Sex_male': 0.65375, 'Embarked_C': 0.1825, 'Embarked_Q': 0.08375, 'Embarked_S': 0.73125}
        self.std = {'Pclass': 0.82946274663789, 'Age': 14.598016684331713, 'SibSp': 1.1305048879270803, 'Parch': 0.8045931581785489, 'Fare': 50.301038853287565, 'Sex_female': 0.47607167822030866, 'Sex_male': 0.47607167822030866, 'Embarked_C': 0.38649770451387105, 'Embarked_Q': 0.2771858175878396, 'Embarked_S': 0.44358696920565804}

        self.data = self._load_data()
        self.feature_size = len(self.data.columns) - 1
    def _load_data(self): #项目结构把数据预处理分为两步：文件处理+df处理、数据标准化
        df = pd.read_csv(self.file_path)
        df = df.drop(columns=["PassengerId", "Ticket", "Cabin"])

        #提取姓名特征并且填补缺失值
        df["Title"] = df["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)
        title_median_age = df.groupby("Title")["Age"].median()
        for title, median_age in title_median_age.items():  #在搜索空间里组合median_age与title
            df.loc[(df["Age"].isnull()) & (df["Title"] == title), "Age"] = median_age   #df.loc[行条件. 列条件]
        """
        另一种简单表达
                df["Age"] = df.groupby("Title")["Age"].transform(
                    lambda x: x.fillna(x.median())
                )
        """

        df = df.dropna(subset=["Age"])  ##删除Age有缺失的行
        # 注意！！！！一定要把多余字符串列用完删掉
        df = df.drop(columns=["Title", "Name"])

        df = pd.get_dummies(df, columns=["Sex", "Embarked"], dtype=int)  ##进行one-hot编码

        #最后一步，数据标准化并return df
        base_features =  ["Pclass", "Age", "SibSp", "Parch", "Fare"] #向量化操作
        for i in range(len(base_features)):
            df[base_features[i]] = (df[base_features[i]] - self.mean[base_features[i]])/self.std[base_features[i]]
        return df
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.data.drop(columns=["Survived"]).iloc[idx] #.iloc[idx]是按行按样本取
        label = self.data["Survived"].iloc[idx]
        #print(features, label)  # 调试看哪一项是字符串
        #原因：数据处理之后还保留了Name和Title两列字符串
        #需要将它们再df里删去，再返回tensor值给模型

        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
train_dataset = TitanicDataset("../data/train_split.csv")#上一个目录../
validation_dataset = TitanicDataset("../data/validation_split.csv")

##----以下代码为复制
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)  # nn.Linear也继承自nn.Module，输入为input_dim,输出一个值

    def forward(self, x):
        return torch.sigmoid(self.linear(x))  # Logistic Regression 输出概率


model = LogisticRegressionModel(train_dataset.feature_size)
model.to("cuda")
model.train()

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

epochs = 100

for epoch in range(epochs):
    correct = 0
    step = 0
    total_loss = 0
    for features, labels in DataLoader(train_dataset, batch_size=256, shuffle=True):
        step += 1
        features = features.to("cuda")
        labels = labels.to("cuda")
        optimizer.zero_grad()
        outputs = model(features).squeeze()
        correct += torch.sum(((outputs >= 0.5) == labels))
        loss = torch.nn.functional.binary_cross_entropy(outputs, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {total_loss/step:.4f}')
    print(f'Training Accuracy: {correct / len(train_dataset)}')

model.eval()
with torch.no_grad():
    correct = 0
    for features, labels in DataLoader(validation_dataset, batch_size=256):
        features = features.to("cuda")
        labels = labels.to("cuda")
        outputs = model(features).squeeze()
        correct += torch.sum(((outputs >= 0.5) == labels))
    print(f'Validation Accuracy: {correct / len(validation_dataset)}')