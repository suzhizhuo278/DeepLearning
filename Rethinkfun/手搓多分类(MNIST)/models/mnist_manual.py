#准备手搓一个深度网络解决MNIST问题
from audioop import bias

import torch
from torch.utils.data import DataLoader, Dataset
#数据预处理,数据集已经用灰度值编码过
class MNISTDataset(Dataset):
    def __init__(self, file_path):
        self.images, self.labels = self._read_file(file_path)

    def _read_file(self, file_path):
        images = []
        labels = []
        with open(file_path, 'r') as f:
            next(f)
            for line in f:
                line = line.rstrip("\n")
                items = line.split(',')
                images.append([float(x) for x in items[1:]])
                labels.append(int(items[0]))
        return images, labels

    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = torch.tensor(image)
        image = image / 255.0
        image = (image - 0.1307) / 0.3105
        label = torch.tensor(label)
        return image, label
    def __len__(self): #__len__：获取样本数
        return len(self.images)

#定义训练集和测试集的dataloader
batch_size = 64
train_dataset = MNISTDataset(file_path='../data/mnist_train.csv')
test_dataset = MNISTDataset(file_path='../data/mnist_test.csv')
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


#定义神经网络
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
layer_sizes = [28*28, 128, 128, 128, 64, 10]
weights = []
biases = []
for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
    W = torch.randn(in_size, out_size, device=device) * torch.sqrt(torch.tensor(2 / in_size))
    b = torch.zeros(out_size, device=device)
    weights.append(W)
    biases.append(b)

#函数定义
def relu(x):
    return torch.clamp(x,min=0)

def relu_grad(x):
    return (x > 0).float()

def softmax(x):
    x_exp = torch.exp(x - x.max(dim=1, keepdim=True).values)
    return x_exp / x_exp.sum(dim=1, keepdim=True)

def cross_entropy(pred, labels):
    N = pred.shape[0]
    one_hot = torch.zeros_like(pred)
    one_hot[torch.arange(N), labels] = 1 #torch.arange(N)是生成0到N-1的整数张量
    loss = - (one_hot * torch.log(pred + 1e-8)).sum() / N #没有指定dim=1，原因是毫无影响，考虑一下熵的可求和性
    return loss, one_hot

#训练循环
learning_rate = 0.1
num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0
    for images, labels in train_loader:
        x = images.to(device)
        y = labels.to(device)
        N = x.shape[0]

        ## 前向传播
        activations = [x]
        pre_acts = []
        for W, b in zip(weights[:-1], biases[:-1]):
            z = activations[-1] @ W + b
            pre_acts.append(z)
            a = relu(z)
            activations.append(a)

        #输出层
        z_out = activations[-1] @ weights[-1] + biases[-1]
        pre_acts.append(z_out)
        y_pred = softmax(z_out)

        #损失
        loss, one_hot = cross_entropy(y_pred, y)
        total_loss += loss.item()

        ##反向传播
        grad_W = [None] * len(weights)
        grad_b = [None] * len(biases)
        #输出层梯度
        dL_dz = (y_pred - one_hot) / N
        grad_W[-1] = activations[-1].t() @ dL_dz
        grad_b[-1] = dL_dz.sum(dim=0)

        #隐层梯度
        for i in range(len(weights)-2,-1,-1):
            dL_dz = dL_dz @ weights[i+1].t() * relu_grad(pre_acts[i])
            grad_W[i] = activations[i].t() @ dL_dz
            grad_b[i] = dL_dz.sum(dim=0)

        #更新参数
        with torch.no_grad():
            for i in range(len(weights)):
                weights[i] -= learning_rate * grad_W[i]
                biases[i] -= learning_rate * grad_b[i]

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, loss: {avg_loss:.4f}")

#测试
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        x = images.view(-1, layer_sizes[0]).to(device)
        y = labels.to(device)
        a = x
        for W, b in zip(weights[:-1], biases[:-1]):
            a = relu(a @ W + b)
        logits = a @ weights[-1] + biases[-1]
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.shape[0]
    print(f"Test Accuracy: {correct / total * 100:.2f}%")