import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from torchvision import datasets

epoch_size=10
batch_size=128

transform=transforms.Compose([
    transforms.ToTensor(), # 将PIL形式的图片转为张量形式，便于后一步处理
    transforms.Normalize((0.1037,),(0.3081,)) # 减去均值除以标准差
])

# 训练集准备
train_dataset=datasets.MNIST(root='./mnist/', train=True,
                             transform=transform, download=True)
train_loader=DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# 测试集准备
test_dataset=datasets.MNIST(root='./mnist/', train=False,
                             transform=transform, download=True)
test_loader=DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

# 神经网络类
class LogisticRegression(torch.nn.Module):
    def __init__(self):
        super(LogisticRegression,self).__init__()
        self.linear1=torch.nn.Linear(784,512)
        self.linear2=torch.nn.Linear(512,256)
        self.linear3=torch.nn.Linear(256,128)
        self.linear4=torch.nn.Linear(128,64)
        self.linear5=torch.nn.Linear(64,10)
        self.relu=torch.nn.ReLU()
    
    def forward(self,x):
        x=x.view(-1,784) # 此时的输入是(N,1,28,28),需要将高维张量变为二维矩阵
        x=self.relu(self.linear1(x))
        x=self.relu(self.linear2(x))
        x=self.relu(self.linear3(x))
        x=self.relu(self.linear4(x))
        return self.linear5(x)

model=LogisticRegression()
criterion=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.5)

def train(epoch):
    for batch_id, (x,y) in enumerate(train_loader):
        y_pred=model(x)
        loss=criterion(y_pred,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('eopch:',epoch,'loss:',loss.item())

def test():
    correct=0
    total=0
    with torch.no_grad():
        for x,y in test_loader:
            y_pred=model(x)
            _,pred_label=torch.max(y_pred,dim=1)
            total+=y_pred.shape[0]
            correct+=(pred_label==y).sum().item()
    print('accuracy:',correct/total)

if __name__ == '__main__':
    for epoch in range(epoch_size):
        train(epoch)
        test()