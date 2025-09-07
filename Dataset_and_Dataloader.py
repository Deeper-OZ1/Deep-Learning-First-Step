import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset,DataLoader


epoch_size=500
batch_size=20

xy=np.loadtxt('breast_cancer.csv',delimiter=',',dtype=np.float32)

# 训练集准备
class BreastCancerTrain(Dataset):
    def __init__(self):
        self.x_data=torch.from_numpy(xy[0:559,:-1])
        self.y_data=torch.from_numpy(xy[0:559,[-1]])
        self.len=self.x_data.shape[0]      

    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]

    def __len__(self):
        return self.len

train_dataset=BreastCancerTrain()
train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)

# 测试集准备
x_test=torch.from_numpy(xy[560:568,:-1])
real_ans=xy[560:568,[-1]]

# 神经网络类
class LogisticRegression(torch.nn.Module):
    def __init__(self):
        super(LogisticRegression,self).__init__()
        self.linear1=torch.nn.Linear(30,15)
        self.linear2=torch.nn.Linear(15,7)
        # self.linear3=torch.nn.Linear(7,3)
        self.linear4=torch.nn.Linear(7,1)
        self.sigmoid=torch.nn.Sigmoid()
        self.relu=torch.nn.ReLU()
    
    def forward(self,x):
        x=self.relu(self.linear1(x))
        x=self.relu(self.linear2(x))
        # x=self.relu(self.linear3(x))
        x=self.sigmoid(self.linear4(x))
        return x

model=LogisticRegression()
criterion=torch.nn.BCELoss(reduction='mean')
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)

if __name__ == '__main__':
    for epoch in range(epoch_size):
        for batch_id,(x,y) in enumerate(train_loader):
            y_pred=model(x)
            loss=criterion(y_pred,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('eopch:',epoch,'batch_id:',batch_id,'loss:',loss.data)
    print('Successfully Trained')

    print('Predict after train:',model(x_test).data)
    print('Real Answer:',real_ans)