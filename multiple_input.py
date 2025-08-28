import numpy as np
import matplotlib.pyplot as plt
import torch
# import torch.nn.functional as F

# 训练集数据
xy=np.loadtxt('breast_cancer.csv',delimiter=',',dtype=np.float32)
x=torch.from_numpy(xy[0:559,:-1])
y=torch.from_numpy(xy[0:559,[-1]])

class LogisticRegression(torch.nn.Module):
    def __init__(self):
        super(LogisticRegression,self).__init__()
        self.linear1=torch.nn.Linear(30,15)
        self.linear2=torch.nn.Linear(15,7)
        self.linear3=torch.nn.Linear(7,3)
        self.linear4=torch.nn.Linear(3,1)
        self.sigmoid=torch.nn.Sigmoid()
        self.relu=torch.nn.ReLU()
    
    def forward(self,x):
        x=self.relu(self.linear1(x))
        x=self.relu(self.linear2(x))
        x=self.relu(self.linear3(x))
        x=self.sigmoid(self.linear4(x))
        return x

model=LogisticRegression()
criterion=torch.nn.BCELoss(reduction='mean')
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)

# print('Predict before train:',model(torch.Tensor([6.0])))
# print('w:',model.linear.weight)
# print('b:',model.linear.bias)

for epoch in range(500):
    y_pred=model(x)
    loss=criterion(y_pred,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('eopch:',epoch,'loss:',loss.data)

# 测试集数据
x_test=torch.from_numpy(xy[560:568,:-1])

print('Predict after train:',model(x_test).data)
print('Real Data:',xy[560:568,[-1]])
# print('w:',model.linear.weight)
# print('b:',model.linear.bias)

# X=np.linspace(0,100,100)
# X_data=torch.Tensor(X).view(100,1)
# Y_data=model(X_data)
# Y=Y_data.data.numpy()

# plt.plot(X,Y)
# plt.xlabel('Hours')
# plt.ylabel('Possibility of Pass')
# plt.grid()
# plt.show()
