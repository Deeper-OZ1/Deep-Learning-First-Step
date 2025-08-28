import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# 逻辑斯蒂回归虽然是一个连续函数，但是解决的是分类问题！！

x=torch.Tensor([[1.0],[2.0],[3.0],[4.0],[5.0]])
y=torch.Tensor([[0.0],[0.0],[0.0],[1.0],[1.0]])

class LogisticRegression(torch.nn.Module):
    def __init__(self):
        super(LogisticRegression,self).__init__()
        self.linear=torch.nn.Linear(1,1)
    
    def forward(self,x):
        y_pred=F.sigmoid(self.linear(x))
        return y_pred

model=LogisticRegression()
criterion=torch.nn.BCELoss(size_average=False)
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)

print('Predict before train:',model(torch.Tensor([6.0])))
print('w:',model.linear.weight)
print('b:',model.linear.bias)

for epoch in range(200):
    y_pred=model(x)
    loss=criterion(y_pred,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('eopch:',epoch,'loss:',loss.data)

print('Predict after train:',model(torch.Tensor([6.0])))
print('w:',model.linear.weight)
print('b:',model.linear.bias)

X=np.linspace(0,100,100)
X_data=torch.Tensor(X).view(100,1)
Y_data=model(X_data)
Y=Y_data.data.numpy()

plt.plot(X,Y)
plt.xlabel('Hours')
plt.ylabel('Possibility of Pass')
plt.grid()
plt.show()
