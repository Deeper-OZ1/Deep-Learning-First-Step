import matplotlib as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import torch
import math

x_data=[1.0,2.0,3.0,4.0,5.0]
y_data=[math.sin(x) for x in x_data]

w1=torch.tensor([0.1],requires_grad=True)
w3=torch.tensor([0.01],requires_grad=True)
w5=torch.tensor([0.001],requires_grad=True)
w7=torch.tensor([0.0001],requires_grad=True)
learning_rate=0.0000000001

def forward(x):
    return w1*x+w3*x**3+w5*x**5+w7*x**7

def loss(x_data,y_data):
    loss=0
    for x,y in zip(x_data,y_data):
        x=torch.tensor(x)
        y=torch.tensor(y)
        y_pred=forward(x)
        loss+=(y-y_pred)**2
    return loss/len(x_data)

print('Predict before training:',forward(6.0).data)

if __name__ =='__main__':
    for epoch in range(500):
        l=loss(x_data,y_data)
        print('epoch:',epoch, 'loss:',l.item(), 'w1:',w1.data, 'w3:',w3.data, 'w5:',w5.data,'w7:',w7.data)
        l.backward()
        w1.data-=learning_rate*w1.grad.data
        w3.data-=learning_rate*w3.grad.data
        w5.data-=learning_rate*w5.grad.data
        w7.data-=learning_rate*w7.grad.data
        w1.grad.data.zero_()
        w3.grad.data.zero_()
        w5.grad.data.zero_()
        w7.grad.data.zero_()
    print('Predict after training:',forward(6.0).data)
    print('Real:',math.sin(6.0))

    



