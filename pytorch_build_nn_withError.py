# 代码问题：在一个模型上不经清除已有参数就进行多次训练，模型被”污染”，结果不正确
import torch
import matplotlib.pyplot as plt
import numpy as np

x=torch.Tensor([[1.0],[2.0],[3.0],[4.0]])
y=torch.Tensor([[3.0],[6.0],[9.0],[12.0]])

class first_nn(torch.nn.Module):
    def __init__(self):
        super(first_nn,self).__init__()
        self.linear=torch.nn.Linear(1,1)
    
    def forward(self,x):
        y_pred=self.linear(x)
        return y_pred

model=first_nn()

# 有无平均
criterion_mean=torch.nn.MSELoss(size_average=True)
criterion_not_mean=torch.nn.MSELoss(size_average=False)
criterion_list=[criterion_not_mean,criterion_mean]

optimizer_Adagrad=torch.optim.Adagrad(model.parameters(),lr=0.01)
optimizer_Adam=torch.optim.Adam(model.parameters(),lr=0.01)
optimizer_Adamax=torch.optim.Adamax(model.parameters(),lr=0.01)
optimizer_ASGD=torch.optim.ASGD(model.parameters(),lr=0.01)
# optimizer_LBFGS=torch.optim.LBFGS(model.parameters(),lr=0.01)
optimizer_RMSprop=torch.optim.RMSprop(model.parameters(),lr=0.01)
optimizer_Rprop=torch.optim.Rprop(model.parameters(),lr=0.01)
optimizer_SGD=torch.optim.SGD(model.parameters(),lr=0.01)
optimizer_list=[optimizer_Adagrad,optimizer_Adam,optimizer_Adamax,optimizer_ASGD,
                optimizer_RMSprop,optimizer_Rprop,optimizer_SGD]

for criterion in criterion_list:
    # 创建一个画布和8个子图，这里使用2行4列的布局
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    # 将axes数组展平，方便遍历
    axes = axes.flatten()

    for i in range(len(optimizer_list)):
        optimizer=optimizer_list[i]
        print('otpimizer:',optimizer)
        loss_list=np.array([])
        for epoch in range(100):
            y_pred=model(x)
            loss=criterion(y_pred,y)
            loss_list=np.append(loss_list,loss.data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print('epoch:',epoch,'loss',loss.item())

        # with torch.no_grad():
        #     print(model.linear.weight.item())
        #     print(model.linear.bias.item())
        #     print(model(torch.Tensor([5.0])).data)

        # 生成X轴采样点
        x_dat = np.linspace(0, 1, 100)

        # 为每个子图绘制不同的内容
        if i==0:
            # axes[i].plot(x, np.sin(x + i/2))
            axes[i].set_title('Adagrad')
        elif i==1:
            # axes[i].plot(x, np.cos(x + i/2))
            axes[i].set_title('Adam')
        elif i==2:
            # axes[i].plot(x, np.cos(x + i/2))
            axes[i].set_title('Adamax')
        elif i==3:
            # axes[i].plot(x, np.cos(x + i/2))
            axes[i].set_title('ASGD')
        elif i==4:
            # axes[i].plot(x, np.cos(x + i/2))
            axes[i].set_title('RMSprop')
        elif i==5:
            # axes[i].plot(x, np.cos(x + i/2))
            axes[i].set_title('Rprop')
        elif i==6:
            # axes[i].plot(x, np.cos(x + i/2))
            axes[i].set_title('SGD')
        axes[7].axis('off')
        
        axes[i].plot(x_dat, loss_list)

        # 设置坐标轴标签
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('y')
        # 添加网格
        axes[i].grid(True, linestyle='--', alpha=0.7)

    # 调整子图之间的间距
    plt.tight_layout()

    # 显示图形
    plt.show()