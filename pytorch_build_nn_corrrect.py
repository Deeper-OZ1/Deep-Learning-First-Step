import torch
import matplotlib.pyplot as plt
import numpy as np

x = torch.Tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.Tensor([[3.0], [6.0], [9.0], [12.0]])

class first_nn(torch.nn.Module):
    def __init__(self):
        super(first_nn, self).__init__()
        self.linear = torch.nn.Linear(1, 1)
    
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

# 有无平均的损失函数
criterion_mean = torch.nn.MSELoss(reduction='mean')  # 推荐使用reduction参数，size_average已过时
criterion_not_mean = torch.nn.MSELoss(reduction='sum')
criterion_list = [criterion_not_mean, criterion_mean]
criterion_names = ["MSE (sum)", "MSE (mean)"]  # 用于图表标题

# 优化器列表及名称
optimizer_classes = [
    torch.optim.Adagrad,
    torch.optim.Adam,
    torch.optim.Adamax,
    torch.optim.ASGD,
    torch.optim.RMSprop,
    torch.optim.Rprop,
    torch.optim.SGD
]
optimizer_names = [
    'Adagrad', 'Adam', 'Adamax', 'ASGD',
    'RMSprop', 'Rprop', 'SGD'
]

for criterion_idx, criterion in enumerate(criterion_list):
    # 为每个损失函数创建一个画布
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    fig.suptitle(f"Loss Function: {criterion_names[criterion_idx]}", fontsize=16)  # 总标题
    
    for i, optimizer_class in enumerate(optimizer_classes):
        # 关键修复：每次训练前重新初始化模型，确保公平比较
        model = first_nn()
        # 重新初始化优化器，使用新模型的参数
        optimizer = optimizer_class(model.parameters(), lr=0.01)
        
        loss_list = []  # 使用列表更高效
        for epoch in range(100):
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss_list.append(loss.item())  # 记录损失值
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # 绘制损失曲线（x轴应为epoch，而非0-1）
        axes[i].plot(range(100), loss_list)
        axes[i].set_title(optimizer_names[i])
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel('Loss')
        axes[i].grid(True, linestyle='--', alpha=0.7)
    
    # 处理第8个子图（空白）
    axes[7].axis('off')  # 隐藏未使用的子图
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为总标题留出空间
    plt.show()
