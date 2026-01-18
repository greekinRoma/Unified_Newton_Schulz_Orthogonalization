import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import math

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
output_dir = './output'
os.makedirs(output_dir, exist_ok=True)

# =============================
# CVPR风格设置
# =============================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "axes.grid": True,
    "grid.alpha": 0.3
})

# =============================
# 多项式模型
# =============================
class PolyModel(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.N = N
        # a_1,...,a_{N-1} 可训练
        self.a = nn.Parameter(torch.randn(N-1))
        
    def forward(self, x):
        y = x.clone()
        for k in range(1, self.N):
            y = y + torch.abs(self.a[k-1]) * x * (1 - x**2)**(2**(k-1))
        # 计算 b
        b = math.exp(0.5) * (2**(self.N/2) - 1 - torch.sum(torch.abs(self.a)))
        y = y + b * x * (1 - x**2)**(2**(self.N-1))
        return y

# =============================
# 数据
# =============================
x_data = torch.linspace(0, 1, 10000)
t_data = torch.ones_like(x_data)

# =============================
# 不同 N 对比
# =============================
N_list = [1, 5, 9, 10, 12, 13, 14, 15, 16]
# 鲜明离散颜色
color_list = ['b', 'g', 'm', 'c', 'orange', 'purple', 'brown', 'pink', 'olive', 'navy', 'teal', 'red']

plt.figure(figsize=(6,4))  # 论文图常用 6x4 英寸

for i, N in enumerate(N_list):
    model = PolyModel(N)
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.5)
    loss_fn = nn.MSELoss()
    
    # 训练
    num_epochs = 5000
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        y_pred = model(x_data)
        loss = loss_fn(y_pred, t_data)
        loss.backward()
        optimizer.step()
        scheduler.step()
    
    # 评估
    model.eval()
    with torch.no_grad():
        y_eval = model(x_data)
    print(y_eval[10])
    # 绘制曲线
    plt.plot(x_data.numpy(), y_eval.numpy(), color=color_list[i % len(color_list)], linewidth=1.8, label=f'N={N}')

# 目标函数
plt.plot(x_data.numpy(), np.ones_like(x_data), 'k--', label='Target y=1', linewidth=2)

plt.xlabel('x', fontsize=10)
plt.ylabel('y', fontsize=10)
plt.title('Polynomial Fitting for Different N', fontsize=11)
plt.xlim([0,1])
plt.ylim([0.,1.1])
plt.legend(fontsize=8, loc='lower right', frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'polynomial_N_optimized_CVPR_colors.pdf'), dpi=300)
plt.show()

print("CVPR-style plot with vivid colors saved to:", os.path.join(output_dir, 'polynomial_N_optimized_CVPR_colors.pdf'))