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
# 多项式模型（自动加载权重）
# =============================

# =============================
# 数据
# =============================
x_data = torch.linspace(0, 1., 10000)  # 归一化到[0,1]
t_data = torch.ones_like(x_data)

# =============================
# 训练函数
# =============================
def train_model(model, num_epochs=20000, lr=0.1, save_every=1000):
    """
    训练模型
    
    Args:
        model: 要训练的模型
        num_epochs: 训练轮数
        lr: 学习率
        save_every: 每隔多少轮保存一次权重
    
    Returns:
        losses: 损失历史
    """
    print(f"开始训练模型 N={model.N}...")
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.5)
    loss_fn = nn.MSELoss()
    
    losses = []
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        y_pred = model.get_curve(x_data)
        loss = loss_fn(y_pred, t_data)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % save_every == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")
            # 可选: 保存中间权重
            # model.save_weights()
    
    print(f"训练完成! 最终损失: {losses[-1]:.6f}")
    return losses

# =============================
# 评估模型
# =============================
def evaluate_model(model, x_eval=None):
    """
    评估模型
    
    Args:
        model: 要评估的模型
        x_eval: 评估点，如果为None则使用默认数据
    
    Returns:
        y_eval: 模型输出
        metrics: 评估指标字典
    """
    if x_eval is None:
        x_eval = x_data
        
    model.eval()
    with torch.no_grad():
        y_eval = model.get_curve(x_eval)
        
    # 计算评估指标
    mse = torch.mean((y_eval - t_data) ** 2).item()
    max_error = torch.max(torch.abs(y_eval - t_data)).item()
    
    metrics = {
        'mse': mse,
        'max_error': max_error,
        'y_at_0.1': y_eval[1000].item(),  # x=0.1
        'y_at_0.5': y_eval[5000].item(),  # x=0.5
        'y_at_1.0': y_eval[-1].item(),    # x=1.0
    }
    
    return y_eval, metrics

# =============================
# 可视化函数
# =============================
def plot_training_loss(losses, N):
    """绘制训练损失曲线"""
    plt.figure(figsize=(8, 4))
    plt.plot(losses, 'b-', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'Training Loss for N={N}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    loss_path = os.path.join(output_dir, f'training_loss_N_{N}.pdf')
    plt.savefig(loss_path, dpi=300)
    plt.close()
    print(f"训练损失图已保存到: {loss_path}")

def plot_model_prediction(model, x_eval=None):
    """绘制模型预测结果"""
    if x_eval is None:
        x_eval = x_data
        
    y_eval, metrics = evaluate_model(model, x_eval)
    
    plt.figure(figsize=(8, 6))
    plt.plot(x_eval.numpy(), y_eval.numpy(), 'b-', linewidth=2.5, label=f'Polynomial (N={model.N})')
    plt.plot(x_eval.numpy(), t_data.numpy(), 'r--', linewidth=2, label='Target y=1')
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.title(f'Polynomial Fitting with N={model.N}', fontsize=16)
    plt.xlim([0, 1])
    plt.ylim([0., 1.1])
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f'polynomial_N_{model.N}.pdf')
    plt.savefig(plot_path, dpi=300)
    plt.show()
    
    return plot_path, metrics

def plot_comparison(N_list, use_saved_weights=True):
    """
    生成多个N的对比图
    
    Args:
        N_list: N值列表
        use_saved_weights: 是否使用已保存的权重
    """
    plt.figure(figsize=(10, 6))
    
    # 颜色列表
    color_list = ['b', 'g', 'm', 'c', 'orange', 'purple', 'brown', 'pink', 'olive', 'navy', 'teal', 'red']
    
    for i, N in enumerate(N_list):
        # 创建模型（如果use_saved_weights=True，会自动加载权重）
        model = PolyModel(N, auto_load=use_saved_weights)
        
        # 评估模型
        model.eval()
        with torch.no_grad():
            y_eval = model(x_data)
        
        # 绘制曲线
        plt.plot(x_data.numpy(), y_eval.numpy(), 
                color=color_list[i % len(color_list)], 
                linewidth=2.0, 
                label=f'N={N}')
    
    # 目标函数
    plt.plot(x_data.numpy(), np.ones_like(x_data), 'k--', label='Target y=1', linewidth=2.5)
    
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Polynomial Fitting for Different N', fontsize=14)
    plt.xlim([0, 1])
    plt.ylim([0., 1.1])
    plt.legend(fontsize=10, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = 'polynomial_comparison_saved.pdf' if use_saved_weights else 'polynomial_comparison_retrained.pdf'
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.show()
    
    print(f"\n对比图已保存到: {save_path}")
    return save_path

# =============================
# 主程序
# =============================
def main():
    """主函数"""
    # 配置参数
    N_to_train = 14  # 要处理的N值
    retrain = False   # 是否强制重新训练
    
    print("=" * 50)
    print(f"多项式拟合演示 (N={N_to_train})")
    print("=" * 50)
    
    # 步骤1: 创建模型（自动加载权重如果存在）
    model = PolyModel(N_to_train, auto_load=(not retrain))
    # 步骤2: 如果需要重新训练，则训练模型
    if retrain or not os.path.exists(model.weight_path):
        print(f"\n开始训练模型 N={N_to_train}...")
        losses = train_model(model, num_epochs=5000)
        
        # 保存训练好的权重
        model.save_weights()
        
        # 绘制训练损失曲线
        plot_training_loss(losses, N_to_train)
    else:
        print(f"\n使用预训练模型 N={N_to_train}")
    
    # 步骤3: 评估模型
    print(f"\n评估模型 N={N_to_train}...")
    plot_path, metrics = plot_model_prediction(model)
    
    # 步骤4: 打印评估结果
    print(f"\n模型 N={N_to_train} 的评估结果:")
    print(f"  权重文件: {model.weight_path}")
    print(f"  y(0.1) = {metrics['y_at_0.1']:.6f}")
    print(f"  y(0.5) = {metrics['y_at_0.5']:.6f}")
    print(f"  y(1.0) = {metrics['y_at_1.0']:.6f}")
    print(f"  均方误差(MSE): {metrics['mse']:.6e}")
    print(f"  最大误差: {metrics['max_error']:.6f}")
    
    # 步骤5: 可选 - 生成对比图
    generate_comparison = False  # 设置为True以生成对比图
    if generate_comparison:
        print(f"\n生成对比图...")
        N_list = [1, 5, 10, 12, 14, 15]
        plot_comparison(N_list, use_saved_weights=True)
    
    print(f"\n处理完成！所有输出文件保存在: {output_dir}")

class PolyModel(nn.Module):
    def __init__(self, N, weight_path=None, auto_load=True, size= 128):
        """
        多项式模型类
        
        Args:
            N: 多项式阶数
            weight_path: 权重文件路径，如果为None则自动构建
            auto_load: 是否自动加载权重（如果文件存在）
        """
        super().__init__()
        self.N = N
        self.eye = torch.eye(size).cuda()
        # a_1,...,a_{N-1} 可训练
        self.a = nn.Parameter(torch.randn(N-1))
        self.b = nn.Parameter(torch.randn(1))
        self.c = None
        # 如果未指定权重路径，自动构建
        if weight_path is None:
            weight_path = os.path.join(output_dir, f'poly_model_N_{N}.pth')
        self.weight_path = weight_path
        
        # 自动加载权重（如果文件存在）
        if auto_load and os.path.exists(self.weight_path):
            self.load_weights(self.weight_path)
            self.c = math.exp(0.5) * (2**(self.N/2)-1) - torch.sum(torch.abs(self.a))
            print(f"模型 N={N}: 已从 '{self.weight_path}' 加载预训练权重")
        else:
            print(f"模型 N={N}: 使用随机初始化")
            train_model(self)  # 初始化时训练模型以获得合理权重
            self.save_weights(self.weight_path)
        
    
    
    def save_weights(self, path=None):
        """保存模型权重"""
        if path is None:
            path = self.weight_path
        self.a.data = torch.abs(self.a.clone().detach())
        torch.save({
            'N': self.N,
            'a': self.a.data,
            'b': self.b.data,
            'state_dict': self.state_dict()
        }, path)
        print(f"模型 N={self.N}: 权重已保存到 '{path}'")

    def load_weights(self, path=None):
        """加载模型权重"""
        if path is None:
            path = self.weight_path
            
        if not os.path.exists(path):
            print(f"警告: 权重文件 '{path}' 不存在")
            return False
            
        checkpoint = torch.load(path)
        
        # 检查N是否匹配
        if checkpoint['N'] != self.N:
            print(f"警告: 权重文件的N={checkpoint['N']}与模型N={self.N}不匹配")
            return False
        self.a.data = torch.abs(checkpoint['a'])
        self.b.data = checkpoint['b']      
        self.load_state_dict(checkpoint['state_dict'])
        return True
    
    def get_curve(self, x):

        y = x.clone()
        for k in range(1, self.N):
            y = y + torch.abs(self.a[k-1]) * x * (1 - x**2)**(2**(k-1))
        # 计算 b
        if self.c is None:
            c = math.exp(0.5) * (2**(self.N/2)-1) - torch.sum(torch.abs(self.a))
        else:
            c = self.c
        y = y + c * x * (1 - x**2)**(2**(self.N-1))
        return y
    
    
    def forward(self, X):
        print(torch.linalg.svd(X)[1])
        # S_only = np.linalg.svd(X.cpu().numpy(), compute_uv=False)
        # print("奇异值:", S_only)
        I = self.eye
        X_1 = I - X @ X.T
        X_2 = X_1 @ X_1
        X_3 = X_2 @ X_2
        X_4 = X_3 @ X_3
        X_5 = X_4 @ X_4
        X_6 = X_5 @ X_5
        X_7 = X_6 @ X_6
        X_8 = X_7 @ X_7
        X_9 = X_8 @ X_8
        X_10 = X_9 @ X_9
        X_11 = X_10 @ X_10
        X_12 = X_11 @ X_11
        X_13 = X_12 @ X_12
        X_14 = X_13 @ X_13
        Y = self.a[0] * X_1 + self.a[1] * X_2 + self.a[2] * X_3 + self.a[3] * X_4 + \
            self.a[4] * X_5 + self.a[5] * X_6 + self.a[6] * X_7 + self.a[7] * X_8 + \
            self.a[8] * X_9 + self.a[9] * X_10 + self.a[10] * X_11 + self.a[11] * X_12 + self.a[12] * X_13  + self.c * X_14 + I
        return Y @ X
    

if __name__ == "__main__":
    main()