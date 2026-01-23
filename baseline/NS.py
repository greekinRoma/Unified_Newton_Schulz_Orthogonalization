from re import M
import re
import time
from turtle import forward
from numpy import fix
from paramiko import ChannelException
from sympy import E
from torch import nn
import torch
import torch.nn.functional as F
import math
import shutil
import os
from matplotlib import pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
class NSLayer(nn.Module):
    def __init__(self, kernel, channel=8, bias=True):
        super(NSLayer, self).__init__()
        self.weight = nn.Parameter(torch.ones(1, 1, 1, 7)+0.5, requires_grad=True)
        self.mask = torch.eye(16, 16,requires_grad=False).reshape(1,1,16,16)
    def forward(self, input):
        A = self.mask - torch.matmul(input, torch.transpose(input, 2, 3))
        B = torch.matmul(A, A)
        C = torch.matmul(B, B)
        D = torch.matmul(C, C)
        E = torch.matmul(D, D)
        F = torch.matmul(E, E)
        G = torch.matmul(F, F)
        weight = self.weight
        Mat = weight[..., 0:1] * A + weight[..., 1:2] * B + weight[..., 2:3] * C + weight[..., 3:4] * D + weight[..., 4:5] * E + weight[..., 5:6] * F + weight[..., 6:7] * G
        out = input + torch.matmul(Mat, input)
        return out
class fixed_NSLayer(nn.Module):
    def __init__(self, channel=8,kernel=16):
        super(fixed_NSLayer, self).__init__()
        self.mask = torch.eye(kernel, kernel,requires_grad=False).reshape(1,1,kernel,kernel)
    def forward(self, input):
        A = torch.matmul(input, torch.transpose(input, 2, 3))
        B = torch.matmul(A, A)
        Mat = -1.5*A + 0.5*B
        out = 2.*input + torch.matmul(Mat, input)
        return out
class PZA(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale = nn.Parameter(torch.zeros(1, 1, 1, 1) + 0.36, requires_grad=True)
        self.layer = nn.Sequential(fixed_NSLayer(),
                                   fixed_NSLayer(),
                                   fixed_NSLayer(),
                                   fixed_NSLayer(),
                                   fixed_NSLayer())
    def forward(self, input):
        input = input-torch.mean(input,dim=-1,keepdim=True)
        input = torch.nn.functional.normalize(input=input, dim=-1,p=2)
        out = self.layer(input*self.scale)
        out = torch.nn.functional.normalize(input=out, dim=-1,p=2)
        return out

if __name__ == "__main__":
    # Example usage
    network = PZA()
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = f"results_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    shutil.copy("train_NS.py", os.path.join(save_dir,"NS.py"))
    losses = []
    avg_losses = []
    mask = torch.eye(16, 16,requires_grad=False).reshape(1,1,16,16)
    batch_size = 4
    for epoch in range(10000):
        x = torch.rand(batch_size, 8, 16, 64)
        weight = torch.rand(batch_size, 8, 16, 16)
        x = torch.matmul(weight, x)
        optimizer.zero_grad()
        out = network(x)
        loss = torch.sum((torch.matmul(out, out.transpose(3, 2))-mask)**2)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()/batch_size}")
        losses.append(loss.item())
        with open(os.path.join(save_dir,"log.txt"),"a") as f:
            f.write(f"Epoch {epoch+1}, Loss: {loss.item()}\n")
        if (epoch + 1) % 1000 == 0:
            avg_loss = sum(losses[-1000:]) / 1000
            avg_losses.append(avg_loss)
            print(f"Epoch {epoch+1}, Loss: {loss.item()/batch_size}, Avg Loss (last 1000): {avg_loss}")
            # 保存损失数据
            torch.save({
                'losses': losses,
                'avg_losses': avg_losses,
                'epoch': epoch
            }, os.path.join(save_dir, 'loss_data.pt'))
            
            # 保存模型
            torch.save({
                'model_state_dict': network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(save_dir, 'model_checkpoint.pt'))
            
            # 绘制损失曲线
            plt.figure(figsize=(10, 6))
            plt.plot(losses, alpha=0.3, label='Instant Loss')
            plt.plot(range(999, len(losses), 1000), avg_losses, label='Average Loss (per 1000 steps)', linewidth=2)
            plt.yscale('log')
            plt.xlabel('Epoch')
            plt.ylabel('Loss (log scale)')
            plt.title('Training Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, 'loss_plot.png'))
            plt.close()
        else:
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}, Loss: {loss.item()/batch_size}")
    
    torch.save(network.state_dict(), os.path.join(save_dir,"model.pth"))