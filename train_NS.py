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
    def __init__(self,kernel=4):
        super(NSLayer, self).__init__()
        self.weight = nn.Parameter(torch.tensor([ 4.9403e-01, -4.4625e-01,  8.2544e-01,  9.9267e-01,  1.6915e+00,
        -2.0835e+00,  2.9114e+00, -5.6555e+00,  3.5984e+00, -1.0205e+01,
         2.0828e+01, -2.2237e-03,  9.7854e-03,  1.5735e+02]), requires_grad=True)
        self.mask = nn.Parameter(torch.eye(kernel).reshape(1,1,kernel,kernel),requires_grad=False)
    def forward(self, input):
        A = self.mask - torch.matmul(input, torch.transpose(input, 2, 3))#**1
        B = torch.matmul(A, A)#**2
        C = torch.matmul(B, A)#**3
        D = torch.matmul(C, A)#**4
        E = torch.matmul(D, A)#**5
        F = torch.matmul(E, A)#**6
        G = torch.matmul(F, A)#**7
        # H = torch.matmul(G, A)#**128
        # I = torch.matmul(H, A)#**256
        # J = torch.matmul(I, A)#**512
        # K = torch.matmul(J, A)#**1024
        # L = torch.matmul(K, A)#**2048
        # M = torch.matmul(L, A)#**4096
        # N = torch.matmul(M, A)#**8192
        weight = self.weight
        Mat =  weight[0]*A + weight[1] * B + weight[2]*C +weight[3]*D+  weight[4] * E+  weight[5] * F + weight[6] * G + weight[7]* torch.eye(8)
        out = input + torch.matmul(Mat, input)
        return out
class PZA(nn.Module):
    def __init__(self, kernel =4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer = NSLayer(kernel=kernel)
    def forward(self, input):
        input = torch.nn.functional.normalize(input=input, dim=-1,p=2)
        out = self.layer(input/math.sqrt(kernel))
        return out

if __name__ == "__main__":
    # Example usage
    kernel = 8
    network = PZA(kernel=kernel)
    optimizer = torch.optim.Adam(network.parameters(), lr=50.)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = f"results_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    shutil.copy("train_NS.py", os.path.join(save_dir,"NS.py"))
    losses = []
    avg_losses = []
    mask = torch.eye(kernel, kernel,requires_grad=False).reshape(1,1,kernel,kernel)
    batch_size = 8
    for epoch in range(40000):
        x = torch.randn(batch_size,8,kernel,64)
        weight = torch.rand(batch_size, 8, kernel, kernel)
        x = torch.matmul(weight, x)
        optimizer.zero_grad()
        out = network(x)
        loss = torch.sum((torch.matmul(out, out.transpose(3, 2))-mask)**2)
        loss.backward()
        optimizer.step()
        scheduler.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()/batch_size}")
        losses.append(loss.item()/batch_size)
        with open(os.path.join(save_dir,"log.txt"),"a") as f:
            f.write(f"Epoch {epoch+1}, Loss: {loss.item()/batch_size}\n")
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
                print(f"Epoch {epoch+1}, Loss: {loss.item()}")
    print(network.layer.weight)
    x = torch.concat([torch.rand(batch_size//2, 8, 16, 64),torch.randn(batch_size//2,8,16,64)], dim=0)
    weight = torch.rand(batch_size, 8, 16, 16)
    x = torch.matmul(weight, x)
    out = network(x)
    torch.save(network.state_dict(), os.path.join(save_dir,"model.pth"))