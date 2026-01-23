import torch
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class MuonNS:
    def __init__(self, iter_num=5):
        self.iter_num = iter_num

    def newtonschulz5(self, X):
        """
        X: (N, N) torch.Tensor
        """
        a, b, c = 3.4445, -4.7750, 2.0315
        X_2 = X @ X.T
        A = X
        B =  X_2 @ X
        C = X_2 @ B
        return a * A + b *B + c * C

    def forward(self, X):
        for _ in range(self.iter_num):
            X = self.newtonschulz5(X)
        return X


if __name__ == "__main__":
    torch.manual_seed(0)

    # ==========================
    # 1. 构造矩阵输入
    # ==========================
    N = 32
    A = torch.randn(N, N)

    # 可选：归一化（NS 迭代常见操作，防止发散）
    A = A / torch.norm(A)

    # ==========================
    # 2. NS 迭代
    # ==========================
    model = OriginNSCurve(iter_num=8)
    Y = model.forward(A)

    print("Output shape:", Y.shape)

    # ==========================
    # 3. 可视化（示例：第一行）
    # ==========================
    plt.plot(Y[0].detach().cpu().numpy())
    plt.title("First row of NS output matrix")
    plt.show()
