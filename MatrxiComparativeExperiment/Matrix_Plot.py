from ast import mod
import os
import time
import torch
import numpy as np
import torch
from torch.profiler import profile, ProfilerActivity
from NS import OriginNS
from Muon import MuonNS
from Cesista import CesistaNS
from CANS import CaNS
from Our import PolyModel
# =====================================================
# 0. 全局配置
# =====================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "./matrices"
LOG_DIR = "./logs"
DTYPE = torch.float32

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# =====================================================
# 1. Matrix Bank（多矩阵 + H×W）
# =====================================================
class MatrixBank:
    def __init__(self, save_dir, device, dtype=torch.float32):
        self.save_dir = save_dir
        self.device = device
        self.dtype = dtype
        os.makedirs(save_dir, exist_ok=True)
        
    def get(self, name, H, W, idx, singular_values=None):
        """
        获取或生成矩阵
        
        参数:
        - name: 矩阵名称
        - H: 行数
        - W: 列数
        - idx: 索引
        - singular_values: 指定奇异值，如果为None则随机生成
        """
        path = f"{self.save_dir}/{name}_{H}x{W}_{idx}.pt"
        
        if os.path.exists(path):
            A = torch.load(path, map_location=self.device)
        else:
            if singular_values is None:
                # 默认：生成在(0, +∞)的奇异值
                singular_values = self._generate_singular_values(min(H, W))
            
            # 构造具有指定奇异值的矩阵
            A = self._construct_matrix_from_singular_values(H, W, singular_values)
            torch.save(A.cpu(), path)
            
        return A
    
    def _generate_singular_values(self, k, distribution='log_uniform', **kwargs):
        """
        生成在(0, +∞)范围内的奇异值
        
        参数:
        - k: 奇异值个数
        - distribution: 分布类型
          'log_uniform': 对数均匀分布，在[1e-6, 1e6]范围
          'exponential': 指数分布
          'half_normal': 半正态分布
          'gamma': Gamma分布
        """
        if distribution == 'log_uniform':
            # 在对数空间均匀分布，覆盖多个数量级
            min_val = kwargs.get('min_val', 1)
            max_val = kwargs.get('max_val', 10)
            log_min = torch.log(torch.tensor(min_val, device=self.device))
            log_max = torch.log(torch.tensor(max_val, device=self.device))
            
            # 在对数空间均匀采样，然后取指数
            log_values = torch.rand(k, device=self.device) * (log_max - log_min) + log_min
            return torch.exp(log_values)
            
        elif distribution == 'exponential':
            # 指数分布，参数λ控制尺度
            lambd = kwargs.get('lambd', 1.0)
            return torch.distributions.Exponential(lambd).sample((k,)).to(self.device)
            
        elif distribution == 'half_normal':
            # 半正态分布
            scale = kwargs.get('scale', 1.0)
            normal_samples = torch.randn(k, device=self.device) * scale
            return torch.abs(normal_samples)
            
        elif distribution == 'gamma':
            # Gamma分布
            shape = kwargs.get('shape', 2.0)
            scale = kwargs.get('scale', 1.0)
            return torch.distributions.Gamma(shape, scale).sample((k,)).to(self.device)
            
        elif distribution == 'power_law':
            # 幂律分布
            alpha = kwargs.get('alpha', 2.5)
            min_val = kwargs.get('min_val', 1e-6)
            max_val = kwargs.get('max_val', 1e6)
            
            # 从均匀分布转换到幂律分布
            u = torch.rand(k, device=self.device)
            if alpha != 1:
                values = ((max_val**(1-alpha) - min_val**(1-alpha)) * u + min_val**(1-alpha))**(1/(1-alpha))
            else:
                values = min_val * (max_val/min_val)**u
            return values
            
        else:
            raise ValueError(f"不支持的分布类型: {distribution}")
    
    def _construct_matrix_from_singular_values(self, H, W, singular_values):
        """
        从奇异值构造矩阵
        
        参数:
        - H: 行数
        - W: 列数
        - singular_values: 奇异值向量
        """
        k = min(H, W)
        
        if len(singular_values) != k:
            raise ValueError(f"奇异值数量应为{min(H, W)}，但得到{len(singular_values)}")
        
        # 确保所有奇异值都是正的
        singular_values = torch.abs(singular_values) + 1e-10  # 防止零值
        
        # 创建奇异值对角矩阵
        if H >= W:
            # Σ是H×W矩阵，左上角k×k对角块为奇异值
            S = torch.zeros(H, W, device=self.device, dtype=self.dtype)
            S[:k, :k] = torch.diag(singular_values)
        else:
            # Σ是H×W矩阵，左上角k×k对角块为奇异值
            S = torch.zeros(H, W, device=self.device, dtype=self.dtype)
            S[:k, :k] = torch.diag(singular_values)
        
        # 生成随机正交矩阵U和V
        # 方法1: 使用QR分解生成随机正交矩阵
        U = torch.qr(torch.randn(H, H, device=self.device))[0]
        V = torch.qr(torch.randn(W, W, device=self.device))[0]
        
        # 方法2: 使用SVD生成随机正交矩阵（更稳定）
        # U, _, _ = torch.linalg.svd(torch.randn(H, H, device=self.device))
        # V, _, _ = torch.linalg.svd(torch.randn(W, W, device=self.device))
        
        # 构造矩阵: A = U @ Σ @ V^T
        A = U @ S @ V.T
        
        # 验证奇异值是否正确
        computed_singular_values = torch.linalg.svdvals(A)
        
        # 检查奇异值是否匹配（允许微小误差）
        # if not torch.allclose(computed_singular_values[:k], singular_values, rtol=1e-4):
        #     print(f"警告: 构造的矩阵奇异值与目标有差异")
        #     print(f"目标奇异值: {singular_values[:10]}")
        #     print(f"实际奇异值: {computed_singular_values[:10]}")
        
        return A
    
    def get_matrix_with_condition_number(self, name, H, W, idx, condition_number):
        """
        获取具有特定条件数的矩阵
        
        参数:
        - condition_number: 条件数 = σ_max / σ_min
        """
        k = min(H, W)
        
        # 生成几何级数的奇异值
        sigma_max = 1.0  # 最大奇异值设为1
        sigma_min = sigma_max / condition_number
        
        if k > 1:
            # 在log空间均匀分布
            log_sigma_max = torch.log(torch.tensor(sigma_max, device=self.device))
            log_sigma_min = torch.log(torch.tensor(sigma_min, device=self.device))
            
            # 生成几何级数
            exponents = torch.linspace(0, 1, k, device=self.device)
            log_values = log_sigma_max - exponents * (log_sigma_max - log_sigma_min)
            singular_values = torch.exp(log_values)
        else:
            singular_values = torch.tensor([sigma_max], device=self.device)
        
        return self.get(name, H, W, idx, singular_values=singular_values)
    
    def get_random_matrix(self, name, H, W, idx, distribution='normal'):
        """
        获取随机矩阵（作为对比）
        """
        path = f"{self.save_dir}/{name}_{H}x{W}_{idx}.pt"
        
        if os.path.exists(path):
            A = torch.load(path, map_location=self.device)
        else:
            if distribution == 'normal':
                A = torch.randn(H, W, device=self.device, dtype=self.dtype)
            elif distribution == 'uniform':
                A = torch.rand(H, W, device=self.device, dtype=self.dtype) * 2 - 1  # [-1, 1]
            else:
                raise ValueError(f"不支持的分布: {distribution}")
            
            torch.save(A.cpu(), path)
        
        return A


# =====================================================
# 2. Newton–Schulz（可替换）
# =====================================================
def newton_schulz(A, T):
    H, W = A.shape
    if H != W:
        raise ValueError("NS requires square matrix")

    I = torch.eye(H, device=A.device, dtype=A.dtype)
    normA = torch.linalg.norm(A, ord=1) * torch.linalg.norm(A, ord=float("inf"))
    X = A.T / normA

    for _ in range(T):
        X = X @ (2 * I - A @ X)

    return X


# =====================================================
# 3. 相似度（AX ≈ I）
# =====================================================
def similarity_to_identity(A, X):
    N = A.shape[0]
    I = torch.eye(N, device=A.device, dtype=A.dtype)
    err = torch.linalg.norm(X @ X.T - I, ord="fro")
    return err


# =====================================================
# 4. FLOPs（统一，不乘样本数）
# =====================================================
def estimate_flops_torch(model, A):
    """
    使用 torch.profiler 统计 func(A, T) 的 FLOPs

    func: 你的 NS 函数
    A   : 输入矩阵 (torch.Tensor)
    T   : 迭代次数
    """
    A = A / torch.norm(A @ A.T, p=2).sqrt()
    activities = [ProfilerActivity.CPU]
    if A.is_cuda:
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        with_flops=True,
        record_shapes=False
    ) as prof:
        model.forward(A)

    total_flops = 0
    for evt in prof.key_averages():
        if evt.flops is not None:
            total_flops += evt.flops

    return total_flops


# =====================================================
# 5. Benchmark（按尺寸统一统计）
# =====================================================
def run_ns_benchmark(model, matrix_specs):
    bank = MatrixBank(SAVE_DIR, DEVICE)
    results = []

    for spec in matrix_specs:
        name, H, W, num = spec["name"], spec["H"], spec["W"], spec["num"]

        times, errors = [], []

        for i in range(num):
            A = bank.get(name, H, W, i)
            X = A / torch.norm(A @ A.T, p=2).sqrt()
            torch.cuda.synchronize() if DEVICE == "cuda" else None
            t0 = time.time()
            X = model.forward(X)
            t1 = time.time()
            torch.cuda.synchronize() if DEVICE == "cuda" else None

            times.append(t1 - t0)
            errors.append(similarity_to_identity(A, X))

        result = {
            "name": name,
            "shape": f"{H}x{W}",
            "num": num,
            "avg_time": float(torch.mean(torch.tensor(times))),
            "avg_error": float(torch.mean(torch.tensor(errors))),
            "flops": estimate_flops_torch(model, A),
        }
        results.append(result)

    return results


# =====================================================
# 6. 日志保存
# =====================================================
def save_results_txt(results, T):
    filename = f"NS_T{T}_{DEVICE}.txt"
    path = os.path.join(LOG_DIR, filename)

    with open(path, "w") as f:
        f.write("Newton–Schulz Benchmark Results\n")
        f.write(f"Device      : {DEVICE}\n")
        f.write(f"Iterations  : {T}\n\n")
        f.write("Name   Shape     Num   AvgTime(s)   AvgError     FLOPs\n")
        f.write("-" * 60 + "\n")

        for r in results:
            line = (
                f"{r['name']:<6} "
                f"{r['shape']:<9} "
                f"{r['num']:<5d} "
                f"{r['avg_time']:<12.6f} "
                f"{r['avg_error']:<12.3e} "
                f"{r['flops']:.3e}\n"
            )
            f.write(line)

    print(f"\n[Saved] Results written to {path}")


# =====================================================
# 7. 使用示例
# =====================================================
if __name__ == "__main__":

    matrix_specs = [
        {"name": "A", "H": 128, "W": 128, "num": 1000},
        {"name": "B", "H": 128, "W": 512, "num": 1000},
        {"name": "C", "H": 128, "W": 1024, "num": 1000},
    ]
    model = OriginNS(iter_num=8)
    # model = MuonNS(iter_num=5)
    # model = CesistaNS(iter_num=5)
    # model = CaNS()
    # model = PolyModel(N=14, auto_load=True)
    results = run_ns_benchmark(model, matrix_specs)
    save_results_txt(results, T=5)

    print("\n========== Summary ==========")
    for r in results:
        print(
            f"{r['name']} {r['shape']} | "
            f"err={r['avg_error']:.3e} "
            f"FLOPs={r['flops']:.3e}"
        )
    print("=============================")
