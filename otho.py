import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from train_NS import PZA
class WhiteningTestPlatform:
    def __init__(self):
        self.datasets = {}
        self.results = {}
        self.model = PZA()

    def generate_data(self, n_samples=1000, random_state=None):
        """生成多种类型的测试数据集"""
        np.random.seed(random_state)
        
        # 高斯分布数据
        gaussian = np.random.randn(n_samples, 2)
        self.datasets['Gaussian'] = gaussian
        
        #  correlated Gaussian
        corr_gaussian = np.dot(np.random.randn(n_samples, 2), 
                              np.array([[1, 0.8], [0.2, 1]]))
        self.datasets['Correlated Gaussian'] = corr_gaussian
        
        # 使用sklearn生成更多类型的数据
        blob_data, _ = make_blobs(n_samples=n_samples, centers=3, 
                                 cluster_std=1.5, random_state=random_state)
        self.datasets['Blobs'] = blob_data
        
        moon_data, _ = make_moons(n_samples=n_samples, noise=0.1, 
                                 random_state=random_state)
        self.datasets['Moons'] = moon_data
        
        circle_data, _ = make_circles(n_samples=n_samples, noise=0.05, 
                                     factor=0.5, random_state=random_state)
        self.datasets['Circles'] = circle_data
        
        # 非零均值数据
        non_zero_mean = gaussian + np.array([3, -2])
        self.datasets['Non-zero Mean'] = non_zero_mean
        
        return self.datasets
    
    def pca_whitening(self, X):
        """PCA白化"""
        # 中心化
        X_centered = X - np.mean(X, axis=0)
        
        # 计算协方差矩阵
        cov = np.cov(X_centered, rowvar=False)
        
        # 特征值分解
        eig_vals, eig_vecs = np.linalg.eigh(cov)
        
        # 白化矩阵
        epsilon = 1e-5  # 防止除以零
        whitening_matrix = np.dot(eig_vecs, np.diag(1.0 / np.sqrt(eig_vals + epsilon)))
        
        # 应用白化
        X_whitened = np.dot(X_centered, whitening_matrix)
        
        return X_whitened, whitening_matrix
    
    def zca_whitening(self, X):
        """ZCA白化（保持原始方向）"""
        # 中心化
        X_centered = X - np.mean(X, axis=0)
        
        # 计算协方差矩阵
        cov = np.cov(X_centered, rowvar=False)
        
        # 特征值分解
        eig_vals, eig_vecs = np.linalg.eigh(cov)
        
        # 白化矩阵
        epsilon = 1e-5
        whitening_matrix = np.dot(eig_vecs, np.diag(1.0 / np.sqrt(eig_vals + epsilon)))
        zca_matrix = np.dot(whitening_matrix, eig_vecs.T)
        
        # 应用ZCA白化
        X_zca = np.dot(X_centered, zca_matrix)
        
        return X_zca, zca_matrix
    
    def standardize(self, X):
        """标准化数据（零均值，单位方差）"""
        scaler = StandardScaler()
        return scaler.fit_transform(X)
    
    def evaluate_whitening(self, X, X_whitened, method_name):
        """评估白化效果"""
        # 计算协方差矩阵
        cov_original = np.cov(X.T)
        cov_whitened = np.cov(X_whitened.T)
        
        # 计算特征值
        eigvals_original = np.linalg.eigvals(cov_original)
        eigvals_whitened = np.linalg.eigvals(cov_whitened)
        
        # 计算与单位矩阵的差异
        identity_matrix = np.eye(cov_whitened.shape[0])
        frobenius_norm = np.linalg.norm(cov_whitened - identity_matrix, 'fro')
        
        # 计算均值
        mean_original = np.mean(X, axis=0)
        mean_whitened = np.mean(X_whitened, axis=0)
        
        results = {
            'method': method_name,
            'original_cov': cov_original,
            'whitened_cov': cov_whitened,
            'original_eigvals': eigvals_original,
            'whitened_eigvals': eigvals_whitened,
            'frobenius_norm': frobenius_norm,
            'original_mean': mean_original,
            'whitened_mean': mean_whitened
        }
        
        return results
    
    def run_tests(self, dataset_name, X):
        """在给定数据集上运行所有白化方法"""
        results = {}
        
        # 标准化
        X_standardized = self.standardize(X)
        results['Standardized'] = self.evaluate_whitening(X, X_standardized, 'Standardized')
        
        # PCA白化
        X_pca_whitened, _ = self.pca_whitening(X)
        results['PCA Whitening'] = self.evaluate_whitening(X, X_pca_whitened, 'PCA Whitening')
        
        # ZCA白化
        X_zca_whitened, _ = self.zca_whitening(X)
        results['ZCA Whitening'] = self.evaluate_whitening(X, X_zca_whitened, 'ZCA Whitening')
        
        self.results[dataset_name] = results
        return results
    
    def visualize_results(self, dataset_name, X, results):
        """可视化原始数据和白化后的数据"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Whitening Results: {dataset_name}', fontsize=16)
        
        # 原始数据
        axes[0, 0].scatter(X[:, 0], X[:, 1], alpha=0.6)
        axes[0, 0].set_title('Original Data')
        axes[0, 0].grid(True)
        
        # 标准化数据
        X_standardized = results['Standardized']['whitened_cov']
        axes[0, 1].scatter(X_standardized[:, 0], X_standardized[:, 1], alpha=0.6)
        axes[0, 1].set_title('Standardized Data')
        axes[0, 1].grid(True)
        
        # PCA白化数据
        X_pca = results['PCA Whitening']['whitened_cov']
        axes[1, 0].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
        axes[1, 0].set_title('PCA Whitened Data')
        axes[1, 0].grid(True)
        
        # ZCA白化数据
        X_zca = results['ZCA Whitening']['whitened_cov']
        axes[1, 1].scatter(X_zca[:, 0], X_zca[:, 1], alpha=0.6)
        axes[1, 1].set_title('ZCA Whitened Data')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # 打印评估结果
        print(f"\nEvaluation Results for {dataset_name}:")
        print("=" * 50)
        for method, result in results.items():
            print(f"\n{method}:")
            print(f"  Frobenius norm from identity: {result['frobenius_norm']:.6f}")
            print(f"  Whitened eigenvalues: {result['whitened_eigvals']}")
    
    def run_comprehensive_test(self, n_samples=1000, random_state=42):
        """运行全面的测试"""
        # 生成数据
        self.generate_data(n_samples, random_state)
        
        # 对每个数据集运行测试
        for name, data in self.datasets.items():
            print(f"\nTesting on {name} dataset...")
            results = self.run_tests(name, data)
            self.visualize_results(name, data, results)

# 使用示例
if __name__ == "__main__":
    # 创建测试平台
    test_platform = WhiteningTestPlatform()
    
    # 运行全面测试
    test_platform.run_comprehensive_test(n_samples=1000, random_state=42)
    
    # 也可以单独测试某个数据集
    # test_platform.generate_data()
    # data = test_platform.datasets['Correlated Gaussian']
    # results = test_platform.run_tests('Correlated Gaussian', data)
    # test_platform.visualize_results('Correlated Gaussian', data, results)