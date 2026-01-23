import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path

def plot_npy_curves_from_folder(folder_path, output_dir=None, plot_individual=True, plot_combined=True):
    """
    读取文件夹下所有.npy文件并绘制曲线
    
    参数:
        folder_path: 包含.npy文件的文件夹路径
        output_dir: 输出图像的目录，如果为None则使用folder_path
        plot_individual: 是否绘制每个文件的单独曲线图
        plot_combined: 是否绘制所有曲线的组合图
    """
    # 确保文件夹存在
    if not os.path.exists(folder_path):
        print(f"错误: 文件夹 '{folder_path}' 不存在")
        return
    
    # 设置输出目录
    if output_dir is None:
        output_dir = folder_path
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有的.npy文件
    npy_files = glob.glob(os.path.join(folder_path, "*.npy"))
    
    if not npy_files:
        print(f"在文件夹 '{folder_path}' 中没有找到.npy文件")
        return
    
    print(f"找到 {len(npy_files)} 个.npy文件:")
    for i, file in enumerate(npy_files):
        print(f"  {i+1}. {os.path.basename(file)}")
    
    # 存储所有数据和文件名
    all_data = []
    file_names = []
    
    # 读取每个.npy文件
    for file_path in npy_files:
        try:
            data = np.load(file_path, allow_pickle=True)
            all_data.append(data)
            file_names.append(os.path.basename(file_path))
            print(f"已加载: {os.path.basename(file_path)}, 形状: {data.shape}, 类型: {data.dtype}")
        except Exception as e:
            print(f"加载文件 {file_path} 时出错: {e}")
    
    # 如果没有成功加载任何数据，则退出
    if not all_data:
        print("没有成功加载任何数据")
        return
    
    # 绘制每个文件的单独曲线图
    if plot_individual:
        print("\n正在绘制单独曲线图...")
        for i, (data, name) in enumerate(zip(all_data, file_names)):
            plt.figure(figsize=(12, 8))
            
            # 处理数据形状
            if data.ndim == 1:
                # 一维数组，使用索引作为x轴
                x = np.arange(len(data))
                y = data
                plt.plot(x, y, 'b-', linewidth=2, label='数据')
                plt.xlabel('索引', fontsize=12)
                plt.ylabel('值', fontsize=12)
                
            elif data.ndim == 2:
                # 二维数组
                if data.shape[1] >= 2:
                    # 假设第一列是x，第二列是y
                    x = data[:, 0]
                    y = data[:, 1]
                    plt.plot(x, y, 'b-', linewidth=2, label='数据')
                    plt.xlabel('X', fontsize=12)
                    plt.ylabel('Y', fontsize=12)
                    
                    # 如果有多于2列，绘制额外的列
                    if data.shape[1] > 2:
                        for col in range(2, min(data.shape[1], 6)):  # 最多绘制5列
                            plt.plot(x, data[:, col], alpha=0.7, linewidth=1.5, 
                                    label=f'列{col}')
                else:
                    # 只有一列，使用索引作为x轴
                    x = np.arange(len(data))
                    y = data[:, 0]
                    plt.plot(x, y, 'b-', linewidth=2, label='数据')
                    plt.xlabel('索引', fontsize=12)
                    plt.ylabel('值', fontsize=12)
                    
            elif data.ndim == 3:
                # 三维数组，处理为2D显示
                print(f"文件 {name} 是三维数组，形状: {data.shape}，将进行特殊处理")
                # 这里可以根据需要自定义三维数据的显示方式
                # 例如：显示第一个通道或平均值
                if len(data) > 0:
                    y = data[0].flatten()[:1000]  # 取前1000个点
                    x = np.arange(len(y))
                    plt.plot(x, y, 'b-', linewidth=2, label='数据（第一通道展平）')
                    plt.xlabel('索引', fontsize=12)
                    plt.ylabel('值', fontsize=12)
            else:
                print(f"文件 {name} 的维度 {data.ndim} 不受支持")
                continue
            
            # 设置图表属性
            plt.title(f'{name} - 曲线图', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # 添加统计信息
            stats_text = f"""
数据统计:
  最小值: {y.min():.6f}
  最大值: {y.max():.6f}
  平均值: {y.mean():.6f}
  标准差: {y.std():.6f}
  数据点: {len(y)}
  文件: {name}
            """
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            plt.text(0.02, 0.98, stats_text.strip(), transform=plt.gca().transAxes,
                    fontsize=9, verticalalignment='top', bbox=props)
            
            plt.tight_layout()
            
            # 保存图像
            output_name = os.path.splitext(name)[0] + '_curve.png'
            output_path = os.path.join(output_dir, output_name)
            plt.savefig(output_path, dpi=300)
            plt.close()
            
            print(f"  已保存: {output_name}")
    
    # 绘制所有曲线的组合图
    if plot_combined and len(all_data) > 1:
        print("\n正在绘制组合曲线图...")
        plt.figure(figsize=(14, 10))
        
        # 颜色和线型
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_data)))
        line_styles = ['-', '--', '-.', ':'] * 5
        
        for i, (data, name, color) in enumerate(zip(all_data, file_names, colors)):
            # 提取y数据
            if data.ndim == 1:
                y = data
                x = np.arange(len(y))
            elif data.ndim == 2 and data.shape[1] >= 2:
                x = data[:, 0]
                y = data[:, 1]
            elif data.ndim == 2:
                y = data[:, 0]
                x = np.arange(len(y))
            else:
                # 对于高维数据，尝试展平
                y = data.flatten()[:1000]  # 限制为前1000个点
                x = np.arange(len(y))
            
            # 对数据进行归一化（可选）
            y_normalized = (y - y.min()) / (y.max() - y.min() + 1e-10)
            
            # 绘制曲线
            line_style = line_styles[i % len(line_styles)]
            plt.plot(x[:min(1000, len(x))], y_normalized[:min(1000, len(y))], 
                    color=color, linestyle=line_style, linewidth=1.5, 
                    label=f'{name} (归一化)')
        
        plt.title('所有.npy文件的曲线（归一化）', fontsize=16)
        plt.xlabel('索引/时间', fontsize=12)
        plt.ylabel('归一化值', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 图例放在图表外部
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=10)
        
        plt.tight_layout()
        
        # 保存组合图
        output_path = os.path.join(output_dir, 'all_curves_combined.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"已保存组合图: {output_path}")
        
        # 绘制非归一化的组合图
        plt.figure(figsize=(14, 10))
        
        for i, (data, name, color) in enumerate(zip(all_data, file_names, colors)):
            # 提取y数据
            if data.ndim == 1:
                y = data
                x = np.arange(len(y))
            elif data.ndim == 2 and data.shape[1] >= 2:
                x = data[:, 0]
                y = data[:, 1]
            elif data.ndim == 2:
                y = data[:, 0]
                x = np.arange(len(y))
            else:
                y = data.flatten()[:1000]
                x = np.arange(len(y))
            
            # 绘制原始曲线
            line_style = line_styles[i % len(line_styles)]
            plt.plot(x[:min(1000, len(x))], y[:min(1000, len(y))], 
                    color=color, linestyle=line_style, linewidth=1.5, 
                    label=name)
        
        plt.title('所有.npy文件的曲线（原始值）', fontsize=16)
        plt.xlabel('索引/时间', fontsize=12)
        plt.ylabel('值', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 图例放在图表外部
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=10)
        
        plt.tight_layout()
        
        # 保存原始值组合图
        output_path = os.path.join(output_dir, 'all_curves_original.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"已保存原始值组合图: {output_path}")
    
    # 生成数据分析报告
    print("\n生成数据分析报告...")
    report_lines = [
        "=" * 60,
        ".npy文件数据分析报告",
        "=" * 60,
        f"文件夹路径: {folder_path}",
        f"找到的文件数量: {len(npy_files)}",
        f"成功加载的文件数量: {len(all_data)}",
        "",
        "文件详细信息:",
    ]
    
    for i, (name, data) in enumerate(zip(file_names, all_data)):
        report_lines.append(f"\n{i+1}. {name}:")
        report_lines.append(f"   形状: {data.shape}")
        report_lines.append(f"   数据类型: {data.dtype}")
        report_lines.append(f"   维度: {data.ndim}")
        
        # 计算统计信息
        if data.ndim == 1:
            flat_data = data
        else:
            flat_data = data.flatten()
            
        report_lines.append(f"   最小值: {flat_data.min():.6f}")
        report_lines.append(f"   最大值: {flat_data.max():.6f}")
        report_lines.append(f"   平均值: {flat_data.mean():.6f}")
        report_lines.append(f"   标准差: {flat_data.std():.6f}")
    
    report_lines.extend([
        "",
        "生成的文件:",
    ])
    
    if plot_individual:
        for name in file_names:
            base_name = os.path.splitext(name)[0]
            report_lines.append(f"   {base_name}_curve.png")
    
    if plot_combined and len(all_data) > 1:
        report_lines.extend([
            "   all_curves_combined.png",
            "   all_curves_original.png"
        ])
    
    report_lines.append("=" * 60)
    
    # 保存报告
    report_text = "\n".join(report_lines)
    report_path = os.path.join(output_dir, 'npy_data_analysis_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\n分析报告已保存到: {report_path}")
    print("\n所有曲线图已生成完成！")

def create_sample_npy_files(sample_folder, num_files=5):
    """
    创建示例.npy文件用于测试
    """
    os.makedirs(sample_folder, exist_ok=True)
    
    print(f"在 {sample_folder} 中创建示例.npy文件...")
    
    # 创建正弦波数据
    for i in range(num_files):
        # 生成x值
        x = np.linspace(0, 4*np.pi, 1000)
        
        # 生成不同的y值（正弦波，频率和振幅不同）
        frequency = 1 + i * 0.5
        amplitude = 1 + i * 0.2
        y = amplitude * np.sin(frequency * x)
        
        # 添加一些噪声
        noise = np.random.normal(0, 0.1, len(y))
        y += noise
        
        # 保存为二维数组（x和y）
        data = np.column_stack([x, y])
        
        # 保存文件
        file_path = os.path.join(sample_folder, f'sine_wave_{i+1}.npy')
        np.save(file_path, data)
        print(f"  创建: {os.path.basename(file_path)}")
    
    # 创建一维数组数据
    for i in range(num_files):
        # 生成随机walk数据
        steps = 500
        random_walk = np.cumsum(np.random.randn(steps) * 0.1)
        
        # 保存为一维数组
        file_path = os.path.join(sample_folder, f'random_walk_{i+1}.npy')
        np.save(file_path, random_walk)
        print(f"  创建: {os.path.basename(file_path)}")
    
    print(f"示例文件创建完成，共创建了 {num_files*2} 个.npy文件")

# 主函数
if __name__ == "__main__":
    import sys
    
    # 获取命令行参数或使用默认值
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        # 默认使用当前目录下的"npy_data"文件夹
        folder_path = "./Curve"
    
    # 如果文件夹不存在，创建示例数据
    if not os.path.exists(folder_path):
        print(f"文件夹 '{folder_path}' 不存在，正在创建示例数据...")
        create_sample_npy_files(folder_path, num_files=3)
    
    # 设置输出目录（与输入文件夹相同）
    output_dir = os.path.join(folder_path, "plots")
    
    # 绘制所有.npy文件的曲线
    plot_npy_curves_from_folder(
        folder_path=folder_path,
        output_dir=output_dir,
        plot_individual=True,
        plot_combined=True
    )
    
    print(f"\n所有图像已保存到: {output_dir}")